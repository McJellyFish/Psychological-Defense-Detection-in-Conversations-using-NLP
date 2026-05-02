import logging
from typing import Dict, List
from matplotlib import ticker
import torch
import os

import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, repeat
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import classification_reportW
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from scipy.stats import loguniform, uniform, randint
import RNNModel

#How many models to train at once
NUM_MODELS = 5
BASE_LR = 1e-3

EPOCHS = 250
#abort if improvement isn't seen within this many epoches
PATIENCE = 50
TOURNAMENT_EVERY = 1
DO_TOURNEY = True
SURVIVORS = 2
BATCH_SIZE = 512

logging.basicConfig(
    filename=f"bagged_xgboost_{NUM_MODELS}_{BASE_LR}_{EPOCHS}_{PATIENCE}.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s  - %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

def print_and_log(*objects, sep=' ', end='\n', file=None, flush=False):
    print(*objects, sep=sep, end=end, file=file, flush=flush)
    logger.info(sep.join(map(str, objects)) + end)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = False


MAX_TURNS = 50

#Data
train_data = json.load(open("train.json", "r", encoding="utf-8"))
val_data = json.load(open("./evals/test_label_full.json", "r", encoding="utf-8"))
print_and_log(len(train_data), len(val_data))

SENTENCE_MODEL_NAME = "google/embeddinggemma-300m"
st = SentenceTransformer(SENTENCE_MODEL_NAME)
st.eval()

speaker_map = {"seeker": 0, "supporter": 1}
device = "cuda" if torch.cuda.is_available() else "cpu"

#Embeddings
def precompute_embeddings(data, cache_file):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True) 
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings: {cache_file}")
        cached = torch.load(cache_file)
        for item, emb in zip(data, cached):
            item["st_embeddings"] = emb
        return data

    print(f"Generating embeddings: {cache_file}")
    all_embeddings = []
    for item in tqdm(data, desc="Encoding"):
        turns = item["dialogue"][:MAX_TURNS]
        texts = [t["text"] for t in turns]
        emb = st.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        item["st_embeddings"] = emb
        all_embeddings.append(emb.cpu())
    torch.save(all_embeddings, cache_file)
    return data

train_data = precompute_embeddings(train_data, "./rnn_model_outputs/train_embeddings.pt")
val_data   = precompute_embeddings(val_data,   "./rnn_model_outputs/val_embeddings.pt")

#Collate
def collate_fn(batch):
    emb_list, spk_list, labels = [], [], []
    st_dim = batch[0]["st_embeddings"].shape[-1]
    max_T = max(min(len(item["dialogue"]), MAX_TURNS) for item in batch)

    for item in batch:
        emb = item["st_embeddings"][:MAX_TURNS]
        T = emb.size(0)
        device = emb.device
        spk = torch.tensor(
            [speaker_map.get(t["speaker"], 0) for t in item["dialogue"][:MAX_TURNS]],
            dtype=torch.long, device=device,
        )
        if T < max_T:
            pad_emb = torch.zeros(max_T - T, st_dim, device=device, dtype=emb.dtype)
            pad_spk = torch.zeros(max_T - T, dtype=torch.long, device=device)
            emb = torch.cat([emb, pad_emb], dim=0)
            spk = torch.cat([spk, pad_spk], dim=0)
        emb_list.append(emb)
        spk_list.append(spk)
        labels.append(item["label"])

    return {
        "utterance_embeddings": torch.stack(emb_list).float(),
        "speaker_ids": torch.stack(spk_list),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

#Names
ROOTS = {
    "virtue":   ["fort", "virt", "piet", "iust", "fidel", "nobl", "clem", "sever"],
    "nature":   ["sol", "lun", "aquil", "silv", "flam", "vent", "terr", "cael", "mar"],
    "war":      ["bell", "mars", "milit", "pugn", "armat", "victor", "triumph"],
    "darkness": ["nox", "umbr", "tenebr", "mortem", "sanguin"],
    "power":    ["rex", "domin", "imper", "magn", "maxim", "august"],
    "animal":   ["aquil", "corv", "lupus", "leon", "taur", "drac"],
}
SUFFIXES = ["us", "ius", "anus", "inus", "atus", "ensis", "or", "ax", "ix", "ex", "ianus", "inus"]
CONNECTORS = ["", "i", "o", "a"]
LATIN_FIRST_NAMES = [
    "Marcus", "Lucius", "Gaius", "Tiberius", "Cassius", "Octavius", "Aurelius",
    "Julius", "Flavius", "Nero", "Maximus", "Severus", "Drusus", "Claudius",
    "Aelius", "Domitius", "Valerius", "Antonius", "Ignis", "Umbra", "Fortis",
    "Victoris", "Ferrum", "Noctis", "Sanguis", "Gladius", "Aeternus", "Rex",
    "Fatum", "Dominus", "Legionis", "Triumphus", "Mortis", "Caelus", "Quintus",
    "Publius", "Decimus", "Servius", "Vibius", "Manlius", "Cicero", "Scaevola",
    "Brutus", "Cato", "Pompeius", "Hadrianus", "Traianus", "Romulus", "Remus",
    "Vespasianus", "Germanicus", "Agrippa", "Varro", "Crassus", "Corvus", "Lupus",
    "Aquila", "Draco", "Leo", "Taurus", "Vulcanus", "Sol", "Luna", "Stella",
    "Tempestus", "Tonitrus", "Venator", "Bellator", "Invictus", "Titanus",
    "Praetor", "Centurio", "Imperius", "Magnus", "Altus", "Celer", "Durus",
    "Fidelis", "Justus", "Nobilis", "Pius", "Vindex", "Virtus", "Tenebris",
    "Lux", "Aurum", "Argentum", "Bronzus", "Silvanus", "Marinus", "Ventus",
    "Flamma", "Gelidus", "Terranus", "Orbis",
]

def generate_latin_name(compound: bool = None) -> str:
    compounds = max(2, min(4, random.randint(2, 4))) if compound else 1
    categories = random.choices(list(ROOTS.keys()), k=compounds)
    parts = [random.choice(ROOTS[cat]) for cat in categories]
    base = parts[0]
    for part in parts[1:]:
        base += random.choice(CONNECTORS) + part
    suffix = random.choice(SUFFIXES)
    for s in SUFFIXES:
        if base.endswith(s):
            base = base[:-len(s)]
            break
    return (base + suffix).capitalize()

#Models
models = [
    RNNModel.DialogueRNNModelOrig(num_labels=9, st=st, name=generate_latin_name(True)).to(device)
    for _ in range(NUM_MODELS)
]
optimizers = [torch.optim.Adam(m.parameters(), lr=BASE_LR) for m in models]

#Training
def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, data: List[Dict], train_for: int = 1):
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=False)
    model.train()
    total_loss, total_steps = 0, 0
    pbar = tqdm(chain.from_iterable(repeat(loader, train_for)), total=len(loader) * train_for, desc="Training")
    for batch in pbar:
        emb = batch["utterance_embeddings"].to(device)
        spk = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)
        loss = model(emb, spk, labels)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_steps += 1
        pbar.set_postfix(loss=f"{total_loss / total_steps:.4f}")
    return total_loss / total_steps

def evaluate_f1(data, batch_size=32, model=None):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            emb = batch["utterance_embeddings"].to(device)
            spk = batch["speaker_ids"].to(device)
            preds = torch.argmax(model(emb, spk)["logits"], dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch["labels"].numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (
        f1_score(y_true, y_pred, average="macro"),
        f1_score(y_true, y_pred, average="weighted"),
        f1_score(y_true, y_pred, average="micro"),
        f1_score(y_true, y_pred, average=None, labels=np.arange(9)),
    )

#Tournament
FREEDOM, EXILE, EXECUTION = "freedom", "exile", "execution"

def emperor_decree(favor):
    freedom_w = 1.0 + favor
    exile_w = 1.0
    execution_w = 1.0 + (1 - favor)
    total = freedom_w + exile_w + execution_w
    roll = random.random() * total
    if roll < freedom_w:
        return FREEDOM
    elif roll < freedom_w + exile_w:
        return EXILE
    return EXECUTION

def emperor_satisfaction(score_a, score_b):
    diff = abs(score_a - score_b)
    return max(0.0, min(1.0, min(1.0, diff * 5) + random.uniform(-0.1, 0.1)))

def model_clash_tournament(scores, models):
    gladiators = list(range(len(scores)))
    round_num = 1
    print_and_log("THE COLOSSEUM OPENS ITS GATES.")
    print_and_log(f"{len(gladiators)} GLADIATORS ENTER. ONLY {SURVIVORS} SHALL BE GRANTED VICTORY.\n")

    while len(gladiators) > SURVIVORS:
        random.shuffle(gladiators)
        print_and_log(f"=== TRIAL — ROUND {round_num} ===")
        print_and_log(f"Remaining gladiators: {[models[i].name for i in gladiators]}")
        next_round = []

        for i in range(0, len(gladiators), 2):
            remaining_unprocessed = len(gladiators) - i
            if len(next_round) + remaining_unprocessed <= SURVIVORS:
                next_round.extend(gladiators[i:])
                print_and_log("THE EMPEROR RAISES HIS HAND. Enough. The remaining gladiators are spared.\n")
                break
            if i + 1 == len(gladiators):
                print_and_log(f"Gladiator {models[gladiators[i]].name} advances (bye).")
                next_round.append(gladiators[i])
                continue

            a, b = gladiators[i], gladiators[i + 1]
            score_a, score_b = scores[a], scores[b]
            print_and_log(f"Duel: {models[a].name} (F1={score_a:.4f}) vs {models[b].name} (F1={score_b:.4f})")
            winner, loser = (a, b) if score_a >= score_b else (b, a)
            winner_score, loser_score = scores[winner], scores[loser]
            favor = emperor_satisfaction(winner_score, loser_score)
            decree = emperor_decree(favor)
            print_and_log(f"Victor: {models[winner].name} | Emperor's favor: {favor:.2f} | Decree: {decree}")
            next_round.append(winner)

        gladiators = next_round
        round_num += 1

    print_and_log(f"FINAL SURVIVORS: {[models[i].name for i in gladiators]}")
    return gladiators

#Plotting
def plot_f1_curves(train_history, val_history, train_macro, val_macro, num_classes=9):
    plt.style.use("dark_background")
    train_history = np.array(train_history)
    val_history = np.array(val_history)
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    for cls in range(num_classes):
        axes[0].plot(train_history[:, cls], linestyle="--", label=f"Level {cls}")
    axes[0].set_title("Train Per-Class F1 After Tourney", color="white")
    axes[0].set_ylabel("F1 Score", color="white")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8, facecolor="#111111", edgecolor="white")

    for cls in range(num_classes):
        axes[1].plot(val_history[:, cls], linestyle="-", label=f"Level {cls}")
    axes[1].set_title("Validation Per-Class F1", color="white")
    axes[1].set_ylabel("F1 Score", color="white")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize=8, facecolor="#111111", edgecolor="white")

    axes[2].plot(train_macro, linestyle="--", label="Train Macro F1")
    axes[2].plot(val_macro, linestyle="-", label="Val Macro F1")
    axes[2].set_title("Macro F1", color="white")
    axes[2].set_xlabel("Epoch", color="white")
    axes[2].set_ylabel("Macro F1", color="white")
    axes[2].grid(True, alpha=0.3)
    axes[2].yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    axes[2].legend(facecolor="#111111", edgecolor="white")

    fig.patch.set_facecolor("#0e0e0e")
    plt.tight_layout()
    plt.savefig("./graph.png")

#Loop
train_f1_history, val_f1_history = [], []
train_macro_history, val_macro_history = [], []
best_val_macro = -float("inf")
no_improve_count = 0

print_and_log("THE EMPEROR HAS DECREED THE TOURNAMENT'S START!")

for epoch in range(EPOCHS):
    print_and_log(f"\nEpoch {epoch + 1}")

    for i in range(NUM_MODELS):
        print_and_log(f"Training model {i}, {models[i].name}")
        train_epoch(models[i], optimizers[i], train_data, TOURNAMENT_EVERY)

    scores = []
    for i in range(NUM_MODELS):
        val_macro, _, _, _ = evaluate_f1(val_data, model=models[i])
        scores.append(val_macro)
        print_and_log(f"Model {i}, {models[i].name} val F1: {val_macro:.4f}")

    if DO_TOURNEY:
        winners = model_clash_tournament(scores, models)
        best_idx = max(winners, key=lambda i: scores[i])
        print_and_log("Survivors:", winners, "| Crowned:", best_idx)

        best_state = models[best_idx].state_dict()

        for i in range(NUM_MODELS):
            if i in winners:
                models[i].name = models[i].name + " " + random.choice(LATIN_FIRST_NAMES)
            else:
                models[i].load_state_dict(best_state)
                models[i].name = generate_latin_name(True)
                optimizers[i] = torch.optim.Adam(models[i].parameters(), lr=BASE_LR)

        train_macro, _, _, train_per_class = evaluate_f1(train_data, model=models[best_idx])
        val_macro, _, _, val_per_class = evaluate_f1(val_data, model=models[best_idx])

        train_f1_history.append(train_per_class)
        val_f1_history.append(val_per_class)
        train_macro_history.append(train_macro)
        val_macro_history.append(val_macro)
        plot_f1_curves(train_f1_history, val_f1_history, train_macro_history, val_macro_history)

        if val_macro > best_val_macro:
            torch.save(models[best_idx].state_dict(), "./rnn_model_outputs/dialogue_model_best.pt")
            print_and_log(f"New best: {models[best_idx].name} F1={val_macro:.4f}")
            best_val_macro = val_macro
            no_improve_count = 0
        else:
            no_improve_count += 1
            progress = no_improve_count / PATIENCE
            if progress <= 0.2:
                print_and_log("The emperor watches silently.")
            elif progress <= 0.4:
                print_and_log("The emperor begins to lean forward. The crowd grows uneasy.")
            elif progress <= 0.7:
                print_and_log("The emperor's favor wanes. Executions are whispered in the stands.")
            elif progress <= 0.9:
                print_and_log("The emperor is visibly displeased. Sand turns heavy with dread.")
            else:
                print_and_log("The emperor's patience is nearly exhausted. The arena trembles.")
            print_and_log(f"No-improve: {no_improve_count}/{PATIENCE}")

            if no_improve_count >= PATIENCE:
                print_and_log(f"THE EMPEROR HAS RUN OUT OF PATIENCE AT EPOCH {epoch + 1}. THE GAMES ARE FORSAKEN.")
                break

#Eval
model = models[best_idx]
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.load_state_dict(torch.load("./rnn_model_outputs/dialogue_model_best.pt"))

def evaluate_model(data, batch_size=32, model=None):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            emb = batch["utterance_embeddings"].to(device)
            spk = batch["speaker_ids"].to(device)
            preds = torch.argmax(model(emb, spk)["logits"], dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch["labels"].numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print_and_log(classification_report(y_true, y_pred, labels=np.arange(9), digits=4))

evaluate_model(train_data, 64, model=model)

#Features
def build_features(data, batch_size=8):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    X, y = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            out = model(
                utterance_embeddings=batch["utterance_embeddings"].to(device),
                speaker_ids=batch["speaker_ids"].to(device),
            )
            X.append(out["features"].cpu().numpy())
            y.append(batch["labels"].numpy())
    return np.vstack(X), np.concatenate(y)

print_and_log("Extracting train features...")
X_train, y_train = build_features(train_data, batch_size=64)
print_and_log("Extracting val features...")
X_val, y_val = build_features(val_data, batch_size=64)
print_and_log(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

#XGBoost
X = np.concatenate([X_train, X_val])
y = np.concatenate([y_train, y_val])
test_fold = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])
ps = PredefinedSplit(test_fold)

param_dist = {
    "max_depth": randint(4, 12),
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0, 1),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "colsample_bylevel": uniform(0.5, 0.5),
    "learning_rate": loguniform(0.005, 0.2),
    "reg_alpha": loguniform(1e-3, 10),
    "reg_lambda": loguniform(0.1, 10),
    "max_bin": [128, 256, 512],
}

xgb = XGBClassifier(
    n_estimators=200, objective="multi:softprob", num_class=9,
    eval_metric="mlogloss", tree_method="hist", early_stopping_rounds=50, verbose=False,
)
search = RandomizedSearchCV(
    estimator=xgb, param_distributions=param_dist, n_iter=200,
    cv=ps, scoring="f1_macro", n_jobs=1, verbose=2, refit=True,
)
search.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
print_and_log(search.best_params_, search.best_score_)

#Sweep
def evals():
    rows = []
    best_params = {k: v.item() if hasattr(v, "item") else v for k, v in search.best_params_.items()}
    for i in range(50, 501, 50):
        xgb = XGBClassifier(
            **best_params, n_estimators=i,
            objective="multi:softprob", num_class=9, eval_metric="mlogloss",
            tree_method="hist", grow_policy="lossguide",
            random_state=42, n_jobs=4, verbosity=0,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = xgb.predict(X_val)
        report = classification_report(y_val, preds, output_dict=True, zero_division=0)
        row = {
            "n_estimators": i,
            "accuracy": accuracy_score(y_val, preds),
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
        for label in sorted(set(y_val)):
            row[f"label_{label}_f1"] = report.get(str(label), {}).get("f1-score", 0)
        rows.append(row)
        print_and_log(f"Done: {i}", row)
    return pd.DataFrame(rows)

df_scores = evals()
print_and_log(df_scores)

#Plot
label_cols = [c for c in df_scores.columns if c.startswith("label_") and c.endswith("_f1")]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("#121212")
ax.set_facecolor("#1b1b1b")
colors = plt.cm.tab10(np.linspace(0, 1, max(len(label_cols), 1)))

for i, col in enumerate(label_cols):
    ax.plot(df_scores["n_estimators"], df_scores[col], marker="o", linewidth=2,
            color=colors[i], label=f"Label {col.replace('label_', '').replace('_f1', '')}")

ax.plot(df_scores["n_estimators"], df_scores["macro_f1"],    linestyle="--", linewidth=3, color="white", label="macro_f1")
ax.plot(df_scores["n_estimators"], df_scores["weighted_f1"], linestyle=":",  linewidth=3, color="gray",  label="weighted_f1")

ax.set_xlabel("n_estimators")
ax.set_ylabel("F1 Score")
ax.set_title("Per-Label F1 Scores vs Number of Estimators")
ax.set_xticks(df_scores["n_estimators"])
ax.tick_params(axis="x", rotation=45)
ax.grid(True, which="both", alpha=0.3)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()