import json
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_scheduler,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#  CONFIG 
MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 64
EPOCHS     = 64
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  DATA LOADING 
def strip_non_ascii(text: str) -> str:
    """Remove non-ASCII characters but preserve newlines."""
    return re.sub(r"[^\x20-\x7E\n]", "", text)

def load_json(path: str):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)
    
def build_context(dialogue, tokenizer) -> str:
    lines = [t["text"] for t in dialogue if t["speaker"] == "seeker"]

    context_lines = []
    total_len = 0

    for line in reversed(lines):
        tokenized = tokenizer(line, add_special_tokens=True)
        length = len(tokenized["input_ids"])

        if total_len + length <= tokenizer.model_max_length:
            context_lines.insert(0, line)
            total_len += length
        else:
            break

    return strip_non_ascii("\n".join(context_lines))

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def build_training_rows(training_data):
    texts, labels = [], []
    for record in training_data:
        text=build_context(record['dialogue'],tokenizer)
        label = record["label"]  # now keep original 0-8 label
        texts.append(text)
        labels.append(label)
    return texts, labels

class MultiClassDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# TOKENIZE & SPLIT 
data = load_json("./training_with_items_added.json")
texts, labels = build_training_rows(data)

def tokenize(text_list):
    return tokenizer(
        text_list,
        truncation=True,
        padding=True
    )

train_dataset = MultiClassDataset(tokenize(texts), labels)
val_dataset   = MultiClassDataset(tokenize(texts), labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

#  MODEL 
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=9,  # multiclass
)
model.to(DEVICE)

# OPTIMIZER & SCHEDULER 
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)

# TRAINING LOOP
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        loss   = model(**batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels_batch = batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            probs = torch.softmax(model(**batch).logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels_batch.numpy())
    print(classification_report(all_labels, all_preds, digits=4))
    return all_preds

# TRAIN
print(f"Training on {DEVICE}\n")
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}  |  Loss: {train_loss:.4f}")

print("\n Validation ")
evaluate(model, val_loader)

# SAVE 
model.config.id2label = {i: str(i) for i in range(9)}
model.config.label2id = {str(i): i for i in range(9)}
model.save_pretrained("./evidence_classifier_multiclass")
tokenizer.save_pretrained("./evidence_classifier_multiclass")
print("\nModel saved to ./evidence_classifier_multiclass")

#  INFERENCE 
def predict_probabilities(text: str, model=model, tokenizer=tokenizer) -> np.ndarray:
    """
    Returns a vector of probabilities for labels 0-8.
    """
    model.eval()
    enc = tokenizer(
        strip_non_ascii(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu().numpy()
    return probs


if __name__ == "__main__":
    test_data = load_json("./test.json")
    for record in test_data:
        text=build_context(record['dialogue'],tokenizer)

        probs = predict_probabilities(text)
        print(f"Text: {text[:50]}...  |  Probabilities: {probs}")
