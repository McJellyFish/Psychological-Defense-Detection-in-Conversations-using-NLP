# ============================================================
# DMRS-Q Inference Pipeline
# ============================================================
# Classifies patient utterances from counselling dialogues
# into Defense Mechanism Rating Scale (DMRS-Q) levels using:
#   - A DistilBERT multi-class classifier
#   - A RAG memory system for item/mechanism retrieval
#   - A local LLM (gpt-oss) for structured reasoning
#   - Dr. Mappy (fine-tuned Ministral-8B) for defense narration
# ============================================================

#  Imports 

import csv
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict

import ollama
import pandas as pd

from mappyutils.dr_mappy_output_eval import (
    parse_paragraph,
    post_process,
)

#  Type aliases 

Record = Dict[str, Any]

#  Logging 

logging.basicConfig(
    filename="inferenceNotebook.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - [%(filename)s] - %(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


#  Utility helpers 


def print_and_log(*objects, sep=" ", end="\n", file=None, flush=False) -> None:
    """Print *objects* to stdout and also write the same message to the logger."""
    print(*objects, sep=sep, end=end, file=file, flush=flush)
    message = sep.join(map(str, objects)) + end
    logger.info(message)

def load_json(path: str) -> list:
    """Load a JSON file and return its contents, or an empty list on ``FileNotFoundError``."""
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_json(path: str, data: dict) -> None:
    """Serialise *data* to *path* as indented UTF-8 JSON."""
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)




#  Prompt / annotation constants 

CORE_PRINCIPLES = """
***Core Annotation Principles***
**Primacy of context**: Assign a level while considering the current label AND the preceding dialogue.
**Function-Oriented Principle**: Ask what the utterance achieves for the speaker in relation to stress or conflict.
**Emotion is not defense**: Pure feeling statements are not defenses unless there is clear avoidance, distortion, or transformation.
**Acknowledge mature coping**: Actively look for constructive coping methods that are grounded in an acknowledgment of reality.
"""

EVIDENCE_BLOCK = """
***Evidence and Rationale***
    •    Quote only what is necessary. Prefer short spans over paraphrase.
    •    Ground each claim in the target utterance or its immediate context.
    •    Avoid inferences about stable traits unless the dialogue explicitly supports them.
    •    When evidence is weak, state why. Apply a low confidence score.
"""

DISAMBIG_BLOCK = """
***Disambiguation Rules***
    •    **Multiple signals in one utterance** — Choose the dominant function. If two are truly inseparable, prefer the lower maturity level or mark 8 when evidence is insufficient.
    •    **Mature coping vs phatic closings** — Simple thanks, greetings, and farewells are label 0. Rich reflections on received help may support a mature level if explicitly stated.
    •    **Negative talk about others** — Require signs of self-esteem protection or blame shifting. Factual criticism alone is not a defense.
    •    **Intellectual style** — Distinguish descriptive or analytic language from efforts to keep feelings at a distance. The latter supports an obsessional level when grounded in text.
    •    **Across time requirements** — Signals that demand longitudinal evidence are rarely observable in short dialogues. Do not infer them without explicit cues in the current context.
"""

VALID_JSON_FORMAT = """{
    "level": (integer),
    "reasoning": (string),
    "evidence": (string),
    "confidence_score": (float)
}"""


#  DMRS-Q level definitions 

def get_dmrsq_levels() -> str:
    """
    Return a formatted description of all DMRS-Q defense levels (0–8).

    This is exposed as an LLM tool so the model can look up level definitions
    before making a classification.

    Returns:
        Multi-line string describing each defense level and its associated mechanisms.
    """
    return """\
Available Defence levels (Defense Mechanism Rating Scale tiers):
1 = Action Defense Level (Acting out / Help-rejecting complaining / Passive aggression)
  - The person perceives their distress as entirely caused by outside forces and, unable to contain it, acts impulsively on the world or themselves to release tension — without thinking through the consequences.
2 = Major Image-distorting Defense Level (Splitting of self-image / Splitting of object's image / Projective identification)
  - To cope with intense anxiety, the person sees themselves and others in extreme all-or-nothing terms (all good or all bad), which temporarily reduces inner threat but pushes others away.
3 = Disavowal Defense Level (Denial / Projection / Rationalization / Autistic fantasy)
  - The person refuses to acknowledge uncomfortable parts of reality or their own role in a problem, often blaming something external, which prevents them from taking any steps to fix it.
4 = Minor Image-distorting Defense Level (Devaluation of self-image / Devaluation of other's image / Idealization of self-image / Idealization of other's image / Omnipotence)
  - When facing failure, criticism, or shame, the person slightly distorts reality to protect their self-esteem, though this doesn't actually help them deal with the problem.
5 = Neurotic Defense Level (Displacement / Dissociation / Reaction formation / Repression)
  - The person can feel the emotions tied to a problem but unconsciously blocks out the actual thought or wish driving them, leaking it out in distorted, indirect ways.
6 = Obsessional Defense Level (Intellectualization / Isolation of affects / Undoing)
  - The person stays intellectually aware of a problem but emotionally detaches from it, expressing feelings only indirectly through minimizing, over-generalizing, or contradicting themselves.
7 = Highly Adaptive Defense Level (Affiliation / Altruism / Anticipation / Humor / Self-assertion / Self-observation / Sublimation / Suppression)
  - The healthiest coping style, where a person honestly faces their stressors, takes full personal responsibility, and seeks help when needed to get the best possible outcome.
8 = Needs more evidence.
  - There is insufficient context to determine the defense mechanism. This label should be used sparingly.
"""


#  Context helpers 

#  Dr. Mappy narration helpers 


def write_to_log(record, prompt, response, LOG_FILE="dr_mappy_log_new.csv"):
    """
    Writes a log entry to a specified CSV file.
    Parameters:
        record (dict): A dictionary containing log entry details, expected to have an 'id' key.
        prompt (str): The input prompt that was processed.
        response (dict): A dictionary containing the response, expected to have a 'message' key with a 'content' subkey.
        LOG_FILE (str, optional): The name of the log file. Defaults to "dr_mappy_log_new.csv".
    The function checks if the log file exists; if not, it creates the file and writes the headers.
    Each log entry includes the record ID, input prompt, output response content, and the current Unix timestamp.
    """
    # Create CSV if missing, with headers
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["ID", "Input", "Output", "unix_timestamp"])

        writer.writerow(
            [
                record.get("id", ""),
                prompt,
                response["message"]["content"],
                int(time.time()),
            ]
        )


def dr_mappy_evaluates_a_record(record: Record):
    """
    Ask the Dr. Mappy fine-tuned model to narrate which DMRS defenses apply
    to the seeker in *record*, logging the interaction to ``dr_mappy_log_new.csv``.

    Args:
        record: A dialogue record containing ``"dialogue"`` and ``"id"`` keys.

    Returns:
        The model's free-text response describing detected defenses.
    """
    SYSTEM_PROMPT = (
        "You are a mental health analysis assistant trained to analyze mental state based on text posts. "
        "Evaluate the text post provided by the user, answer their question, "
        "and then provide an evidence-based explanation in the voice of the eccentric 'Dr. Dennis Mappy,'"
        "a relaxed, fun-loving, and wise doctor of psychology whom specializes in explaining his logic in a fun way that is still simple and informative."
        "You may ONLY use regular ascii characters, no emojis.  "
    )

    ctx = []
    for turn in record["dialogue"]:
        if turn["speaker"] == "seeker":
            ctx.append(turn["text"])
        else:
            ctx.append("[REDACTED]")
    fullt = "\n".join(ctx)

    prompt = f"""Consider this chat log: ```{fullt}``` Question: What DMRS defenses apply to the seeker, prioritizing the log at the end?"""

    for attempt in range(8):  # Retry up to 8 times
        try:
            response = ollama.chat(
                model="hf.co/CrosswaveOmega/ministral8b-drmappy-mental-lora-unquantized-Q8_0-GGUF:Q8_0",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.4,
                },
            )
            write_to_log(record, prompt, response)
            return response["message"]["content"]
        except Exception as e:
            if attempt < 7:  # If not the last attempt, log the error and retry
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
            else:
                logger.error(f"All attempts failed for record {record['id']}: {e}")
                raise  # Re-raise the last exception



#  Main pipeline 

# Expected label distribution in the test set (derived from paper counts)
ESTIMATED_LABELS_IN_TESTSET: Dict[int, int] = {
    0: 75,   # No Defense
    1: 28,   # Action Level
    2: 16,   # Major Image-Distorting
    3: 25,   # Disavowal
    4: 21,   # Minor Image-Distorting
    5: 13,   # Neurotic
    6: 44,   # Obsessional
    7: 243,  # Highly Adaptive
    8: 7,    # Needs More Information
}

#  Load data 
with open("./test.json", "r") as file:
    data = json.load(file)

#  Run everything 

# Test results where generated by the average result from 10 trials.
# It's shortened to two for now
for i in range(0,2):#10):
    estimated_labels_remaining = dict(ESTIMATED_LABELS_IN_TESTSET)
    outputs: dict = {}

    for item in data:
        idval = item["id"]

        if idval not in outputs:
            orig_label = -1


            mappy_result = dr_mappy_evaluates_a_record(item)
            print(idval, mappy_result)

            pout = parse_paragraph(mappy_result)
            post = post_process([pout])
            print(post)

            matched_label = next(iter(post["label_presence_all"]), None)
            output_val = {"label": matched_label}

            if matched_label is not None and estimated_labels_remaining[matched_label] <= 0:
                print(f"MISMATCH: label {matched_label} is out of tokens!")
            if matched_label is not None:
                estimated_labels_remaining[matched_label] -= 1

            outputs[idval] = (item, output_val)

            with open("without_llm_outputs_nex.json", "w", encoding="utf8") as f:
                json.dump(outputs, f, indent=4, ensure_ascii=False)
        else:
            _, output_val = outputs[idval]
            output_label = output_val["label"]
            if output_label is not None and estimated_labels_remaining[output_label] <= 0:
                print(f"MISMATCH: label {output_label} is out of tokens!")
            if output_label is not None:
                estimated_labels_remaining[output_label] -= 1

    print(estimated_labels_remaining)

    #  Tally and write final predictions 
    allcounts: Counter = Counter()
    nextval: list = []

    for idval, (orig, new) in outputs.items():
        allcounts[new["label"]] += 1
        nex = orig.copy()
        nex["label"] = new["label"]
        nextval.append(nex)

    print(allcounts)
    logger.info(allcounts)

    with open("without_llm_outputs_nex.json", "w", encoding="utf8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

    with open(f"prediction_{i}.json", "w", encoding="utf8") as f:
        json.dump(nextval, f, indent=4, ensure_ascii=False)

# Preform final evaluation.
df = pd.read_csv("./dr_mappy_log_new.csv")

grouped = defaultdict(list)

for row in df.to_dict(orient="records"):
    grouped[row["ID"]].append(row)

mappy_analysis = {}

for rid, rows in grouped.items():
    parsed = [parse_paragraph(r["Output"]) for r in rows]
    post = post_process(parsed)

    mappy_analysis[rid] = {
        "analysis": post
    }

averaged = []

for rid, v in mappy_analysis.items():
    label_scores = defaultdict(float)

    for pair in v["analysis"]["label_pairs"]:
        for lbl in pair:
            label_scores[lbl] += 1 / len(pair)

    pred = max(label_scores.items(), key=lambda x: x[1])[0] if label_scores else 7

    averaged.append({
        "id": rid,
        "label": pred
    })
