# PsyDefDetect: Psychological Defense Mechanism Detection using NLP

## Overview
PsyDefDetect is a Natural Language Processing system designed to identify psychological defense mechanisms in emotional support conversations. Unlike traditional sentiment analysis, this project focuses on deeper clinical reasoning by classifying user responses based on the Defense Mechanism Rating Scale (DMRS).

The system combines large language models, embedding techniques, and hybrid reasoning pipelines to analyze multi-turn dialogue and accurately predict defense levels.

---

## Objective
The goal of this project is to build an intelligent system that:
- Analyzes multi-turn conversations
- Understands psychological and contextual signals
- Classifies defense mechanisms across 9 levels (0–8)
- Improves mental health NLP through structured reasoning

---

## Dataset
- Dataset: PsyDefConv (annotated ESConv dataset)
- Dialogues: 200
- Total utterances: 4,709
- Labeled seeker utterances: 2,336
- Classes: 9 (Defense Levels 0–8)

The dataset is highly imbalanced, with certain classes (especially level 7) dominating the distribution, which introduces challenges during training.

---

## System Architecture

### Two-Stage Hybrid Pipeline

1. Generative Model (Ministral 8B-Mental / Mappy)
   - Generates reasoning for predicted defense mechanisms
   - Outputs DMRS mechanisms along with supporting items

2. Structured Reasoning Model (Classifier / GPT-based model)
   - Validates generated reasoning
   - Assigns the final defense level

---

## Models and Approaches

### Transformer-Based Models
- ClinicalBERT
- DistilBERT
- Meta LLaMA 3
- Qwen3-32B (LoRA fine-tuned)

### Custom and Hybrid Models
- Ministral 8B-Mental (domain-specific model)
- Ministral 8B-Mappy (persona-based reasoning model)
- Embedding model based on MPNet with AnGLE loss

### Classical Machine Learning Baseline
- XGBoost with TF-IDF features
- XGBoost with transformer-based embeddings

---

## Key Techniques

- Context-aware dialogue modeling
- DMRS-Q item augmentation
- Handling class imbalance through resampling
- Embedding-based semantic learning
- Hybrid generative and classification pipeline

---

## Results

| Model                     | F1 Score |
|--------------------------|----------|
| Baseline LLM             | ~0.29    |
| Qwen3-32B                | ~0.34    |
| Ministral 8B-Mappy       | ~0.37    |
| XGBoost (Embeddings)     | ~0.20    |

The hybrid pipeline achieved the best overall performance, demonstrating the importance of combining reasoning with classification.

---

## Challenges

- Severe class imbalance in dataset
- Clinical complexity of defense level classification
- Need for full dialogue context understanding
- Sensitivity to input formatting
- Subtle differences between adjacent labels

---

## Key Insights

- Traditional classification approaches are insufficient for this task
- Structured reasoning significantly improves performance
- Domain-specific training is critical for mental health NLP
- Hybrid architectures outperform standalone models

---

## Future Work

- Improve model explainability and interpretability
- Address class imbalance more effectively
- Optimize computational efficiency
- Enhance domain-specific data augmentation techniques

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- XGBoost
- Sentence Transformers (MPNet)
- Large Language Models (Qwen, LLaMA, Ministral)

---

## Contributions

- Implemented multiple NLP pipelines for dialogue understanding
- Trained and evaluated transformer-based and classical models
- Designed a hybrid reasoning and classification architecture
- Performed evaluation using F1-score, precision, and recall metrics

---

## Keywords

Mental Health NLP, Dialogue Analysis, Large Language Models, Text Classification, Contextual Reasoning

---

## References

This project is based on the PsyDefConv dataset and the Defense Mechanism Rating Scale (DMRS) framework for psychological analysis.
