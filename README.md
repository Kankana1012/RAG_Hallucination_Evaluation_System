# RAG_Hallucination_Evaluation_System

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,50:F953C6,100:FF6B6B&height=200&section=header&text=🧠%20Answerability%20·%20Reliability%20·%20Hallucination&fontSize=28&fontColor=ffffff&fontAlignY=38&desc=Detection%20using%20RAG&descAlignY=58&descSize=18&descColor=ffffff&animation=fadeIn" width="100%"/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br/>

***A production-grade RAG system that knows what it doesn't know.***  
***Built to answer confidently when it can, abstain wisely when it can't, and flag hallucinations before they reach the user.***

</div>

---

## 🔍 What This Project Solves

Most RAG pipelines answer every question regardless of whether the retrieved context actually supports it. This project builds a **three-layer safety architecture** on top of RAG:

| Layer | Problem Solved |
|-------|----------------|
| 🎯 **Answerability Gate** | Decides *should I even answer this?* before generating anything |
| 🔗 **Claim-Level Grounding** | Checks if each sentence in the answer is supported by retrieved docs |
| ☣️ **Hallucination Risk Model** | Predicts how likely the generated answer contains fabricated content |

---

## 🏗️ System Architecture

```
User Question
      │
      ▼
┌─────────────────────────────┐
│   Hybrid Retriever          │  BM25 (lexical) + Dense Embeddings (semantic)
│   sentence-transformers     │  → Top-K document chunks
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│   Answerability Classifier  │  13 retrieval-based features
│   Random Forest + Isotonic  │  → Calibrated P(answerable)
│   Calibration               │
└─────────────────────────────┘
      │
      ├── P < 0.40 → ABSTAIN   ("Unable to answer...")
      ├── P < 0.60 → REQUEST MORE EVIDENCE
      └── P ≥ 0.60 → GENERATE ANSWER
                          │
                          ▼
              ┌───────────────────────┐
              │   flan-t5-large       │  Seq2Seq generation
              │   RAG prompt builder  │  Budget-aware context packing
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Claim-Level Support  │  Embedding similarity per sentence
              │  Grounding Check      │  → Unsupported claim rate
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Hallucination Risk   │  Trained on RAGTruth QA dataset
              │  Estimator            │  → P(hallucination)
              └───────────────────────┘
```

---

## ✨ Key Features

### 🔀 Hybrid Retrieval
- Combines **BM25 Okapi** (token-level precision) with **dense vector search** (semantic understanding)
- Configurable alpha blending: `hybrid_score = α × dense + (1 - α) × BM25`
- Chunked corpus with sliding window overlap to handle long technical documents
- Noise filtering for base64-encoded embedded content in IBM documentation

### 🧮 Answerability Classifier
- 13 handcrafted features: top/mean hybrid scores, BM25 scores, dense scores, lexical overlap ratios, score gap, question length, context coverage
- **Random Forest** (400 trees, balanced class weights) with **Isotonic Regression** post-hoc calibration
- Threshold tuned on validation data to constrain unsafe answer rate ≤ 20%

### 📝 RAG Answer Generation
- Generator: `google/flan-t5-large` (encoder-decoder, FP16 on GPU)
- Budget-aware prompt builder respects a 6000-char context window
- Beam search (n=4) with repetition penalty and no-repeat-ngram
- **Extractive fallback**: if the model refuses but the gate says "answer", the system extracts the most relevant sentences directly

### 🧩 Claim-Level Grounding
- Splits the generated answer into individual claims (sentences)
- Embeds each claim and each context chunk
- Flags any claim whose max cosine similarity to any context chunk falls below threshold (0.42)

### ☣️ Hallucination Risk Estimation
- Trained on the [RAGTruth](https://github.com/ParticleMedia/RAGTruth) dataset (real QA responses with human hallucination labels)
- Features: response/context length, token overlap, sentence count, digit density
- Outputs `P(hallucination)` per generated answer

---

## 📊 Evaluation Suite

The project runs a comprehensive, multi-metric evaluation:

| Metric | What It Measures |
|--------|-----------------|
| **Retrieval Recall@K** | Does the correct source document appear in top-K results? |
| **Answerability AUROC** | How well does the classifier separate answerable from unanswerable? |
| **Unsafe Answer Rate** | How often does the system answer when it should abstain? |
| **False Abstention Rate** | How often does the system refuse to answer a valid question? |
| **ROUGE-L** | Lexical overlap between generated and gold answers |
| **BERTScore (F1)** | Semantic similarity between generated and gold answers |
| **LLM Judge (A–E)** | Factuality, completeness, decision appropriateness, overall quality |
| **ECE / Brier Score** | Calibration quality of the answerability probability |
| **Unsupported Claim Rate** | Fraction of answer sentences not grounded in retrieved context |

### 🔬 Ablation Study

Five answerability gate policies are compared head-to-head:

| Policy | Threshold |
|--------|-----------|
| Full policy (default) | answer ≥ 0.60, request ≥ 0.40 |
| Conservative gate | answer ≥ 0.70, request ≥ 0.50 |
| Aggressive gate | answer ≥ 0.50, request ≥ 0.30 |
| Binary gate | answer or abstain only |
| No gate | always answer |

---

## 📦 Tech Stack

| Component | Library / Model |
|-----------|----------------|
| 🗄️ **Dataset** | ![NVIDIA](https://img.shields.io/badge/NVIDIA-TechQA--RAG--Eval-76B900?style=flat-square&logo=nvidia&logoColor=white) ![RAGTruth](https://img.shields.io/badge/ParticleMedia-RAGTruth-FF6B6B?style=flat-square) |
| 🔍 **Sparse Retrieval** | ![BM25](https://img.shields.io/badge/rank__bm25-BM25Okapi-4A90E2?style=flat-square&logo=searchengin&logoColor=white) |
| 🧲 **Dense Retrieval** | ![SentenceTransformers](https://img.shields.io/badge/sentence--transformers-all--MiniLM--L6--v2-FFD21E?style=flat-square&logo=huggingface&logoColor=black) |
| 🤖 **Generator** | ![Flan-T5](https://img.shields.io/badge/Google-flan--t5--large-4285F4?style=flat-square&logo=google&logoColor=white) |
| 🌲 **Answerability Clf** | ![sklearn](https://img.shields.io/badge/scikit--learn-RandomForest%20%2B%20Isotonic-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) |
| 📐 **Text Quality** | ![ROUGE](https://img.shields.io/badge/rouge--score-ROUGE--L-6C63FF?style=flat-square) ![BERTScore](https://img.shields.io/badge/bert--score-DeBERTa--base--MNLI-F953C6?style=flat-square) |
| ☣️ **Hallucination Data** | ![RAGTruth](https://img.shields.io/badge/RAGTruth-Human%20Labelled%20QA-FF4444?style=flat-square&logo=databricks&logoColor=white) |
| 📊 **Visualization** | ![Matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=flat-square&logo=python&logoColor=white) ![Seaborn](https://img.shields.io/badge/seaborn-4C8CBF?style=flat-square&logo=python&logoColor=white) |

---

## 🗄️ Datasets

| Dataset | Source | Used For |
|---------|--------|----------|
| 🟢 **TechQA-RAG-Eval** | [![HuggingFace](https://img.shields.io/badge/HuggingFace-nvidia%2FTechQA--RAG--Eval-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval) | Primary QA benchmark — answerable & unanswerable IBM technical questions with retrieved contexts |
| 🔴 **RAGTruth** | [![GitHub](https://img.shields.io/badge/GitHub-ParticleMedia%2FRAGTruth-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ParticleMedia/RAGTruth) | Hallucination detection — human-labelled RAG responses with span-level hallucination annotations |

---

## 🚀 Getting Started

### Installation

```bash
pip install datasets transformers accelerate sentence-transformers rank_bm25 \
            scikit-learn pandas numpy tqdm rouge-score seaborn matplotlib bert-score
```

### Quick Run

```python
# After running all setup cells, use the full pipeline in one call:
out = answerability_aware_rag_with_risk("How do I configure SSL mutual authentication in IBM HTTP Server?")

print(out["decision"])           # "answer" | "abstain" | "request_more_evidence"
print(out["prob_answerable"])    # e.g. 0.84
print(out["answer"])             # Generated technical answer
print(out["hallucination_risk"]) # e.g. 0.23
print(out["support"]["unsupported_rate"])  # Fraction of unsupported claims
```

### Configuration

Key parameters in **Cell 3**:

```python
MAX_EVAL_EXAMPLES = 80         # Evaluation sample size
TOP_K = 5                      # Retrieved chunks per query
HYBRID_ALPHA = 0.60            # Dense vs BM25 weight
CHUNK_WORDS = 360              # Chunk size (words)
GEN_MODEL_NAME = "google/flan-t5-large"
ANSWER_THRESHOLD = 0.60        # Min P(answerable) to generate
REQUEST_MORE_EVIDENCE_THRESHOLD = 0.40
```

---

## 📁 Notebook Structure

| Cell | Description |
|------|-------------|
| 1–2 | Install packages, imports, GPU setup |
| 3 | Runtime configuration |
| 4–6 | Load TechQA dataset, clean contexts, train/val/test split |
| 7–9 | Build corpus, BM25 index, dense embeddings |
| 10–11 | Hybrid retrieval function + retrieval evaluation |
| 12–13 | Answerability feature extraction + classifier training |
| 14 | Decision policy tuning (threshold optimization) |
| 15–16 | Load generator model + RAG prompt builder |
| 17 | Claim-level grounding analysis |
| 18 | Full system prediction pipeline |
| 19–20 | End-to-end evaluation + error analysis |
| 21–23b | RAGTruth hallucination dataset + model + visualizations |
| 24 | Integrate hallucination risk into system outputs |
| 25 | Single-question interactive demo |
| 26–27 | Performance metrics matrix + visualizations |
| 28 | BERTScore evaluation |
| 29 | LLM-as-Judge evaluation (4 dimensions) |
| 30 | Uncertainty calibration + reliability diagrams |
| 31 | Ablation study over gating policies |

---

## 🎯 Design Philosophy

> *"A RAG system that hallucinates confidently is more dangerous than one that abstains honestly."*

This project treats **refusing to answer** as a first-class outcome, not a failure mode. The answerability gate is tuned to prioritize safety (low unsafe answer rate) over coverage (low false abstention rate). Every generated answer is post-hoc audited at the claim level, giving users a grounded uncertainty signal rather than a single yes/no.

---

## 📄 License

MIT License — see `LICENSE` for details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF6B6B,50:F953C6,100:6C63FF&height=120&section=footer&text=⭐%20If%20this%20project%20helped%20you%2C%20consider%20starring%20the%20repo!&fontSize=16&fontColor=ffffff&fontAlignY=65&animation=fadeIn" width="100%"/>

</div>
