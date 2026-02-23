# CoT-RAG LLM Gene Prioritization & Disease Diagnosis

This repository provides inference pipelines for **rare disease diagnosis** and **gene prioritization** from clinical narratives using large language models (LLMs).  
It implements **Chain-of-Thought (CoT)** prompting, **Retrieval-Augmented Generation (RAG)**, and two hybrid variants (**CoT→RAG** and **RAG→CoT**), plus an **AutoEvaluator** module for systematic evaluation.

> Note on data: synthetic clinical notes in this project are generated using GPT-4 based on Phenopacket-Store records (see the associated manuscript / project description in your work).

---

## What’s included

### Inference pipelines (in `main_script/`)
- **CoT (disease diagnosis)**: multi-step reasoning prompt ending with `POTENTIAL_DISEASES` (**exactly 10** items, single quotes).
  - `main_script/RareDxGPT_inference_CoT.py`
- **RAG (gene prioritization)**: retrieve knowledge from a FAISS index and output `POTENTIAL_GENES` (**exactly 10** items, single quotes).
  - `main_script/RareDxGPT_inference_RAG.py`
- **CoT-driven RAG (CoT → retrieve → finalize)**:
  - `main_script/RareDxGPT_inference_CoT_driven_RAG.py`
- **RAG-driven CoT (retrieve → CoT reasoning)**:
  - `main_script/RareDxGPT_inference_RAG_driven_CoT.py`

### Retrieval stack (used by RAG pipelines)
- **Vector store**: FAISS index loaded from local disk (`FAISS.load_local(...)`)
- **Embeddings**: `NeuML/pubmedbert-base-embeddings` via `langchain_huggingface.HuggingFaceEmbeddings`
- **Reranker**: ColBERTv2 via `ragatouille` (`colbert-ir/colbertv2.0`)

### Evaluation (in `AutoEvaluator/`)
- Evaluates predicted top-10 lists against ground truth.
- Supports E1/E2/E3-style matching and aggregate metrics (as implemented in the code).

---

## Repository structure

```text
.
├── AutoEvaluator/                  # evaluation utilities
├── dataset/                        # (optional) local datasets or dataset scripts
├── main_script/                    # main entrypoints for inference
├── utils/                          # helper utilities (seed, extraction, etc.)
├── LICENSE
└── README.md
```

---

## Requirements

- Python 3.9+ (recommended 3.10)
- PyTorch + CUDA (recommended for vLLM and embeddings)
- Key Python packages used in the scripts:
  - `datasets`, `transformers`, `accelerate`, `vllm`
  - `langchain`, `langchain_huggingface`, `langchain_community`
  - `faiss` (backend for FAISS index)
  - `ragatouille` (ColBERT reranker)
  - `pandas`, `tqdm`

> Tip: If you don’t have `requirements.txt` yet, generate it from your working environment:
> `pip freeze > requirements.txt`

---

## Installation

```bash
git clone https://github.com/WGLab/CoT-RAG-LLM-Gene-Prioritization-Disease-Diagnosis.git
cd CoT-RAG-LLM-Gene-Prioritization-Disease-Diagnosis

conda create -n cotrag python=3.10 -y
conda activate cotrag

# install dependencies (example)
pip install -U pip
pip install -r requirements.txt
```

---

## Data & paths (important)

The current scripts use **absolute paths** (e.g., `load_from_disk("/home/...")`, `FAISS.load_local("/home/...")`).  
To run on your machine, you must update these paths in the scripts or refactor them into CLI args.

Typical inputs:
- **Clinical note dataset**: loaded via `datasets.load_from_disk(...)`
- **Reference list**: e.g., `reference_data/disease_name_full.csv`
- **FAISS index directory**: e.g., `datasets/rag_embedding`

---

## Running inference

### 1) CoT disease diagnosis (Top-10 diseases)
```bash
python main_script/RareDxGPT_inference_CoT.py \
  --batch_size 8 \
  --ratio 0.3 \
  --seed 42 \
  --disease bws
```

**Expected output format (enforced by prompt):**
```text
REASONING:
Step 1: ...
...
POTENTIAL_DISEASES:
1. 'Disease1'
...
10. 'Disease10'
```

### 2) RAG gene prioritization (Top-10 genes)
```bash
python main_script/RareDxGPT_inference_RAG.py \
  --batch_size 8 \
  --ratio 0.3 \
  --seed 42
```

**Expected output format:**
```text
POTENTIAL_GENES:
1. 'Gene1'
...
10. 'Gene10'
```

### 3) CoT-driven RAG (gene)
```bash
python main_script/RareDxGPT_inference_CoT_driven_RAG.py
```

### 4) RAG-driven CoT (gene)
```bash
python main_script/RareDxGPT_inference_RAG_driven_CoT.py
```

---

## Notes on models & authentication

Some scripts call `huggingface_hub.login(token="")`.  
Set your token via environment variable or edit the script accordingly.

Example:
```bash
export HF_TOKEN="YOUR_HF_TOKEN"
```

If using vLLM, ensure your GPU environment is compatible and the model path/name is accessible.

---

## License

MIT License. See [LICENSE](LICENSE).
