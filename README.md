# CoT–RAG LLM for Disease Diagnosis & Gene Prioritization

This repository contains the official implementation of a **CoT–RAG** framework that supports **both disease diagnosis and gene prioritization** across **five inference methods**. It includes end-to-end inference scripts, retrieval components (FAISS + biomedical embeddings + reranking), and an automated evaluator.

**Paper (arXiv):** https://arxiv.org/abs/2503.12286

---

## Methods

All five methods can be run for **disease diagnosis** (Top-10 diseases) **or** **gene prioritization** (Top-10 genes) by switching the task prompt / extraction function where applicable.

### 1) Base Model
**Script:** `main_script/RareDxGPT_inference_vllm.py`  
A vLLM-based script for running base models.

### 2) CoT (Chain-of-Thought)
**Script:** `main_script/RareDxGPT_inference_CoT.py`  
Single-pass CoT prompting with a strict Top-10 output format (e.g., `POTENTIAL_DISEASES` / `POTENTIAL_GENES`). Uses **vLLM** for generation.

### 3) RAG (Retrieval-Augmented Generation)
**Script:** `main_script/RareDxGPT_inference_RAG.py`  
Retrieves knowledge from a **FAISS** index using **PubMedBERT embeddings**, optionally reranks with **ColBERTv2**, and generates a grounded Top-10 list.

### 4) CoT-driven RAG (CoT → Retrieve → Finalize)
**Script:** `main_script/RareDxGPT_inference_CoT_driven_RAG.py`  
Runs multi-step reasoning first, uses intermediate reasoning as the retrieval query, then finalizes the ranked Top-10 list with retrieved evidence.

### 5) RAG-driven CoT (Retrieve → CoT reasoning)
**Script:** `main_script/RareDxGPT_inference_RAG_driven_CoT.py`  
Retrieves first, injects retrieved evidence into a CoT-style reasoning prompt, then outputs the Top-10 list.

---

## Repository layout

```text
.
├── AutoEvaluator/                  # evaluation utilities
├── dataset/                        # (optional) local datasets or dataset scripts
├── main_script/                    # inference entrypoints (five methods)
├── utils/                          # helper utilities (seed, extraction, etc.)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/WGLab/CoT-RAG-LLM-Gene-Prioritization-Disease-Diagnosis.git
cd CoT-RAG-LLM-Gene-Prioritization-Disease-Diagnosis

conda create -n cotrag python=3.10 -y
conda activate cotrag

pip install -U pip
pip install -r requirements.txt
```

> **Note:** `vllm` and FAISS may require CUDA / platform-specific installation. If `faiss-gpu` is not available for your platform, use `faiss-cpu`.

---

## Data & paths (important)

The original scripts contained hard-coded absolute paths (e.g., `/home/...`).  
This repo version expects **relative paths by default** and exposes common paths as CLI arguments.

Typical inputs:
- **Clinical note dataset**: a HuggingFace dataset directory loaded via `datasets.load_from_disk(...)`
- **Reference file**: `reference_data/disease_name_full.csv` (or your own list)
- **FAISS index**: a local directory created previously for retrieval

---

## Usage

Each script now supports a consistent set of path arguments. Examples below assume:

- dataset directory: `datasets/CoTRAG_clinical_notes`
- FAISS index directory: `datasets/rag_embedding`
- reference CSV: `reference_data/disease_name_full.csv`
- outputs: `outputs/`

### 1) CoT
```bash
python main_script/RareDxGPT_inference_CoT.py \
  --data_dir datasets/CoTRAG_clinical_notes \
  --reference_csv reference_data/disease_name_full.csv \
  --output_dir outputs/cot \
  --task disease
```

### 2) RAG
```bash
python main_script/RareDxGPT_inference_RAG.py \
  --data_dir datasets/CoTRAG_clinical_notes \
  --index_dir datasets/rag_embedding \
  --reference_csv reference_data/disease_name_full.csv \
  --output_dir outputs/rag \
  --task gene
```

### 3) CoT-driven RAG
```bash
python main_script/RareDxGPT_inference_CoT_driven_RAG.py \
  --data_dir datasets/CoTRAG_clinical_notes \
  --index_dir datasets/rag_embedding \
  --reference_csv reference_data/disease_name_full.csv \
  --output_dir outputs/cot_driven_rag \
  --task gene
```

### 4) RAG-driven CoT
```bash
python main_script/RareDxGPT_inference_RAG_driven_CoT.py \
  --data_dir datasets/CoTRAG_clinical_notes \
  --index_dir datasets/rag_embedding \
  --reference_csv reference_data/disease_name_full.csv \
  --output_dir outputs/rag_driven_cot \
  --task disease
```

### 5) vLLM
```bash
python main_script/RareDxGPT_inference_vllm.py \
  --data_dir datasets/bws \
  --reference_csv reference_data/disease_name_full.csv \
  --output_dir outputs/vllm \
  --task disease
```

---

## Outputs

All scripts write:
- a raw generation CSV in `--output_dir`
- extracted Top-10 lists (and evaluator summaries if enabled)

---

## Citation

If you use this codebase, please cite:

```bibtex
@article{cotrag2025,
  title={CoT-RAG LLM Gene Prioritization for Disease Diagnosis},
  author={...},
  journal={arXiv preprint arXiv:2503.12286},
  year={2025},
  url={https://arxiv.org/abs/2503.12286}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
