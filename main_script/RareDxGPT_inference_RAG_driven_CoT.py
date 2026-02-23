"""
RAG-driven CoT (Retrieve → CoT reasoning)

This refactored version removes hard-coded absolute paths and exposes them as CLI args.
"""
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from huggingface_hub import login
from vllm import LLM, SamplingParams

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ragatouille import RAGPretrainedModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "utils"))
sys.path.append(str(REPO_ROOT / "AutoEvaluator"))

from set_seed import set_seed
from disease_list_extract import extract_potential_diseases, extract_potential_genes  # type: ignore
from EvaluatorProcessor import EvaluationProcessor  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="RAG-driven CoT (disease / gene)")
    p.add_argument("--task", choices=["disease", "gene"], default="gene")
    p.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "datasets" / "CoTRAG_clinical_notes"))
    p.add_argument("--index_dir", type=str, default=str(REPO_ROOT / "datasets" / "rag_embedding"))
    p.add_argument("--reference_csv", type=str, default=str(REPO_ROOT / "reference_data" / "disease_name_full.csv"))
    p.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "outputs" / "rag_driven_cot"))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--embedding_model", type=str, default="NeuML/pubmedbert-base-embeddings")
    p.add_argument("--retrieve_k", type=int, default=3)
    p.add_argument("--rerank_k", type=int, default=1)
    p.add_argument("--reranker", type=str, default="colbert-ir/colbertv2.0")

    p.add_argument("--model", type=str, default="unsloth/Llama-3.3-70B-Instruct")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=15000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--download_dir", type=str, default=str(REPO_ROOT / ".hf_cache"))
    p.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.hf_token:
        login(token=args.hf_token)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_df = pd.read_csv(args.reference_csv)
    reference_list = list(ref_df["Name"]) if "Name" in ref_df.columns else list(ref_df.iloc[:, 0])

    ds = load_from_disk(args.data_dir)
    if "Text" in ds.column_names:
        ds = ds.rename_column("Text", "clinical_note")
    if "Diagnosis" in ds.column_names and "disease" not in ds.column_names:
        ds = ds.rename_column("Diagnosis", "disease")
    if "Gene" in ds.column_names and "gene" not in ds.column_names:
        ds = ds.rename_column("Gene", "gene")
    ground_truth_list = ds["disease"] if args.task == "disease" else ds["gene"]

    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = FAISS.load_local(args.index_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
    reranker = RAGPretrainedModel.from_pretrained(args.reranker)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # CoT prompt with retrieved evidence
    if args.task == "disease":
        system = "You are a genetic counselor. Identify rare diseases based on phenotypes. Follow output format exactly."
        user_tmpl = """Context:
{context}
---
Question:
{question}

Provide EXACTLY 10 potential diseases.

POTENTIAL_DISEASES:
1. 'Disease1'
...
10. 'Disease10'
"""
    else:
        system = "You are a geneticist specializing in gene-disease associations. Follow output format exactly."
        user_tmpl = """Context:
{context}
---
Question:
{question}

Provide EXACTLY 10 potential genes.

POTENTIAL_GENES:
1. 'Gene1'
...
10. 'Gene10'
"""

    prompt_in_chat_format = [{"role": "system", "content": system}, {"role": "user", "content": user_tmpl}]
    template = tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=max(15000, args.max_tokens),
        trust_remote_code=True,
        download_dir=args.download_dir,
    )

    generations = []
    for note in ds["clinical_note"]:
        retrieved = vector_db.similarity_search(query=note, k=args.retrieve_k)
        docs = [d.page_content for d in retrieved]
        reranked = reranker.rerank(note, docs, k=args.rerank_k)
        context_docs = [d["content"] for d in reranked][: args.rerank_k]
        context = "\nExtracted documents:\n" + "".join([f"Document {i}:::\n{doc}" for i, doc in enumerate(context_docs)])
        final_prompt = template.format(context=context, question=note)

        out = llm.generate(final_prompt, sampling_params)
        generations.append(out[0].outputs[0].text)

    pd.DataFrame(generations).to_csv(out_dir / f"rag_driven_cot_raw_{args.task}.csv", index=False, header=False)

    if args.task == "disease":
        extracted = [extract_potential_diseases(t) for t in generations if t]
    else:
        extracted = [extract_potential_genes(t) for t in generations if t]

    pd.DataFrame(extracted).to_csv(out_dir / f"rag_driven_cot_extracted_{args.task}.csv", index=False, header=False)

    processor = EvaluationProcessor(reference_list, similarity_threshold=1.0, e1_similarity_threshold=0)
    eval_results = processor.evaluate_samples(extracted, list(ground_truth_list), lambda_weight=1.0,
                                             calc_coverage=False, calc_avoidance=False, calc_car=False)
    (out_dir / f"rag_driven_cot_eval_{args.task}.json").write_text(str(eval_results), encoding="utf-8")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
