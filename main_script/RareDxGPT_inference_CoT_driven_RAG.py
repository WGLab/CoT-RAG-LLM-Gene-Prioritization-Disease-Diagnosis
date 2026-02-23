"""
CoT-driven RAG (CoT → Retrieve → Finalize)

This refactored version removes hard-coded absolute paths and exposes them as CLI args.
It keeps the same high-level flow as the original script.
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
from openai import OpenAI

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
    p = argparse.ArgumentParser(description="CoT-driven RAG (disease / gene)")
    p.add_argument("--task", choices=["disease", "gene"], default="gene")
    p.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "datasets" / "CoTRAG_clinical_notes"))
    p.add_argument("--index_dir", type=str, default=str(REPO_ROOT / "datasets" / "rag_embedding"))
    p.add_argument("--reference_csv", type=str, default=str(REPO_ROOT / "reference_data" / "disease_name_full.csv"))
    p.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "outputs" / "cot_driven_rag"))
    p.add_argument("--seed", type=int, default=42)

    # Retrieval
    p.add_argument("--embedding_model", type=str, default="NeuML/pubmedbert-base-embeddings")
    p.add_argument("--retrieve_k", type=int, default=3)
    p.add_argument("--rerank_k", type=int, default=1)
    p.add_argument("--reranker", type=str, default="colbert-ir/colbertv2.0")

    # Generation
    p.add_argument("--model", type=str, default="unsloth/Llama-3.3-70B-Instruct")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--download_dir", type=str, default=str(REPO_ROOT / ".hf_cache"))
    p.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--client", choices=["Huggingface", "OpenAI"], default="Huggingface")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        download_dir=args.download_dir,
        max_num_batched_tokens=max(5024, args.max_tokens),
        gpu_memory_utilization=0.95,
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = FAISS.load_local(args.index_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
    reranker = RAGPretrainedModel.from_pretrained(args.reranker)

    conversation = []

    def add(role: str, content: str):
        conversation.append({"role": role, "content": content})

    def ask(messages):
        if args.client == "OpenAI":
            client = OpenAI()
            resp = client.chat.completions.create(model=args.model, messages=messages, temperature=args.temperature, max_tokens=args.max_tokens)
            return resp.choices[0].message.content
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        out = llm.generate(prompt, sampling_params)
        return out[0].outputs[0].text if out and out[0].outputs else ""

    def run_one(note: str) -> str:
        conversation.clear()
        if args.task == "disease":
            add("system", "You are a genetic counselor specializing in rare diseases.")
            add("user", f"Extract key phenotypic features (HPO terms) from the note:\n{note}")
            step1 = ask(conversation); add("assistant", step1)

            # Retrieve using step1
            retrieved = vector_db.similarity_search(query=step1, k=args.retrieve_k)
            docs = [d.page_content for d in retrieved]
            reranked = reranker.rerank(step1, docs, k=args.rerank_k)
            context_docs = [d["content"] for d in reranked][: args.rerank_k]
            context = "\nExtracted documents:\n" + "".join([f"Document {i}:::\n{doc}" for i, doc in enumerate(context_docs)])
            add("user", f"Contexts: {context}")

            add("user", "Finalize a ranked list of EXACTLY 10 diseases.\n\nPOTENTIAL_DISEASES:\n1. 'Disease1'\n...\n10. 'Disease10'")
            return ask(conversation)

        add("system", "You are a geneticist specializing in gene-disease associations.")
        add("user", f"Extract key phenotypic features (HPO terms) from the note:\n{note}")
        step1 = ask(conversation); add("assistant", step1)

        retrieved = vector_db.similarity_search(query=step1, k=args.retrieve_k)
        docs = [d.page_content for d in retrieved]
        reranked = reranker.rerank(step1, docs, k=args.rerank_k)
        context_docs = [d['content'] for d in reranked][: args.rerank_k]
        context = "\nExtracted documents:\n" + "".join([f"Document {i}:::\n{doc}" for i, doc in enumerate(context_docs)])
        add("user", f"Contexts: {context}")

        add("user", "Finalize a ranked list of EXACTLY 10 genes.\n\nPOTENTIAL_GENES:\n1. 'Gene1'\n...\n10. 'Gene10'")
        return ask(conversation)

    generations = [run_one(n) for n in ds["clinical_note"]]

    pd.DataFrame(generations).to_csv(out_dir / f"cot_driven_rag_raw_{args.task}.csv", index=False, header=False)

    if args.task == "disease":
        extracted = [extract_potential_diseases(t) for t in generations if t]
    else:
        extracted = [extract_potential_genes(t) for t in generations if t]

    pd.DataFrame(extracted).to_csv(out_dir / f"cot_driven_rag_extracted_{args.task}.csv", index=False, header=False)

    processor = EvaluationProcessor(reference_list, similarity_threshold=1.0, e1_similarity_threshold=0)
    eval_results = processor.evaluate_samples(extracted, list(ground_truth_list), lambda_weight=1.0,
                                             calc_coverage=False, calc_avoidance=False, calc_car=False)
    (out_dir / f"cot_driven_rag_eval_{args.task}.json").write_text(str(eval_results), encoding="utf-8")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
