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
    p = argparse.ArgumentParser(description="RAG inference (disease diagnosis / gene prioritization)")
    p.add_argument("--task", choices=["disease", "gene"], default="gene")
    p.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "datasets" / "CoTRAG_clinical_notes"))
    p.add_argument("--index_dir", type=str, default=str(REPO_ROOT / "datasets" / "rag_embedding"))
    p.add_argument("--reference_csv", type=str, default=str(REPO_ROOT / "reference_data" / "disease_name_full.csv"))
    p.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "outputs" / "rag"))
    p.add_argument("--seed", type=int, default=42)

    # Retrieval
    p.add_argument("--embedding_model", type=str, default="NeuML/pubmedbert-base-embeddings")
    p.add_argument("--retrieve_k", type=int, default=3)
    p.add_argument("--rerank_k", type=int, default=1)
    p.add_argument("--reranker", type=str, default="colbert-ir/colbertv2.0")

    # Generation
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

    # Reference list
    ref_df = pd.read_csv(args.reference_csv)
    reference_list = list(ref_df["Name"]) if "Name" in ref_df.columns else list(ref_df.iloc[:, 0])

    # Load dataset
    ds = load_from_disk(args.data_dir)
    if "Text" in ds.column_names:
        ds = ds.rename_column("Text", "clinical_note")
    if "Diagnosis" in ds.column_names and "disease" not in ds.column_names:
        ds = ds.rename_column("Diagnosis", "disease")
    if "Gene" in ds.column_names and "gene" not in ds.column_names:
        ds = ds.rename_column("Gene", "gene")

    ground_truth_list = ds["disease"] if args.task == "disease" else ds["gene"]

    # Retrieval components
    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = FAISS.load_local(args.index_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
    reranker = RAGPretrainedModel.from_pretrained(args.reranker)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Prompt template (kept close to original)
    if args.task == "disease":
        prompt_in_chat_format = [
            {"role": "system", "content": "You are a genetic counselor. Your task is to identify potential rare diseases based on given phenotypes. Follow the output format precisely."},
            {"role": "user", "content": """Context:
{context}
---
Now here is the question you need to answer.
{question}

Based on this information, provide a numbered list of EXACTLY 10 potential rare diseases.

POTENTIAL_DISEASES:
1. 'Disease1'
2. 'Disease2'
3. 'Disease3'
4. 'Disease4'
5. 'Disease5'
6. 'Disease6'
7. 'Disease7'
8. 'Disease8'
9. 'Disease9'
10. 'Disease10'

Ensure all disease names are in single quotes, and there are exactly 10 in the list. Do not deviate from this format or add any explanations.
"""},
        ]
    else:
        prompt_in_chat_format = [
            {"role": "system", "content": "You are a genetic counselor. Your task is to identify potential genes associated with the given phenotypes. Follow the output format precisely."},
            {"role": "user", "content": """Context:
{context}
---
Now here is the question you need to answer.
{question}

Based on this information, provide a numbered list of EXACTLY 10 potential genes.

POTENTIAL_GENES:
1. 'Gene1'
2. 'Gene2'
3. 'Gene3'
4. 'Gene4'
5. 'Gene5'
6. 'Gene6'
7. 'Gene7'
8. 'Gene8'
9. 'Gene9'
10. 'Gene10'

Ensure all gene names are in single quotes, and there are exactly 10 in the list. Do not deviate from this format or add any explanations.
"""},
        ]

    rag_template = tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)

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
        final_prompt = rag_template.format(context=context, question=note)

        out = llm.generate(final_prompt, sampling_params)
        generations.append(out[0].outputs[0].text)

    raw_path = out_dir / f"rag_raw_{args.task}.csv"
    pd.DataFrame(generations).to_csv(raw_path, index=False, header=False)

    if args.task == "disease":
        extracted = [extract_potential_diseases(t) for t in generations if t]
    else:
        extracted = [extract_potential_genes(t) for t in generations if t]
    ext_path = out_dir / f"rag_extracted_{args.task}.csv"
    pd.DataFrame(extracted).to_csv(ext_path, index=False, header=False)

    processor = EvaluationProcessor(reference_list, similarity_threshold=1.0, e1_similarity_threshold=0)
    eval_results = processor.evaluate_samples(
        extracted,
        list(ground_truth_list),
        lambda_weight=1.0,
        calc_coverage=False,
        calc_avoidance=False,
        calc_car=False,
    )
    (out_dir / f"rag_eval_{args.task}.json").write_text(str(eval_results), encoding="utf-8")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
