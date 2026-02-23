"""
vLLM baseline runner (path-refactored)

This script refactors absolute paths into CLI arguments and repo-relative defaults.
It also keeps optional LoRA adapter handling hooks.
"""
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "utils"))
sys.path.append(str(REPO_ROOT / "AutoEvaluator"))

from set_seed import set_seed
from disease_list_extract import extract_potential_diseases, extract_potential_genes  # type: ignore
from disease_gene_convert import gene_list_convert  # type: ignore
from AutoEvaluator import evaluation  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="vLLM baseline (disease / gene)")
    p.add_argument("--task", choices=["disease", "gene"], default="disease")
    p.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "datasets" / "bws"))
    p.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "outputs" / "vllm"))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model", type=str, default=str(REPO_ROOT / "models" / "LLaMA3.2-3B-Instruct"),
                   help="Base model path or HF model id.")
    p.add_argument("--adapter_dir", type=str, default="",
                   help="Optional LoRA/adapter directory (not required).")

    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--tensor_parallel_size", type=int, default=1)

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

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = load_from_disk(args.data_dir)
    # align names
    if "original_text" in ds.column_names and "clinical_note" not in ds.column_names:
        ds = ds.rename_column("original_text", "clinical_note")
    if "response" in ds.column_names and "disease" not in ds.column_names:
        ds = ds.rename_column("response", "disease")

    ground_truth_list = ds["disease"] if args.task == "disease" else ds.get("gene", [])

    def format_messages(note: str):
        if args.task == "disease":
            return [{"role": "system", "content": "You are a genetic counselor. Follow the output format precisely."},
                    {"role": "user", "content": f"{note}\n\nProvide EXACTLY 10 potential rare diseases.\n\nPOTENTIAL_DISEASES:\n1. 'Disease1'\n...\n10. 'Disease10'"}]
        return [{"role": "system", "content": "You are a genetic counselor. Follow the output format precisely."},
                {"role": "user", "content": f"{note}\n\nProvide EXACTLY 10 potential genes.\n\nPOTENTIAL_GENES:\n1. 'Gene1'\n...\n10. 'Gene10'"}]

    prompts = [tokenizer.apply_chat_template(format_messages(n), tokenize=False) for n in ds["clinical_note"]]

    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        download_dir=args.download_dir,
        enable_lora=bool(args.adapter_dir),
    )

    generations = []
    for p in prompts:
        out = llm.generate(p, sampling_params)
        generations.append(out[0].outputs[0].text)

    pd.DataFrame(generations).to_csv(out_dir / f"vllm_raw_{args.task}.csv", index=False, header=False)

    if args.task == "disease":
        extracted = [extract_potential_diseases(t) for t in generations if t]
    else:
        extracted = [extract_potential_genes(t) for t in generations if t]

    pd.DataFrame(extracted).to_csv(out_dir / f"vllm_extracted_{args.task}.csv", index=False, header=False)

    # If your AutoEvaluator expects gene_samples, keep as-is; otherwise adjust.
    try:
        gene_samples = gene_list_convert(extracted)
        results = evaluation(gene_samples, list(ground_truth_list))
        (out_dir / f"vllm_eval_{args.task}.txt").write_text(str(results), encoding="utf-8")
    except Exception as e:
        (out_dir / f"vllm_eval_error_{args.task}.txt").write_text(repr(e), encoding="utf-8")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
