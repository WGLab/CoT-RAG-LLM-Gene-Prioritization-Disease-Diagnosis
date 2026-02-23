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

# Resolve repo root (…/main_script -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "utils"))
sys.path.append(str(REPO_ROOT / "AutoEvaluator"))

from set_seed import set_seed
from disease_list_extract import extract_potential_diseases, extract_potential_genes  # type: ignore
from EvaluatorProcessor import EvaluationProcessor  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="CoT inference (disease diagnosis / gene prioritization)")
    p.add_argument("--task", choices=["disease", "gene"], default="disease",
                   help="Run disease diagnosis (Top-10 diseases) or gene prioritization (Top-10 genes).")
    p.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "datasets" / "CoTRAG_clinical_notes"),
                   help="Path to HuggingFace dataset directory (load_from_disk).")
    p.add_argument("--reference_csv", type=str, default=str(REPO_ROOT / "reference_data" / "disease_name_full.csv"),
                   help="CSV with a column 'Name' used as reference list (as in the original scripts).")
    p.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "outputs" / "cot"),
                   help="Output directory for CSVs.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Model / runtime
    p.add_argument("--model", type=str, default="unsloth/Llama-3.3-70B-Instruct")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=5000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--download_dir", type=str, default=str(REPO_ROOT / ".hf_cache"),
                   help="Where vLLM downloads model weights.")
    p.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""),
                   help="HuggingFace token. Prefer exporting HF_TOKEN env var.")
    return p.parse_args()


def _format_sample(sample_str: str) -> str:
    return sample_str.replace("  \n", "\n")


def _build_prompt(task: str, clinical_note: str) -> list[dict]:
    if task == "disease":
        system = ("You are a genetic counselor specializing in rare diseases. "
                  "Your task is to reason step-by-step based on the given clinical notes "
                  "and identify EXACTLY 10 potential diseases or syndromes associated with the described phenotypes.")
        user = f"""{clinical_note}

Based on the provided clinical notes, follow this reasoning process:

1. **Extract and classify HPO terms**
2. **Assess demographic impact**
3. **Categorize into disease groups**
4. **Narrow down the diagnosis**
5. **Conclude with Top 10**

Output your reasoning for each step clearly, followed by the final disease list.

POTENTIAL_DISEASES:
1. 'Disease1'
...
10. 'Disease10'

Ensure all disease names are in single quotes, and there are exactly 10 in the list. Do not add any additional explanations or deviate from the specified format.
"""
    else:
        system = ("You are a geneticist specializing in gene-disease associations. "
                  "Your task is to reason step-by-step based on the given clinical notes "
                  "and prioritize EXACTLY 10 genes most likely associated with the described phenotypes.")
        user = f"""{clinical_note}

Based on the provided clinical notes, follow this reasoning process:

1. **Extract and classify HPO terms**
2. **Assess demographic impact**
3. **Map to relevant gene-disease associations**
4. **Refine based on inheritance patterns and variant evidence**
5. **Prioritize the top 10 genes**

POTENTIAL_GENES:
1. 'Gene1'
...
10. 'Gene10'

Ensure all gene names are in single quotes, and there are exactly 10 in the list. Do not add any additional explanations or deviate from the specified format.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.hf_token:
        login(token=args.hf_token)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference list (kept consistent with original usage)
    ref_df = pd.read_csv(args.reference_csv)
    reference_list = list(ref_df["Name"]) if "Name" in ref_df.columns else list(ref_df.iloc[:, 0])

    # Load dataset
    ds = load_from_disk(args.data_dir)
    # keep compatibility with existing field names
    if "Text" in ds.column_names:
        ds = ds.rename_column("Text", "clinical_note")
    if "Diagnosis" in ds.column_names and "disease" not in ds.column_names:
        ds = ds.rename_column("Diagnosis", "disease")
    if "Gene" in ds.column_names and "gene" not in ds.column_names:
        ds = ds.rename_column("Gene", "gene")

    ground_truth_list = ds["disease"] if args.task == "disease" else ds["gene"]

    # Build prompts
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = []
    for note in ds["clinical_note"]:
        messages = _build_prompt(args.task, note)
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=max(5000, args.max_tokens),
        trust_remote_code=True,
        download_dir=args.download_dir,
    )

    generations = []
    for prompt in prompts:
        out = llm.generate(prompt, sampling_params)
        generations.append(out[0].outputs[0].text)

    raw_path = out_dir / f"cot_raw_{args.task}.csv"
    pd.DataFrame(generations).to_csv(raw_path, index=False, header=False)

    # Extract + evaluate
    if args.task == "disease":
        extracted = [extract_potential_diseases(t) for t in generations if t]
    else:
        extracted = [extract_potential_genes(t) for t in generations if t]
    extracted = [_format_sample(t) for t in extracted]

    ext_path = out_dir / f"cot_extracted_{args.task}.csv"
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

    eval_path = out_dir / f"cot_eval_{args.task}.json"
    eval_path.write_text(str(eval_results), encoding="utf-8")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
