import json
import os
import re
import gc
import random
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import subprocess
from transformers import (AutoTokenizer,
                         AutoModelForCausalLM,
                         pipeline
                        )
from datasets import load_from_disk, load_dataset
import wandb
from peft import AutoPeftModelForCausalLM, PeftModel, PeftConfig
import sys
sys.path.append(os.path.abspath('/home/wangz12/projects/RareDxGPT/utils'))
from disease_gene_convert import *
from set_seed import *
from util_llama3_70b import *
from huggingface_hub import login
from disease_list_extract import *
from external_analysis_util import *
sys.path.append(os.path.abspath('/home/wangz12/projects/RareDxGPT/AutoEvaluator'))
from AutoEvaluator import *
from EvaluatorProcessor import *
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import gather_object
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def parse_args():
    # Your existing parse_args function
    parser = argparse.ArgumentParser(
        description="RareDxGPT-orpo"
    )
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size for training"
                        )
    
    parser.add_argument("--ratio",
                        type=float,
                        default=0.3,
                        help="Train test split ratio"
                        )

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for train test split")

    parser.add_argument("--disease",
                        type=str,
                        default='bws',
                        help='Disease name for external dataset')

    parser.add_argument("--peft_model_id",
                        type=str,
                        default="x")
    return parser.parse_args()


def loading_dataset(tokenizer):
    # Your existing loading_dataset function
    total_train = load_dataset("csv", data_files="/home/wangz12/projects/RareDxGPT/reference_data/total_train.csv")
    disease_name = pd.read_csv("/home/wangz12/projects/RareDxGPT/reference_data/disease_name_full.csv")
    disease_name = list(disease_name.Name)
    reference_list = disease_name
    full_dataset = total_train['train']
    test_dataset_dict = load_from_disk("/home/wangz12/projects/RareDxGPT/datasets/orpo_dpo_dataset_cask10_10")
    # test_dataset_dict = test_dataset_dict.remove_columns('image_id')
    test_dataset = test_dataset_dict['test']

    dataset1 = full_dataset
    dataset2 = test_dataset
    dataset1_df = dataset1.to_pandas()
    dataset2_df = dataset2.to_pandas()

    dataset1_df['image_id'] = dataset1_df['image_id'].astype(str)
    dataset2_df['image_id'] = dataset2_df['image_id'].astype(str)

    merged_df = pd.merge(dataset2_df[['image_id']], dataset1_df[['image_id', 'Response']], on='image_id', how='left')
    ground_truth_list = merged_df['Response'].tolist()

    return test_dataset, ground_truth_list, reference_list

    
def main():
    # init_cuda()
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    args = parse_args()
    os.environ['HF_HOME'] = '/tmp'
    parent_path = "/home/wangz12/projects"
    child_path = args.peft_model_id
    peft_model_id = os.path.join(parent_path, child_path)
    model_path = "/mnt/isilon/wang_lab/shared/LLaMA3.2-3B-Instruct"
    print(peft_model_id)
    # dist.init_process_group(backend="nccl", init_method="env://")
    set_seed(args.seed)
    login(token ="")
    os.environ['HF_HOME'] = '/tmp'
    # peft_model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  # "unsloth/Llama-3.3-70B-Instruct"
    number_gpus = 1
    sampling_params = SamplingParams(temperature=0.8,top_p=0.8, top_k=10, max_tokens=512)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    # model.gradient_checkpointing_enable()
    print("checkpoint")
    # Load dataset
    # test_dataset, ground_truth_list, reference_list = loading_dataset(tokenizer)#
    _, _, reference_list = loading_dataset(tokenizer)
    test_dataset = load_from_disk(f"/home/wangz12/projects/RareDxGPT/datasets/{args.disease}")
    test_dataset = test_dataset.rename_column("original_text", "clinical_note")
    test_dataset = test_dataset.rename_column("response", "disease")
    # test_dataset = test_dataset.rename_column("Response", "response")
    # ground_truth_list = test_dataset['response']
    ground_truth_list = test_dataset['disease']
    print(ground_truth_list)
    def format_data(example):
        return {"messages": [
        {"role": "system", "content": "You are a genetic counselor. Your task is to identify potential rare diseases based on given phenotypes. Follow the output format precisely."},
        {"role": "user", "content": f"{example['clinical_note']}\n\nBased on this information, provide a numbered list of EXACTLY 10 potential rare diseases.\n\nUse EXACTLY this format:\n\nPOTENTIAL_DISEASES:\n1. 'Disease1'\n2. 'Disease2'\n3. 'Disease3'\n4. 'Disease4'\n5. 'Disease5'\n6. 'Disease6'\n7. 'Disease7'\n8. 'Disease8'\n9. 'Disease9'\n10. 'Disease10'\n\nEnsure all disease names are in single quotes, and there are exactly 10 in the list. Do not deviate from this format or add any explanations."},
        ]}
    def format_gene_data(example):
        return {
            "messages": [
                {"role": "system", "content": "You are a genetic counselor. Your task is to identify potential genes associated with the given phenotypes. Follow the output format precisely."},
                {"role": "user", "content": f"{example['clinical_note']}\n\nBased on this information, provide a numbered list of EXACTLY 10 potential genes.\n\nUse EXACTLY this format:\n\nPOTENTIAL_GENES:\n1. 'Gene1'\n2. 'Gene2'\n3. 'Gene3'\n4. 'Gene4'\n5. 'Gene5'\n6. 'Gene6'\n7. 'Gene7'\n8. 'Gene8'\n9. 'Gene9'\n10. 'Gene10'\n\nEnsure all gene names are in single quotes, and there are exactly 10 in the list. Do not deviate from this format or add any explanations."}
            ]
        }

    def format_chat_template(row):
        row["messages"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return row

    # Format data
    # test_dataset = test_dataset.map(format_data)
    test_dataset = test_dataset.map(format_data)
    test_dataset = test_dataset.map(format_chat_template)
    prompts = test_dataset['messages']

    base_model_id = model_path
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    # base_model.resize_token_embeddings(len(tokenizer))
    # config = PeftConfig.from_pretrained(peft_model_id)
    # model = PeftModel.from_pretrained(base_model,
    #                                   peft_model_id,
    #                                   config=config,
    #                                   device_map="auto",
    #                                   torch_dtype=torch.bfloat16,
    #                                   offload_folder="offload"
    #                                  )
    lora_path = peft_model_id + "/adapter_model.safetensors"
    new_path = peft_model_id + "x" + "/adapter_model.safetensors"
    import safetensors.torch
    tensors =  safetensors.torch.load_file(lora_path)
    print(new_path)
    nonlora_keys = []
    for k in list(tensors.keys()):
        if "lora" not in k:
            nonlora_keys.append(k)

    print(nonlora_keys) 

    for k in nonlora_keys:
        del tensors[k]
    safetensors.torch.save_file(tensors, new_path)
    x = peft_model_id + "x"
    print(x)
    llm = LLM(
    model=base_model_id,
    tensor_parallel_size=number_gpus,
    enable_lora=True,
    max_lora_rank = 256,
    # gpu_memory_utilization=0.85,  
    # max_num_batched_tokens=512,  
    # max_num_seqs=16,  
    trust_remote_code=True,
    download_dir="/mnt/isilon/wang_lab/shared/",
    # quantization="awq" 
)
    inference_list = []
    for prompt in prompts:
        outputs = llm.generate(prompt, 
                               sampling_params,)
                            #    lora_request=LoRARequest("adapters1", 1, x)
                            #   )
        generated_text = outputs[0].outputs[0].text
        inference_list.append(generated_text)
        print(outputs)
    # inference_list = [extraction1(text) for text in inference_list if text]
    # inference_list = [extraction3(text) for text in inference_list if text]
    inference_list = [extract_potential_diseases(text) for text in inference_list if text]
    # # Process results
    # bottom_q = process_ranked_diseases(test_dataset['rejected'])
    # top_k = process_ranked_diseases(test_dataset['chosen'])
    print(inference_list)
    gene_samples = gene_list_convert(inference_list)
    results = evaluation(gene_samples, ground_truth_list)
    print(results)
    # processor = EvaluationProcessor(reference_list, similarity_threshold=0.8, e1_similarity_threshold=0.5)
    # eval_results = processor.evaluate_samples(
    #     inference_list,
    #     ground_truth_list,
    #     # top_k,
    #     # bottom_q,
    #     # k=10,
    #     # q=10,
    #     lambda_weight=1.0,
    #     calc_coverage=False,
    #     calc_avoidance=False,
    #     calc_car=False
    # )
    # print(eval_results)
    
    # # Analysis
    # stats = analyze_disease_rankings(inference_list)
    # print("\nDisease ranking statistics:")
    # for position in sorted(stats.keys()):
    #     print(f"\nPosition {position} disease counts:")
    #     if stats[position]:
    #         for disease, count in sorted(stats[position].items(), key=lambda x: x[1], reverse=True):
    #             print(f"  {disease}: {count}")
    #     else:
    #         print("  No diseases found at this position")
    
    # # Total occurrences
    # all_diseases = {}
    # for pos_stats in stats.values():
    #     for disease, count in pos_stats.items():
    #         all_diseases[disease] = all_diseases.get(disease, 0) + count
    
    # print("\nTotal occurrences across positions:")
    # for disease, count in sorted(all_diseases.items(), key=lambda x: x[1], reverse=True):
    #     print(f"  {disease}: {count}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()