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
from peft import AutoPeftModelForCausalLM
import sys
sys.path.append(os.path.abspath('/home/wangz12/projects/RareDxGPT/utils'))
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
from openai import OpenAI

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain_openai import OpenAIEmbeddings
from ragatouille import RAGPretrainedModel
from typing import Optional, List, Tuple
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

    
def main():
    # init_cuda()
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    args = parse_args()

    set_seed(args.seed)
    login(token = "")
    os.environ['HF_HOME'] = '/tmp'
    peft_model_id =  "unsloth/Llama-3.3-70B-Instruct" # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" # 
    number_gpus = 4
    sampling_params = SamplingParams(temperature=0.8,top_p=0.8, top_k=10, max_tokens=15000)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    disease_name = pd.read_csv("/home/wangz12/projects/RareDxGPT/reference_data/disease_name_full.csv")
    disease_name = list(disease_name.Name)
    reference_list = disease_name

    test_dataset = load_from_disk("/home/wangz12/projects/RareDxGPT/datasets/CoTRAG_clinical_notes")
    test_dataset = test_dataset.rename_column("Text", "clinical_note")
    test_dataset = test_dataset.rename_column("Gene", "gene")
    ground_truth_list = test_dataset['gene']

    EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
    embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
    )
   
    KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local("/home/wangz12/projects/RareDxGPT/datasets/rag_embedding", embeddings=embedding_model, allow_dangerous_deserialization=True)

    MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
    ]

    
    # prompt_in_chat_format = [
    #             {
    #                 "role": "system",
    #                 "content": """You are a genetic counselor. Your task is to identify potential rare diseases based on given phenotypes. Follow the output format precisely.""",
    #             },
    #             {
    #                 "role": "user",
    #                 "content": """Context:
    #         {context}
    #         ---
    #         Now here is the question you need to answer.
    #             {question}
                
    #             Based on this information, provide a numbered list of EXACTLY 10 potential rare diseases.
    
    #             Use EXACTLY this format:
                
    #             POTENTIAL_DISEASES:
    #             1. 'Disease1'
    #             2. 'Disease2'
    #             3. 'Disease3'
    #             4. 'Disease4'
    #             5. 'Disease5'
    #             6. 'Disease6'
    #             7. 'Disease7'
    #             8. 'Disease8'
    #             9. 'Disease9'
    #             10. 'Disease10'
                
    #             Ensure all disease names are in single quotes, and there are exactly 10 in the list. Do not deviate from this format or add any explanations.
    #             """}
    #             ]

    prompt_in_chat_format = [
                {
                    "role": "system",
                    "content": """You are a genetic counselor. Your task is to identify potential genes associated with the given phenotypes. Follow the output format precisely.""",
                },
                {
                    "role": "user",
                    "content": """Context:
            {context}
            ---
            Now here is the question you need to answer.
                {question}
            Based on this information, provide a numbered list of EXACTLY 10 potential genes.
            
            Use EXACTLY this format:
            
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
            
            Ensure all gene names are in single quotes, and there are exactly 10 in the list. 
            Do not deviate from this format or add any explanations.    
                """}
                ]
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    prompts = test_dataset['clinical_note']
    llm = LLM(
    model=peft_model_id,
    tensor_parallel_size=number_gpus, 
    max_num_batched_tokens=15000,  
    trust_remote_code=True,
    download_dir="/mnt/isilon/wang_lab/shared/",
)
    
    inference_list = []
    for prompt in prompts:
        user_query = prompt
        retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=3)
        retrieved_docs = [doc.page_content for doc in retrieved_docs]
        relevant_docs = RERANKER.rerank(user_query, retrieved_docs, k=1)
        relevant_docs = [doc["content"] for doc in relevant_docs]
        relevant_docs = relevant_docs[:1]
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
        final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=user_query)
        outputs = llm.generate(final_prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text
        inference_list.append(generated_text)
        print(outputs)
    print(inference_list)
    df = pd.DataFrame(inference_list)
    df.to_csv('/home/wangz12/projects/RareDxGPT/output_CoT/RAG_cl_result.csv', index=False, header=False)
    inference_list = [extract_potential_genes(text) for text in inference_list if text]
    # Process results
    print(inference_list)
    def format_sample(sample_str):
        return sample_str.replace('  \n', '\n')

    inference_list = [format_sample(text) for text in inference_list]
    processor = EvaluationProcessor(reference_list, similarity_threshold=1.0, e1_similarity_threshold=0)
    eval_results = processor.evaluate_samples(
        inference_list,
        ground_truth_list,
        lambda_weight=1.0,
        calc_coverage=False,
        calc_avoidance=False,
        calc_car=False
    )
    print(eval_results)

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()