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
from datasets import load_from_disk, load_dataset, Dataset
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
from ppktstore.registry import configure_phenopacket_registry
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain_openai import OpenAIEmbeddings
from typing import Optional, List, Tuple

from ragatouille import RAGPretrainedModel
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
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    args = parse_args()
    set_seed(args.seed)
    login(token ="")
    os.environ['HF_HOME'] = '/tmp'
    peft_model_id = "unsloth/Llama-3.3-70B-Instruct"
    number_gpus = 4
    sampling_params = SamplingParams(temperature=0.8,top_p=0.8, top_k=10, max_tokens=1024)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id, trust_remote_code=True)#,  truncation = True)
    dataset = load_from_disk("/home/wangz12/projects/RareDxGPT/datasets/phenopacket_store_filtered_final")
    dataset = dataset[9:10]
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

    
    prompt_in_chat_format = [
                {
                    "role": "system",
                    "content": """Using the information contained in the context,
            give a comprehensive answer to the question.
            Respond only to the question asked, response should be concise and relevant to the question.
            If the answer cannot be deduced from the context, do not give an answer.""",
                },
                {
                    "role": "user",
                    "content": """Context:
            {context}
            ---
            Now here is the question you need to answer.

            Question: {question}""",
                },
            ]
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    ground_truth_list = dataset['gene']

    # conversation_history = [
    #     {
    #         "role": "system",
    #         "content": (
    #             "You are a genetic counselor specializing in rare diseases. "
    #             "Your task is to reason step-by-step based on the given clinical notes to identify the most probable diseases."
    #         ),
    #     }
    # ]
    conversation_history = [
        {
        "role": "system",
        "content": (
            "You are a geneticist specializing in gene-disease associations. "
            "Your task is to reason step-by-step based on the given clinical notes to prioritize the most probable genes associated with the described phenotypes. "
            "Follow a structured approach to extract phenotypic features, map them to relevant genes, and refine the ranking based on inheritance patterns and variant evidence. "
            "Your final output must contain exactly 10 prioritized genes."
            ),
        }
    ]

    llm = LLM(
        model=peft_model_id,
        tensor_parallel_size=number_gpus, 
        max_num_batched_tokens=5024,  
        trust_remote_code=True,
        download_dir="/mnt/isilon/wang_lab/shared/",
        gpu_memory_utilization=0.95
    )
    def ask_client(messages, model=peft_model_id, tokenizer=tokenizer, temperature=0.8, max_tokens=2048, client_name="Huggingface",number_gpus=number_gpus):
        try:
            if client_name == "OpenAI":
                client = OpenAI()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            elif client_name == 'Huggingface':
                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                outputs = llm.generate(prompt, sampling_params)
                
                if not outputs or not outputs[0].outputs:
                    print("Generation failed, returning empty string")
                    return ""
                    
                return outputs[0].outputs[0].text
                
        except Exception as e:
            print(f"Error in ask_client: {e}")
            return "" 
    def add_message(role, content):
        """ Append to conversation history """
        conversation_history.append({"role": role, "content": content})

    def multi_turn_diagnosis(clinical_note):
        """ Multi-turn CoT diagnosis (step-by-step OpenAI API calls) """

        # **Step 1: Extract and classify HPO terms**
        add_message(
            "user",
            f"""Extract key **phenotypic features (HPO terms)** from the following clinical note and classify them according to the **Human Phenotype Ontology (HPO) system**: 
                
                ### **Classify the extracted terms under the following categories:**  
                - **Genitourinary system**  
                - **Cellular phenotype**  
                - **Blood and blood-forming tissues**  
                - **Head and neck**  
                - **Limbs**  
                - **Metabolism/homeostasis**  
                - **Prenatal development or birth**  
                - **Breast**  
                - **Cardiovascular system**  
                - **Digestive system**  
                - **Ear**  
                - **Endocrine system**  
                - **Eye**  
                - **Immune system**  
                - **Integument**  
                - **Musculoskeletal system**  
                - **Nervous system**  
                - **Respiratory system**  
                - **Thoracic cavity**  
                - **Voice**  
                - **Constitutional symptoms**  
                - **Growth abnormality**  
                - **Neoplasm**  

                ### **Guidelines for Classification:**  
                1. **Only include explicitly stated phenotypic features** from the clinical note. **Do not infer missing symptoms.**  
                2. **Each feature must be classified under one of the above categories** according to the official **HPO database**.  
                3. **If a feature does not fit into any category, mark it as "Unclassified."** Do not assume its category.  
                4. **Ensure that classification is medically accurate and follows HPO standards.**  

                Now, classify the **HPO terms** for the following patient:
                {clinical_note}
            """

        )
        step_1_response = ask_client(conversation_history)
        add_message("assistant", step_1_response)
        
        # **Step 2: Assess demographic impact**
        add_message("user", "How do the patient's **age, gender, and ethnicity** impact disease likelihood?")
        step_2_response = ask_client(conversation_history)
        add_message("assistant", step_2_response)

        # **Step 3: Categorize into disease groups**
        add_message("user", "Based on the extracted HPO terms, which broad disease categories are most relevant?")
        step_3_response = ask_client(conversation_history)
        add_message("assistant", step_3_response)

        # **Step 4: Narrow down the diagnosis**
        add_message("user", "Narrow down the diseases by eliminating less likely options based on phenotype severity, inheritance patterns, and clinical features.")
        step_4_response = ask_client(conversation_history)
        add_message("assistant", step_4_response)
        user_query = step_1_response
        retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=3)
        retrieved_docs = [doc.page_content for doc in retrieved_docs]
        relevant_docs = RERANKER.rerank(user_query, retrieved_docs, k=1)
        relevant_docs = [doc["content"] for doc in relevant_docs]
        relevant_docs = relevant_docs[:1]
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]) 
        add_message("user", f"Contexts: {context}")
        # print(context)
        # **Step 5: Provide EXACTLY 10 potential diseases**
        add_message("user", """Finalize a ranked list of EXACTLY 10 diseases that could explain the patient's presentation.
                \n\nPOTENTIAL_DISEASES: 
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
                Ensure all disease names are in single quotes, and there are exactly 10 in the list. Do not add any additional explanations or deviate from the specified format.""")
        step_5_response = ask_client(conversation_history)
        add_message("assistant", step_5_response)
        return step_5_response

    def multi_turn_gene_prioritization(clinical_note):
        """ Multi-turn CoT gene prioritization (step-by-step OpenAI API calls) """

        # **Step 1: Extract and classify HPO terms**
        add_message(
            "user",
            f"""Extract key **phenotypic features (HPO terms)** from the following clinical note and classify them according to the **Human Phenotype Ontology (HPO) system**:
                
                ### **Classify the extracted terms under the following categories:** 
                - **Genitourinary system** 
                - **Cellular phenotype** 
                - **Blood and blood-forming tissues** 
                - **Head and neck** 
                - **Limbs** 
                - **Metabolism/homeostasis** 
                - **Prenatal development or birth** 
                - **Breast** 
                - **Cardiovascular system** 
                - **Digestive system** 
                - **Ear** 
                - **Endocrine system** 
                - **Eye** 
                - **Immune system** 
                - **Integument** 
                - **Musculoskeletal system** 
                - **Nervous system** 
                - **Respiratory system** 
                - **Thoracic cavity** 
                - **Voice** 
                - **Constitutional symptoms** 
                - **Growth abnormality** 
                - **Neoplasm**  

                ### **Guidelines for Classification:**  
                1. **Only include explicitly stated phenotypic features** from the clinical note. **Do not infer missing symptoms.**  
                2. **Each feature must be classified under one of the above categories** according to the official **HPO database**.  
                3. **If a feature does not fit into any category, mark it as "Unclassified."** Do not assume its category.  
                4. **Ensure that classification is medically accurate and follows HPO standards.**  

                Now, classify the **HPO terms** for the following patient:  
                {clinical_note}
            """
        )
        step_1_response = ask_client(conversation_history)
        add_message("assistant", step_1_response)

        # **Step 2: Assess demographic impact**
        add_message("user", "How do the patient's **age, gender, and ethnicity** impact gene involvement likelihood?")
        step_2_response = ask_client(conversation_history)
        add_message("assistant", step_2_response)

        # **Step 3: Map HPO terms to gene-disease associations**
        add_message("user", "Based on the extracted HPO terms, identify the most relevant genes associated with these phenotypes.")
        step_3_response = ask_client(conversation_history)
        add_message("assistant", step_3_response)

        # **Step 4: Refine gene ranking based on inheritance patterns and variant evidence**
        add_message("user", "Refine the list of genes by considering inheritance patterns, known pathogenic variants, and functional impact.")
        step_4_response = ask_client(conversation_history)
        add_message("assistant", step_4_response)

        # Retrieve relevant knowledge from vector database
        user_query = step_1_response
        retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=3)
        retrieved_docs = [doc.page_content for doc in retrieved_docs]
        relevant_docs = RERANKER.rerank(user_query, retrieved_docs, k=1)
        relevant_docs = [doc["content"] for doc in relevant_docs]
        relevant_docs = relevant_docs[:1]
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
        add_message("user", f"Contexts: {context}")

        # **Step 5: Provide EXACTLY 10 prioritized genes**
        add_message("user", """Finalize a ranked list of EXACTLY 10 genes that are most likely associated with the patient's phenotypic features.
                \n\nPOTENTIAL_GENES:
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
                Ensure all gene names are in single quotes, and there are exactly 10 in the list. Do not add any additional explanations or deviate from the specified format.""")
        step_5_response = ask_client(conversation_history)
        add_message("assistant", step_5_response)

        return step_5_response

    inference_list = []
    index_sample = 0
    for note in dataset['clinical_note']:
        print(index_sample)
        index_sample += 1
        conversation_history.clear()
        # conversation_history.append({
        #     "role": "system",
        #     "content": (
        #         "You are a genetic counselor specializing in rare diseases. "
        #         "Your task is to reason step-by-step based on the given clinical notes to identify the most probable diseases."
        #     ),
        # })
        # final_diagnosis =  multi_turn_diagnosis(note)
        conversation_history.append(
        {
        "role": "system",
        "content": (
            "You are a geneticist specializing in gene-disease associations. "
            "Your task is to reason step-by-step based on the given clinical notes to prioritize the most probable genes associated with the described phenotypes. "
            "Follow a structured approach to extract phenotypic features, map them to relevant genes, and refine the ranking based on inheritance patterns and variant evidence. "
            "Your final output must contain exactly 10 prioritized genes."
            ),
        }
        )
        final_diagnosis =  multi_turn_gene_prioritization(note)
        print(conversation_history)
        inference_list.append(final_diagnosis)
        print(final_diagnosis)
    print(inference_list)
    df = pd.DataFrame(inference_list)
    df.to_csv("/home/wangz12/projects/RareDxGPT/output_CoT/arcus_disease_RAT_output.csv", index=False)
    inference_list = [s.replace('<|start_header_id|>assistant<|end_header_id|>\n', 'POTENTIAL_GENES:') for s in inference_list]
    # Process results
    print(inference_list)
    disease_name = pd.read_csv("/home/wangz12/projects/RareDxGPT/reference_data/disease_name_full.csv")
    reference_list = list(disease_name.Name)
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