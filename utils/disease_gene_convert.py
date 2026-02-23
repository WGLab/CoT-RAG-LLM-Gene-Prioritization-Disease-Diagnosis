import requests
import re
import requests

import requests

def get_genes_from_mondo(disease_name):
    try:
        url1 = f"https://api.monarchinitiative.org/v3/api/autocomplete?q={disease_name}"
        resp1 = requests.get(url1).json()
        if not resp1['items']:
            return None  

        id = resp1['items'][0]['id']
        url2 = f"https://api.monarchinitiative.org/v3/api/entity/{id}"
        resp2 = requests.get(url2).json()

        if 'causal_gene' not in resp2 or not resp2['causal_gene']:
            return None 

        causal_gene = resp2['causal_gene'][0]['name']
        return causal_gene
    
    except Exception as e:
        print(f"Error processing disease '{disease_name}': {e}")
        return None


def extract_diseases(text):
    """
    Extract diseases from text that follows the pattern 'POTENTIAL_DISEASES:' followed by numbered list.
    Each disease is listed after a number (e.g., '1.', '2.', etc.)
    
    Args:
        text (str): The text containing potential diseases
        
    Returns:
        list: List of extracted diseases
    """
    diseases = []
    
    if 'POTENTIAL_DISEASES:' in text:
        # Split by 'POTENTIAL_DISEASES:' and take the part after it
        disease_section = text.split('POTENTIAL_DISEASES:')[1].strip()
        
        # Find all matches of numbered items
        matches = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)', disease_section, re.DOTALL)
        
        for match in matches:
            # Clean up the match and add to diseases list
            disease = match.strip()
            if disease:
                diseases.append(disease)
    
    return diseases

def gene_list_convert(disease):
    disease_samples = []
    for sample in disease:
        extracted_diseases = extract_diseases(sample)
        disease_samples.append(extracted_diseases)

    # Now map diseases to genes using the get_genes_from_mondo function
    gene_samples = []
    for diseases in disease_samples:
        genes = []
        for disease in diseases:
            try:
                gene = get_genes_from_mondo(disease)
                genes.append(gene)
            except Exception as e:
                # Handle cases where the API might not return a result
                print(f"Could not find gene for disease: {disease}. Error: {e}")
                genes.append(None)
        gene_samples.append(genes)
    return gene_samples

def evaluation(gene_samples, ground_truth_list):
    top_1 = 0
    top_10 = 0
    hfa = 0
    size = len(gene_samples)
    for idx, sample in enumerate(gene_samples):
        ground_truth = ground_truth_list[idx]
        all_none = all(gene is None for gene in sample) if gene_samples and sample else True
        if not all_none:
            hfa += 1
            if ground_truth in sample:
                top_10 += 1
            if ground_truth == sample[0]:
                top_1 += 1
    return hfa/size, top_10/size, top_1/size