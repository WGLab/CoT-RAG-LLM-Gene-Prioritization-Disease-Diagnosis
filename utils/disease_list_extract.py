import re
def extract_disease_list(raw_text: str) -> str:
    # Find content after assistant marker
    """
    Extract the disease name after <|eot_id|>assistant\n
    """
    match = re.search(r'<\|begin_of_text\|>.*?<\|eot_id\|>assistant\s*(.*?)(?=\[|$)', raw_text, re.DOTALL)
    if match:
        # Extract from \n1. onwards
        disease_list = re.search(r'(\n1\..*?)(?=\n\n|$)', match.group(1), re.DOTALL)
        if disease_list:
            return "POTENTIAL_DISEASES:" + disease_list.group(1)
    return ""


def process_diagnosis_list(diagnosis_list):
    return [extract_disease_list(text) for text in diagnosis_list]

def extract_rank_disease_names(conversation):
    """
    Extract the top k or bottom q disease name
    """
    disease_list = []
    for message in reversed(conversation):
        if message['role'] == 'assistant' and 'POTENTIAL_DISEASES:' in message['content']:
            content = message['content'].split('POTENTIAL_DISEASES:')[1].strip()
            # Extract disease names by removing numbers and dots
            diseases = []
            for line in content.split('\n'):
                # Remove the number and dot at the beginning
                disease = re.sub(r'^\d+\.\s*', '', line.strip())
                if disease:  # Skip empty lines
                    diseases.append(disease)
            return diseases
    return []

def process_ranked_diseases(ranked_diseases):
    return [extract_rank_disease_names(text) for text in ranked_diseases]
def extraction3(text: str) -> str:
    """
    input has to have "POTENTIAL_DISEASES:" in the string, and pick the top 10 diseases
    given in the list if k>10
    """
    if "POTENTIAL_DISEASES:" not in text:
        return ""

    content = text.split("POTENTIAL_DISEASES:")[-1].strip()
    lines = []
    for match in re.finditer(r'\n*(\d+)\.\s*([^\n]+)', content):
        num = int(match.group(1))
        if 1 <= num <= 10:  
            disease = match.group(2).strip()
            lines.append(f"\n{num}. {disease}")
    
    return "POTENTIAL_DISEASES:" + "".join(lines)
def extraction2(text):
    """
    This is only for POTENTIAL_DISEASES formatted one
    """
    # Split by POTENTIAL_DISEASES: and get everything after the second occurrence
    parts = text.split("POTENTIAL_DISEASES:")
    if len(parts) < 3:  # Need at least 3 parts to have 2 occurrences
        return ""
    
    content = "POTENTIAL_DISEASES:" + parts[2]
    
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and non-disease lines
        if not line:
            continue
        
        if line == "POTENTIAL_DISEASES:":
            cleaned_lines.append(line)
        elif line[0].isdigit():
            # Remove quotes and extra characters
            line = line.replace('"', '').replace("'", '')
            line = line.rstrip(' .,') 
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)
def extraction1(text: str) -> str:
    """
    extract disease list after <|im_end|>\assistant, 
    and the first 10 diseases appeared
    """
    if "<|im_end|>\nassistant" in text:
        content = text.split("<|im_end|>\nassistant")[-1].strip()
        disease_entries = []
        current_number = 1
        
        matches = re.finditer(r'\n*\d+\.[\s\'\"]*([^\n]+)', content)
        
        temp_entries = {}
        for match in matches:
            num_str = match.group(0).strip()
            num = int(re.match(r'\d+', num_str).group())
            
            if num == current_number and current_number <= 10:
                disease = re.sub(r'^\d+\.[\s\'\"]*', '', num_str).strip()
                disease = disease.strip('\'"')
                disease = re.sub(r'^Disease\d+[:\s-]+', '', disease)
                
                temp_entries[current_number] = disease
                current_number += 1
            
            if current_number > 10 and len(temp_entries) == 10:
                break
                
        if len(temp_entries) == 10:
            result = "POTENTIAL_DISEASES:"
            for i in range(1, 11):
                result += f"\n{i}. {temp_entries[i]}"
            return result
            
    return ""
def extraction5(text):
    try:
        if "<|im_end|>" not in text:
            # print("No <|im_end|> marker found")
            return ""
            
        content = text.split("<|im_end|>")[-1].strip()
        # print("Found content, first 200 chars:", content[:200])

        import re
        temp_lines = {}
        
        pattern = r'\n\s*(\d+)[\.]\s*(.+?)(?=\n\s*\d+[\.]|\n\n|$)'
        matches = re.finditer(pattern, content)
        
        for match in matches:
            num = int(match.group(1))
            content = match.group(2).strip()
            print(f"Found line {num}: {content[:100]}")
            
            if 1 <= num <= 10:
                temp_lines[num] = cont
        if temp_lines:
            result = "POTENTIAL_DISEASES:"
            for i in range(1, 11):
                if i in temp_lines:
                    result += f"\n{i}. {temp_lines[i]}"
            return result
            
        return ""
        
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""

import re

def extraction7(text):
    """
    Extract the second occurrence of POTENTIAL_DISEASES list from a text string.
    
    Args:
        text (str): Input text containing POTENTIAL_DISEASES lists
        
    Returns:
        str: The second POTENTIAL_DISEASES list if found, empty string if not found
    """
    pattern = r"POTENTIAL_DISEASES:.*(?:\n(?:(?!POTENTIAL_DISEASES:).)*)*"
    matches = re.findall(pattern, text)
    if len(matches) >= 2:
        return matches[1]
    return ""
def extraction8(text: str) -> str:
    """
    extract disease list after <|im_end|>\assistant, 
    and the first 10 diseases appeared
    """
    if "|start_header_id|>assistant<|end_header_id|" in text:
        content = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        disease_entries = []
        current_number = 1
        
        matches = re.finditer(r'\n*\d+\.[\s\'\"]*([^\n]+)', content)
        
        temp_entries = {}
        for match in matches:
            num_str = match.group(0).strip()
            num = int(re.match(r'\d+', num_str).group())
            
            if num == current_number and current_number <= 10:
                disease = re.sub(r'^\d+\.[\s\'\"]*', '', num_str).strip()
                disease = disease.strip('\'"')
                disease = re.sub(r'^Disease\d+[:\s-]+', '', disease)
                
                temp_entries[current_number] = disease
                current_number += 1
            
            if current_number > 10 and len(temp_entries) == 10:
                break
                
        if len(temp_entries) == 10:
            result = "POTENTIAL_DISEASES:"
            for i in range(1, 11):
                result += f"\n{i}. {temp_entries[i]}"
            return result
            
    return ""
def extract_potential_genes(text):
    start_index = text.find("POTENTIAL_GENES:")
    if start_index != -1:
        return text[start_index:]
    return ""

def extract_potential_diseases(text):
    start_index = text.find("POTENTIAL_DISEASES:")
    if start_index != -1:
        return text[start_index:]
    return ""
    