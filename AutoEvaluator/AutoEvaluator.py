# import re
# from difflib import SequenceMatcher

# class AutoEvaluator:
#     def __init__(self, reference_list, similarity_threshold=0.8, e1_similarity_threshold=0.5):
#         self.reference_list = [self.preprocess_disease_name(disease) for disease in reference_list]
#         self.similarity_threshold = similarity_threshold
#         self.e1_similarity_threshold = e1_similarity_threshold

#     @staticmethod
#     def preprocess_disease_name(name):
#         return re.sub(r'[^\w\s]', '', name.lower()).strip()

#     @staticmethod
#     def disease_similarity(disease1, disease2):
#         return SequenceMatcher(None, disease1, disease2).ratio()

#     def extract_diseases(self, output_string):
#         diseases_block = re.search(r"POTENTIAL_DISEASES:\n(.*?)(?:\n\n|$)", output_string, re.DOTALL)
#         if diseases_block:
#             diseases = re.findall(r"\d+\.\s*([^\n]*)", diseases_block.group(1))
#             return [self.preprocess_disease_name(disease) for disease in diseases]
#         return []

#     def E1(self, extracted_diseases):
#         for disease in extracted_diseases:
#             for ref_disease in self.reference_list:
#                 similarity = self.disease_similarity(disease, ref_disease)
#                 if similarity >= self.e1_similarity_threshold:
#                     return True
#         return False

#     def find_best_match(self, processed_extracted, ground_truth_parts):
#         best_match = None
#         best_similarity = 0
#         matched_ground_truth = None

#         for gt_part in ground_truth_parts:
#             for disease in processed_extracted:
#                 similarity = self.disease_similarity(gt_part, disease)
#                 if similarity > best_similarity:
#                     best_similarity = similarity
#                     best_match = disease
#                     matched_ground_truth = gt_part

#         is_match_found = best_similarity >= self.similarity_threshold
#         return is_match_found, best_match, best_similarity, matched_ground_truth

#     def E2(self, output_string, ground_truth):
#         extracted_diseases = self.extract_diseases(output_string)
#         e1_result = self.E1(extracted_diseases)
        
#         if not e1_result:
#             return False, None, 0, None

#         ground_truth_parts = [self.preprocess_disease_name(part.strip()) for part in ground_truth.split(';')]
#         return self.find_best_match(extracted_diseases, ground_truth_parts)

#     def E3(self, output_string, ground_truth):
#         extracted_diseases = self.extract_diseases(output_string)
#         if not extracted_diseases:
#             return False, None, 0, None, False
            
#         e1_result = self.E1(extracted_diseases)
#         if not e1_result:
#             return False, None, 0, None, False

#         ground_truth_parts = [self.preprocess_disease_name(part.strip()) for part in ground_truth.split(';')]
        
#         first_disease = extracted_diseases[0]
#         best_similarity = 0
#         matched_ground_truth = None

#         for gt_part in ground_truth_parts:
#             similarity = self.disease_similarity(gt_part, first_disease)
#             if similarity > best_similarity:
#                 best_similarity = similarity
#                 matched_ground_truth = gt_part

#         is_match_found = best_similarity >= self.similarity_threshold
#         return is_match_found, first_disease, best_similarity, matched_ground_truth, False

#     def calculate_coverage_rate(self, top_k_predictions, ground_truth_top_k):
#         """
#         Calculate coverage rate between model's top k predictions and ground truth top k
        
#         Args:
#             top_k_predictions: Top k predictions from the model
#             ground_truth_top_k: Top k ground truth diseases
#         Returns:
#             float: Coverage rate (intersection / k)
#         """
#         if not top_k_predictions:
#             return 0.0
            
#         intersection = 0
#         for pred in top_k_predictions:
#             for gt in ground_truth_top_k:
#                 if self.disease_similarity(pred, gt) >= self.similarity_threshold:
#                     intersection += 1
#                     break
#         return intersection / len(top_k_predictions)

#     def calculate_avoidance_rate(self, bottom_q_predictions, ground_truth_top_k):
#         """
#         Calculate how well bottom q predictions avoid ground truth top k
        
#         Args:
#             bottom_q_predictions: Bottom q predictions from the model
#             ground_truth_top_k: Top k ground truth diseases
#         Returns:
#             float: Avoidance rate (1 - intersection / q)
#         """
#         if not bottom_q_predictions:
#             return 0.0
            
#         intersection = 0
#         for pred in bottom_q_predictions:
#             for gt in ground_truth_top_k:
#                 if self.disease_similarity(pred, gt) >= self.similarity_threshold:
#                     intersection += 1
#                     break
#         return 1 - (intersection / len(bottom_q_predictions))

#     def calculate_car(self, output_string, ground_truth_top_k, bottom_predictions, k, q, lambda_weight=0.5):
#         """
#         Calculate Coverage-Avoidance Rate (CAR) for a single prediction
        
#         Args:
#             output_string: The model's output string containing potential diseases
#             ground_truth_top_k: List of top k ground truth diseases
#             bottom_predictions: List of bottom-q predicted diseases
#             k: Number of top diseases to consider
#             q: Number of bottom diseases to consider
#             lambda_weight: Weight parameter for balancing coverage and avoidance
        
#         Returns:
#             tuple: (coverage_rate, avoidance_rate, car_score)
#         """
#         # Extract and preprocess model predictions
#         extracted_diseases = self.extract_diseases(output_string)
#         if not extracted_diseases:
#             return 0.0, 0.0, 0.0

#         # Preprocess ground truth top k
#         processed_ground_truth = [self.preprocess_disease_name(d) for d in ground_truth_top_k]
        
#         # Get top-k and bottom-q predictions
#         top_k_predictions = extracted_diseases[:k]
#         bottom_q_predictions = [self.preprocess_disease_name(d) for d in bottom_predictions[:q]]

#         # Calculate coverage and avoidance rates
#         coverage_rate = self.calculate_coverage_rate(top_k_predictions, processed_ground_truth)
#         avoidance_rate = self.calculate_avoidance_rate(bottom_q_predictions, processed_ground_truth)

#         # Calculate CAR score
#         if coverage_rate == 0 and avoidance_rate == 0:
#             return 0.0, 0.0, 0.0

#         denominator = lambda_weight * coverage_rate + avoidance_rate
#         if denominator == 0:
#             return coverage_rate, avoidance_rate, 0.0

#         car_score = ((1 + lambda_weight) * coverage_rate * avoidance_rate) / denominator
#         return coverage_rate, avoidance_rate, car_score
import re
from difflib import SequenceMatcher
from typing import Optional, Tuple


class AutoEvaluator:
    def __init__(self, reference_list, similarity_threshold=0.8, e1_similarity_threshold=0.0):
        self.reference_list = [self.preprocess_disease_name(disease) for disease in reference_list]
        self.similarity_threshold = similarity_threshold
        self.e1_similarity_threshold = e1_similarity_threshold

    @staticmethod
    def preprocess_disease_name(name):
        return re.sub(r'[^\w\s]', '', name.lower()).strip()

    @staticmethod
    def disease_similarity(disease1, disease2):
        return SequenceMatcher(None, disease1, disease2).ratio()

    def extract_diseases(self, output_string):
        diseases_block = re.search(r"POTENTIAL_DISEASES:\n(.*?)(?:\n\n|$)", output_string, re.DOTALL)
        if diseases_block:
            diseases = re.findall(r"\d+\.\s*([^\n]*)", diseases_block.group(1))
            return [self.preprocess_disease_name(disease) for disease in diseases]
        return []

    def E1(self, extracted_diseases):
        for disease in extracted_diseases:
            for ref_disease in self.reference_list:
                similarity = self.disease_similarity(disease, ref_disease)
                if similarity >= self.e1_similarity_threshold:
                    return True
        return False

    def find_best_match(self, processed_extracted, ground_truth_parts):
        best_match = None
        best_similarity = 0
        matched_ground_truth = None

        for gt_part in ground_truth_parts:
            for disease in processed_extracted:
                similarity = self.disease_similarity(gt_part, disease)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = disease
                    matched_ground_truth = gt_part

        is_match_found = best_similarity >= self.similarity_threshold
        return is_match_found, best_match, best_similarity, matched_ground_truth

    def E2(self, output_string, ground_truth):
        extracted_diseases = self.extract_diseases(output_string)
        e1_result = self.E1(extracted_diseases)
        
        if not e1_result:
            return False, None, 0, None

        ground_truth_parts = [self.preprocess_disease_name(part.strip()) for part in ground_truth.split(';')]
        return self.find_best_match(extracted_diseases, ground_truth_parts)

    def E3(self, output_string, ground_truth):
        extracted_diseases = self.extract_diseases(output_string)
        if not extracted_diseases:
            return False, None, 0, None, False
            
        e1_result = self.E1(extracted_diseases)
        if not e1_result:
            return False, None, 0, None, False

        ground_truth_parts = [self.preprocess_disease_name(part.strip()) for part in ground_truth.split(';')]
        
        first_disease = extracted_diseases[0]
        best_similarity = 0
        matched_ground_truth = None

        for gt_part in ground_truth_parts:
            similarity = self.disease_similarity(gt_part, first_disease)
            if similarity > best_similarity:
                best_similarity = similarity
                matched_ground_truth = gt_part

        is_match_found = best_similarity >= self.similarity_threshold
        return is_match_found, first_disease, best_similarity, matched_ground_truth, False

    def calculate_coverage_rate(self, top_k_predictions, ground_truth_top_k):
        if not top_k_predictions:
            return 0.0
            
        intersection = 0
        for pred in top_k_predictions:
            for gt in ground_truth_top_k:
                if self.disease_similarity(pred, gt) >= self.similarity_threshold:
                    intersection += 1
                    break
        return intersection / len(top_k_predictions)

    def calculate_avoidance_rate(self, bottom_q_predictions, ground_truth_top_k):
        if not bottom_q_predictions:
            return 0.0
            
        intersection = 0
        for pred in bottom_q_predictions:
            for gt in ground_truth_top_k:
                if self.disease_similarity(pred, gt) >= self.similarity_threshold:
                    intersection += 1
                    break
        return 1 - (intersection / len(bottom_q_predictions))

    def calculate_optional_metrics(self, output_string, ground_truth_top_k=None, 
                                 bottom_predictions=None, k=None, q=None, 
                                 lambda_weight=0.5) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        coverage_rate = None
        avoidance_rate = None
        car_score = None
        
        extracted_diseases = self.extract_diseases(output_string)
        
        if ground_truth_top_k is not None and k is not None:
            top_k_predictions = extracted_diseases[:k]
            coverage_rate = self.calculate_coverage_rate(top_k_predictions, ground_truth_top_k)
            
        if bottom_predictions is not None and ground_truth_top_k is not None and q is not None:
            processed_bottom = [self.preprocess_disease_name(d) for d in bottom_predictions[:q]]
            avoidance_rate = self.calculate_avoidance_rate(processed_bottom, ground_truth_top_k)
            
        if coverage_rate is not None and avoidance_rate is not None:
            if coverage_rate == 0 and avoidance_rate == 0:
                car_score = 0.0
            else:
                denominator = lambda_weight * coverage_rate + avoidance_rate
                if denominator == 0:
                    car_score = 0.0
                else:
                    car_score = ((1 + lambda_weight) * coverage_rate * avoidance_rate) / denominator
                    
        return coverage_rate, avoidance_rate, car_score


