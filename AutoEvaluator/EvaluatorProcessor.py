# import re
# import os
# from tqdm import tqdm
# from typing import List, Dict, Tuple
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from dataclasses import dataclass
# from AutoEvaluator import AutoEvaluator

# @dataclass
# class EvaluationResult:
#     e1_result: bool
#     e2_result: bool
#     e3_result: bool
#     coverage_rate: float
#     avoidance_rate: float
#     car_score: float

# class EvaluationProcessor:
#     def __init__(self, reference_list: List[str], similarity_threshold: float = 0.8,
#                  e1_similarity_threshold: float = 0.9, max_workers: int = None):
#         self.evaluator = AutoEvaluator(reference_list, similarity_threshold, e1_similarity_threshold)
#         self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 4)

#     def process_sample(self, args: Tuple[Dict, str, List[str], List[str], int, int, float]) -> EvaluationResult:
#         """
#         Process a single sample with all metrics including CAR
        
#         Args:
#             args: Tuple containing (sample, ground_truth, ground_truth_top_k, bottom_predictions, k, q, lambda_weight)
#         """
#         sample, ground_truth, ground_truth_top_k, bottom_predictions, k, q, lambda_weight = args
#         try:
#             output_string = sample
            
#             e1_result = self.evaluator.E1(self.evaluator.extract_diseases(output_string))
#             e2_result, _, _, _ = self.evaluator.E2(output_string, ground_truth)
#             e3_result, _, _, _, _ = self.evaluator.E3(output_string, ground_truth)
            
#             coverage_rate, avoidance_rate, car_score = self.evaluator.calculate_car(
#                 output_string, 
#                 ground_truth_top_k, 
#                 bottom_predictions, 
#                 k, 
#                 q, 
#                 lambda_weight
#             )
            
#             return EvaluationResult(
#                 e1_result=e1_result,
#                 e2_result=e2_result,
#                 e3_result=e3_result,
#                 coverage_rate=coverage_rate,
#                 avoidance_rate=avoidance_rate,
#                 car_score=car_score
#             )
            
#         except Exception as e:
#             print(f"Error processing sample: {str(e)}")
#             return EvaluationResult(
#                 e1_result=False,
#                 e2_result=False,
#                 e3_result=False,
#                 coverage_rate=0.0,
#                 avoidance_rate=0.0,
#                 car_score=0.0
#             )

#     def evaluate_samples(self, 
#                         inference_list: List[Dict], 
#                         ground_truth_list: List[str],
#                         ground_truth_top_k_list: List[List[str]], 
#                         bottom_predictions_list: List[List[str]],
#                         k: int,
#                         q: int,
#                         lambda_weight: float = 0.5) -> Dict[str, float]:
#         """
#         Evaluate samples using multiple threads
        
#         Args:
#             inference_list: List of model outputs
#             ground_truth_list: List of full ground truth strings
#             ground_truth_top_k_list: List of ground truth top k lists
#             bottom_predictions_list: List of bottom q predictions
#             k: Number of top predictions to consider
#             q: Number of bottom predictions to consider
#             lambda_weight: Weight for CAR calculation
#         """
#         correct_E1, correct_E2, correct_E3 = 0, 0, 0
#         total_coverage, total_avoidance, total_car = 0.0, 0.0, 0.0
#         total_samples = len(inference_list)

#         sample_args = [
#             (sample, ground_truth, gt_top_k, bottom_preds, k, q, lambda_weight)
#             for sample, ground_truth, gt_top_k, bottom_preds 
#             in zip(inference_list, ground_truth_list, ground_truth_top_k_list, bottom_predictions_list)
#         ]
        
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             future_to_sample = {
#                 executor.submit(self.process_sample, args): args
#                 for args in sample_args
#             }
            
#             with tqdm(total=total_samples, desc="Evaluating samples") as pbar:
#                 for future in as_completed(future_to_sample):
#                     try:
#                         result = future.result()
                        
#                         correct_E1 += int(result.e1_result or result.e2_result or result.e3_result)
#                         correct_E2 += int(result.e2_result or result.e3_result)
#                         correct_E3 += int(result.e3_result)

#                         total_coverage += result.coverage_rate
#                         total_avoidance += result.avoidance_rate
#                         total_car += result.car_score
                        
#                     except Exception as e:
#                         print(f"Error processing result: {str(e)}")
#                     pbar.update(1)

#         return {
#             "E1_score": correct_E1 / total_samples,
#             "E2_score": correct_E2 / total_samples,
#             "E3_score": correct_E3 / total_samples,
#             "coverage_rate": total_coverage / total_samples,
#             "avoidance_rate": total_avoidance / total_samples,
#             "car_score": total_car / total_samples
#         }
import re
import os
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from AutoEvaluator import AutoEvaluator


@dataclass
class EvaluationResult:
    e1_result: bool
    e2_result: bool
    e3_result: bool
    coverage_rate: Optional[float] = None
    avoidance_rate: Optional[float] = None
    car_score: Optional[float] = None


class EvaluationProcessor:
    def __init__(self, reference_list: List[str], similarity_threshold: float = 0.8,
                 e1_similarity_threshold: float = 0.0, max_workers: int = None):
        self.evaluator = AutoEvaluator(reference_list, similarity_threshold, e1_similarity_threshold)
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 4)

    def process_sample(self, args: Tuple[Dict, str, Optional[List[str]], Optional[List[str]], 
                                       Optional[int], Optional[int], float, bool, bool, bool]) -> EvaluationResult:

        (sample, ground_truth, ground_truth_top_k, bottom_predictions, 
         k, q, lambda_weight, calc_coverage, calc_avoidance, calc_car) = args
        
        try:
            output_string = sample
        
            e1_result = self.evaluator.E1(self.evaluator.extract_diseases(output_string))
            e2_result, _, _, _ = self.evaluator.E2(output_string, ground_truth)
            e3_result, _, _, _, _ = self.evaluator.E3(output_string, ground_truth)
            
            coverage_rate, avoidance_rate, car_score = None, None, None
            
            if any([calc_coverage, calc_avoidance, calc_car]):
                coverage_rate, avoidance_rate, car_score = self.evaluator.calculate_optional_metrics(
                    output_string,
                    ground_truth_top_k if calc_coverage or calc_car else None,
                    bottom_predictions if calc_avoidance or calc_car else None,
                    k if calc_coverage or calc_car else None,
                    q if calc_avoidance or calc_car else None,
                    lambda_weight if calc_car else 0.5
                )
            
            return EvaluationResult(
                e1_result=e1_result,
                e2_result=e2_result,
                e3_result=e3_result,
                coverage_rate=coverage_rate if calc_coverage else None,
                avoidance_rate=avoidance_rate if calc_avoidance else None,
                car_score=car_score if calc_car else None
            )
            
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            return EvaluationResult(
                e1_result=False,
                e2_result=False,
                e3_result=False
            )

    def evaluate_samples(self,
                        inference_list: List[Dict],
                        ground_truth_list: List[str],
                        ground_truth_top_k_list: Optional[List[List[str]]] = None,
                        bottom_predictions_list: Optional[List[List[str]]] = None,
                        k: Optional[int] = None,
                        q: Optional[int] = None,
                        lambda_weight: float = 0.5,
                        calc_coverage: bool = False,
                        calc_avoidance: bool = False,
                        calc_car: bool = False) -> Dict[str, float]:
        """
        评估多个样本
        """
        correct_E1, correct_E2, correct_E3 = 0, 0, 0
        total_coverage = 0.0 if calc_coverage else None
        total_avoidance = 0.0 if calc_avoidance else None
        total_car = 0.0 if calc_car else None
        total_samples = len(inference_list)

        sample_args = [
            (sample, ground_truth, 
             ground_truth_top_k_list[i] if ground_truth_top_k_list else None,
             bottom_predictions_list[i] if bottom_predictions_list else None,
             k, q, lambda_weight,
             calc_coverage, calc_avoidance, calc_car)
            for i, (sample, ground_truth)
            in enumerate(zip(inference_list, ground_truth_list))
        ]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sample = {
                executor.submit(self.process_sample, args): args
                for args in sample_args
            }
            
            with tqdm(total=total_samples, desc="Evaluating samples") as pbar:
                for future in as_completed(future_to_sample):
                    try:
                        result = future.result()
                        
                        correct_E1 += int(result.e1_result or result.e2_result or result.e3_result)
                        correct_E2 += int(result.e2_result or result.e3_result)
                        correct_E3 += int(result.e3_result)

                        if calc_coverage and result.coverage_rate is not None:
                            total_coverage += result.coverage_rate
                        if calc_avoidance and result.avoidance_rate is not None:
                            total_avoidance += result.avoidance_rate
                        if calc_car and result.car_score is not None:
                            total_car += result.car_score
                        
                    except Exception as e:
                        print(f"Error processing result: {str(e)}")
                    pbar.update(1)

        # 构建结果字典
        results = {
            "E1_score": correct_E1 / total_samples,
            "E2_score": correct_E2 / total_samples,
            "E3_score": correct_E3 / total_samples,
        }
        
        # 添加可选的评估结果
        if calc_coverage:
            results["coverage_rate"] = total_coverage / total_samples if total_coverage is not None else 0.0
        if calc_avoidance:
            results["avoidance_rate"] = total_avoidance / total_samples if total_avoidance is not None else 0.0
        if calc_car:
            results["car_score"] = total_car / total_samples if total_car is not None else 0.0

        return results


