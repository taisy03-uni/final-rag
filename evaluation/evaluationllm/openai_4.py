#!/usr/bin/env python3
"""
OpenAI Case Law Retrieval and Evaluation Script
Uses OpenAI API to find relevant case law for legal scenarios and evaluates accuracy.
"""

import json
import os
from openai import OpenAI
from typing import List, Dict, Any
import re
from datetime import datetime


def create_case_law_prompt(scenario: str, question: str, law_type: str) -> str:
    """
    Create a focused prompt for OpenAI to find relevant case law.
    """
    prompt = f"""
You are a legal research assistant specializing in UK case law. Your task is to identify the most relevant UK court cases for the given legal scenario.

SCENARIO: {scenario}

LEGAL QUESTION: {question}

AREA OF LAW: {law_type}

REQUIREMENTS:
1. Find 2-5 highly relevant UK court cases from 2000 onwards
2. Focus on cases from England and Wales Court of Appeal (EWCA), High Court (EWHC), Supreme Court (UKSC)
3. Each case must be from year 2000 or later
4. Return ONLY case citations in the exact format: [YEAR] COURT NUMBER
5. Ensure cases are directly relevant to the legal principles involved

RESPONSE FORMAT:
Return your answer as a JSON object with this exact structure:
{{
    "relevant_cases": [
        {{
            "case_id": "[YEAR] COURT NUMBER",
            "relevance_reason": "Brief explanation of why this case is relevant"
        }}
    ]
}}

EXAMPLES of correct case citation formats:
- [2023] EWCA Civ 1234
- [2022] EWHC 567 (QB)
- [2021] UKSC 45

Focus on landmark cases and recent precedents that directly address the legal issues in the scenario. Be precise and authoritative.
"""
    return prompt

def query_openai_for_cases(scenario: str, question: str, law_type: str) -> Dict[str, Any]:
    """
    Query OpenAI API for relevant case law.
    """
    try:
        prompt = create_case_law_prompt(scenario, question, law_type)
            
        response = client.responses.create(
            model="gpt-4.1",  
            input=[
                {"role": "system", "content": "You are an expert UK legal researcher with comprehensive knowledge of case law. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_output_tokens=1000
        )

        # Extract the text output
        response_text = response.output_text
        
        # Try to extract JSON from the response
        try:
            # Look for JSON block in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response_text)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response_text}")
            return {"relevant_cases": [], "error": "Failed to parse JSON"}
            
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return {"relevant_cases": [], "error": str(e)}

def validate_case_format(case_id: str) -> bool:
    """
    Validate that case_id follows correct UK court citation format and is from 2000+.
    """
    # Pattern for UK case citations: [YYYY] COURT_CODE NUMBER
    pattern = r'\[(\d{4})\]\s+(EWCA|EWHC|UKSC|UKHL|EWCOP|UKUT|UKFTT)\s+'
    match = re.match(pattern, case_id)
    
    if not match:
        return False
    
    year = int(match.group(1))
    return year >= 2000

def calculate_accuracy_metrics(predicted_cases: List[Dict], ground_truth_cases: List[Dict]) -> Dict[str, float]:
    """
    Calculate accuracy metrics by comparing predicted cases with ground truth.
    """
    # Extract case IDs for comparison
    predicted_ids = {case.get("case_id", "").strip() for case in predicted_cases}
    ground_truth_ids = {case.get("case_id", "").strip() for case in ground_truth_cases}
    
    # Remove empty strings
    predicted_ids.discard("")
    ground_truth_ids.discard("")
    
    # Calculate metrics
    true_positives = len(predicted_ids.intersection(ground_truth_ids))
    false_positives = len(predicted_ids - ground_truth_ids)
    false_negatives = len(ground_truth_ids - predicted_ids)
    
    precision = true_positives / len(predicted_ids) if len(predicted_ids) > 0 else 0
    recall = true_positives / len(ground_truth_ids) if len(ground_truth_ids) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_count": len(predicted_ids),
        "ground_truth_count": len(ground_truth_ids)
    }

def evaluate_single_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single query from the dataset.
    """
    scenario = query_data.get("scenario", "")
    question = query_data.get("question", "")
    law_type = query_data.get("law_type", "")
    ground_truth_cases = query_data.get("relevant_cases", [])
    query_num = query_data.get("num", 0)
    
    print(f"\n--- Evaluating Query {query_num} ---")
    print(f"Law Type: {law_type}")
    print(f"Question: {question[:100]}...")
    
    # Query OpenAI for cases
    openai_response = query_openai_for_cases(scenario, question, law_type)
    predicted_cases = openai_response.get("relevant_cases", [])
    
    # Validate case formats
    valid_cases = []
    invalid_cases = []
    
    for case in predicted_cases:
        case_id = case.get("case_id", "")
        if validate_case_format(case_id):
            valid_cases.append(case)
        else:
            invalid_cases.append(case)
    
    # Calculate accuracy metrics
    metrics = calculate_accuracy_metrics(valid_cases, ground_truth_cases)
    
    print(f"OpenAI found {len(predicted_cases)} cases ({len(valid_cases)} valid, {len(invalid_cases)} invalid)")
    print(f"Ground truth has {len(ground_truth_cases)} cases")
    print(f"Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1_score']:.2f}")
    
    return {
        "query_num": query_num,
        "law_type": law_type,
        "scenario": scenario[:200] + "..." if len(scenario) > 200 else scenario,
        "question": question,
        "predicted_cases": predicted_cases,
        "valid_cases": valid_cases,
        "invalid_cases": invalid_cases,
        "ground_truth_cases": ground_truth_cases,
        "metrics": metrics,
        "openai_response": openai_response
    }

def run_full_evaluation(eval_file_path: str) -> Dict[str, Any]:
    """
    Run evaluation on the full dataset.
    """
    with open(eval_file_path, 'r') as f:
        eval_data = json.load(f)
    
    results = []
    total_metrics = {
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "predicted_count": 0,
        "ground_truth_count": 0
    }
    
    print(f"Starting evaluation of {len(eval_data)} queries...")
    
    for query_data in eval_data:
        result = evaluate_single_query(query_data)
        results.append(result)
        
        # Accumulate metrics
        metrics = result["metrics"]
        for key in total_metrics:
            total_metrics[key] += metrics[key]
    
    # Calculate average metrics
    num_queries = len(eval_data)
    avg_metrics = {
        "avg_precision": total_metrics["precision"] / num_queries,
        "avg_recall": total_metrics["recall"] / num_queries,
        "avg_f1_score": total_metrics["f1_score"] / num_queries,
        "total_true_positives": total_metrics["true_positives"],
        "total_false_positives": total_metrics["false_positives"],
        "total_false_negatives": total_metrics["false_negatives"],
        "total_predicted": total_metrics["predicted_count"],
        "total_ground_truth": total_metrics["ground_truth_count"]
    }
    
    # Calculate overall precision, recall, F1
    if total_metrics["predicted_count"] > 0:
        overall_precision = total_metrics["true_positives"] / total_metrics["predicted_count"]
    else:
        overall_precision = 0
        
    if total_metrics["ground_truth_count"] > 0:
        overall_recall = total_metrics["true_positives"] / total_metrics["ground_truth_count"]
    else:
        overall_recall = 0
        
    if (overall_precision + overall_recall) > 0:
        overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall)
    else:
        overall_f1 = 0
    
    avg_metrics.update({
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1
    })
    
    return {
        "timestamp": datetime.now().isoformat(),
        "total_queries": num_queries,
        "individual_results": results,
        "summary_metrics": avg_metrics
    }

def main():
    """
    Main execution function.
    """
    # Path to the evaluation dataset
    eval_file_path = "evaluation/evaluation3/eval3.json"
    
    print("OpenAI Case Law Evaluation System")
    print("=" * 50)
    print(f"Using evaluation dataset: {eval_file_path}")
    print(f"OpenAI API Key configured: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
    
    # Run the evaluation
    evaluation_results = run_full_evaluation(eval_file_path)
    
    # Save results
    output_file = "evaluation/evaluationllm/gpt-5-mini-2025-08-07.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    metrics = evaluation_results["summary_metrics"]
    print(f"Total Queries Evaluated: {evaluation_results['total_queries']}")
    print(f"Average Precision: {metrics['avg_precision']:.3f}")
    print(f"Average Recall: {metrics['avg_recall']:.3f}")
    print(f"Average F1 Score: {metrics['avg_f1_score']:.3f}")
    print(f"Overall Precision: {metrics['overall_precision']:.3f}")
    print(f"Overall Recall: {metrics['overall_recall']:.3f}")
    print(f"Overall F1 Score: {metrics['overall_f1']:.3f}")
    print(f"Total Cases Predicted: {metrics['total_predicted']}")
    print(f"Total Ground Truth Cases: {metrics['total_ground_truth']}")
    print(f"Total Correct Matches: {metrics['total_true_positives']}")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = ""

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    main()