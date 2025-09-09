import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pinecone import Pinecone
from openai import OpenAI
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import re
import statistics
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

class LegalDomain(Enum):
    CONTRACT = "Contract Law"
    TORT = "Tort Law"
    EMPLOYMENT = "Employment Law"
    PUBLIC = "Public Law"
    INTELLECTUAL_PROPERTY = "Intellectual Property Law"
    CRIMINAL = "Criminal Law"
    PROPERTY = "Property Law"
    UNKNOWN = "Unknown"

class QueryType(Enum):
    LIABILITY = "Liability"
    REMEDIES = "Remedies"
    PROCEDURAL = "Procedural"
    CLASSIFICATION = "Classification"
    DAMAGES = "Damages"
    RIGHTS = "Rights"
    UNKNOWN = "Unknown"

@dataclass
class LegalMetadata:
    """Structured metadata for legal cases and queries"""
    domain: LegalDomain
    query_type: QueryType
    key_concepts: List[str]
    legal_principles: List[str]
    parties: List[str]
    remedy_sought: Optional[str]
    factual_keywords: List[str]
    procedural_context: Optional[str]
    case_complexity: str  # simple, moderate, complex

class MetadataRAGEvaluator:
    def __init__(self):
        self.pc = Pinecone(api_key="")
        self.index = self.pc.Index(host="")
        self.client = OpenAI()
        
    def extract_metadata_with_llm(self, scenario: str, question: str, law_type: str = "") -> LegalMetadata:
        """Extract structured metadata using OpenAI"""
        
        prompt = f"""
        You are a legal expert tasked with extracting structured metadata from legal scenarios and questions.
        
        Scenario: {scenario}
        Question: {question}
        Law Type: {law_type}
        
        Extract and return the following metadata in JSON format:
        {{
            "domain": "One of: Contract Law, Tort Law, Employment Law, Public Law, Intellectual Property Law, Criminal Law, Property Law, or Unknown",
            "query_type": "One of: Liability, Remedies, Procedural, Classification, Damages, Rights, or Unknown",
            "key_concepts": ["list", "of", "key", "legal", "concepts"],
            "legal_principles": ["list", "of", "relevant", "legal", "principles"],
            "parties": ["list", "of", "parties", "involved"],
            "remedy_sought": "what remedy or outcome is being sought (or null)",
            "factual_keywords": ["key", "factual", "elements"],
            "procedural_context": "procedural stage or context (or null)",
            "case_complexity": "simple, moderate, or complex"
        }}
        
        Focus on legal accuracy and be comprehensive but concise.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a legal expert specializing in case law analysis and metadata extraction."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            metadata_json = response.choices[0].message.content.strip()
            # Clean the response to extract JSON
            if "```json" in metadata_json:
                metadata_json = metadata_json.split("```json")[1].split("```")[0]
            elif "```" in metadata_json:
                metadata_json = metadata_json.split("```")[1]
            
            metadata_dict = json.loads(metadata_json)
            
            # Convert to structured metadata
            return LegalMetadata(
                domain=LegalDomain(metadata_dict.get("domain", "Unknown")),
                query_type=QueryType(metadata_dict.get("query_type", "Unknown")),
                key_concepts=metadata_dict.get("key_concepts", []),
                legal_principles=metadata_dict.get("legal_principles", []),
                parties=metadata_dict.get("parties", []),
                remedy_sought=metadata_dict.get("remedy_sought"),
                factual_keywords=metadata_dict.get("factual_keywords", []),
                procedural_context=metadata_dict.get("procedural_context"),
                case_complexity=metadata_dict.get("case_complexity", "moderate")
            )
            
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            # Return default metadata
            return LegalMetadata(
                domain=LegalDomain.UNKNOWN,
                query_type=QueryType.UNKNOWN,
                key_concepts=[],
                legal_principles=[],
                parties=[],
                remedy_sought=None,
                factual_keywords=[],
                procedural_context=None,
                case_complexity="moderate"
            )
    
    def enhance_query_with_metadata(self, query: str, metadata: LegalMetadata) -> List[str]:
        """Create enhanced queries using extracted metadata"""
        enhanced_queries = [query]  # Original query
        
        # Domain-specific enhancement
        if metadata.domain != LegalDomain.UNKNOWN:
            enhanced_queries.append(f"{metadata.domain.value} {query}")
        
        # Key concepts enhancement
        if metadata.key_concepts:
            concepts_str = " ".join(metadata.key_concepts[:3])
            enhanced_queries.append(f"{query} {concepts_str}")
            enhanced_queries.append(f"{concepts_str} {query}")
        
        # Legal principles enhancement
        if metadata.legal_principles:
            principles_str = " ".join(metadata.legal_principles[:2])
            enhanced_queries.append(f"{principles_str} {query}")
        
        # Query type specific enhancement
        if metadata.query_type != QueryType.UNKNOWN:
            if metadata.query_type == QueryType.LIABILITY:
                enhanced_queries.append(f"liability negligence fault {query}")
            elif metadata.query_type == QueryType.REMEDIES:
                enhanced_queries.append(f"damages compensation remedy {query}")
            elif metadata.query_type == QueryType.PROCEDURAL:
                enhanced_queries.append(f"procedure process court {query}")
        
        # Factual keywords enhancement
        if metadata.factual_keywords:
            facts_str = " ".join(metadata.factual_keywords[:3])
            enhanced_queries.append(f"{facts_str} {query}")
        
        return enhanced_queries[:6]  # Limit to 6 queries 
    
    def metadata_guided_search(self, query: str, metadata: LegalMetadata, namespace: str = "chunks", top_k: int = 5) -> Dict:
        """Perform metadata-guided search with multiple strategies"""
        enhanced_queries = self.enhance_query_with_metadata(query, metadata)
        
        all_results = []
        seen_uris = set()
        
        # Weight queries based on metadata confidence and type
        query_weights = [1.2, 1.0, 0.9, 0.8, 0.7, 0.6]  # Decreasing weights
        
        for i, enhanced_query in enumerate(enhanced_queries):
            try:
                results = self.baseline_search(enhanced_query, namespace, 3)
                weight = query_weights[i] if i < len(query_weights) else 0.5
                for hit in results['result']['hits']:
                    uri = hit['fields']['uri']
                    if uri not in seen_uris:
                        hit['_score'] = hit['_score'] * weight
                        hit['metadata_enhanced'] = True
                        all_results.append(hit)
                        seen_uris.add(uri)
                        
            except Exception as e:
                print(f"Error with enhanced query '{enhanced_query}': {e}")
                continue
        
        # Sort by weighted scores
        all_results.sort(key=lambda x: x['_score'], reverse=True)
        
        return {'result': {'hits': all_results[:top_k]}}
    
    def baseline_search(self, query: str, namespace: str, top_k: int = 5) -> Dict:
        """Baseline search without reranking"""
        results = self.index.search(
            namespace=namespace,
            query={"inputs": {"text": query}, "top_k": top_k},
            fields=["uri", "text", "name"]
        )
        return results
    
    def multi_query_search(self, queries: List[str], namespace: str, top_k: int) -> Dict:
        """Search with multiple queries and combine results"""
        all_results = []
        seen_uris = set()
        
        for query in queries:
            try:
                results = self.baseline_search(query, namespace, 3)
                for hit in results['result']['hits']:
                    uri = hit['fields']['uri']
                    if uri not in seen_uris:
                        all_results.append(hit)
                        seen_uris.add(uri)
            except Exception as e:
                print(f"Error with query '{query}': {e}")
                continue
        
        all_results.sort(key=lambda x: x['_score'], reverse=True)
        return {'result': {'hits': all_results[:top_k]}}

    def evaluate_retrieval(self, query: str, expected_uris: List[str], metadata: LegalMetadata, namespace: str = "chunks", top_k: int = 5) -> Tuple[List[str], float, float, float, float]:
        """Evaluate retrieval results against expected URIs"""
        retrieved_results = self.metadata_guided_search(query, metadata, namespace, top_k)
        
        retrieved_uris = [hit['fields']['uri'] for hit in retrieved_results['result']['hits']]
        
        precision, recall = self.calculate_precision_recall(expected_uris, retrieved_uris)
        f1_score = self.calculate_f1_score(precision, recall)
        total_score = self.calculate_total_score(expected_uris, retrieved_uris, [hit['_score'] for hit in retrieved_results['result']['hits']])
        
        print(f"Expected URIs: {expected_uris}")
        print(f"Retrieved URIs: {retrieved_uris}")
        
        return retrieved_uris, precision, recall, f1_score, total_score

    def calculate_precision_recall(self, expected_uris: List[str], retrieved_uris: List[str]) -> Tuple[float, float]:
        """Calculate precision and recall"""
        if not expected_uris:
            return 0.0, 0.0

        true_positive = len(set(expected_uris) & set(retrieved_uris))
        false_positive = len(set(retrieved_uris) - set(expected_uris))
        false_negative = len(set(expected_uris) - set(retrieved_uris))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

        return precision, recall
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_total_score(self, expected_uris: List[str], retrieved_uris: List[str], scores: List[float]) -> float:
        """Calculate total score based on expected URIs and their scores"""
        total_score = 0
        for uri in expected_uris:
            if uri in retrieved_uris:
                index = retrieved_uris.index(uri)
                total_score += scores[index] if index < len(scores) else 0
        return total_score
    
    def evaluate_all(self, data: List[Dict[str, Any]], namespace: str = "chunks", top_k: int = 5) -> List[Dict[str, Any]]:
        """Evaluate all queries in the provided data"""
        results = []
        for item in data:
            query = item['question']
            expected_uris = [i.get("uri") for i in item["relevant_cases"]]
            scenario = item.get('scenario', "")
            law_type = item.get('law_type', "")
            
            metadata = self.extract_metadata_with_llm(scenario, query, law_type)
            query = scenario + " " + query  # Combine scenario and question
            print(f"Query: {query}")
            retrieved_uris, precision, recall, f1_score, total_score = self.evaluate_retrieval(query, expected_uris, metadata, namespace, top_k)
            
            results.append({
                'query': query,
                'expected_uris': expected_uris,
                'retrieved_uris': retrieved_uris,
                'metadata': metadata,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'total_score': total_score
            })
        return results
    
    def compute_global_metrics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute total TP, FP, TN, FN and global precision, recall, F1"""
        total_tp = total_fp = total_fn = 0

        for res in evaluation_results:
            expected = set(res["expected_uris"])
            retrieved = set(res["retrieved_uris"])

            tp = len(expected & retrieved)
            fp = len(retrieved - expected)
            fn = len(expected - retrieved)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Precision, Recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

if __name__ == "__main__":
    evaluator = MetadataRAGEvaluator()
    test_data = load_json("evaluation/evaluation3/eval3.json")  # Load your test queries with expected URIs
    evaluation_results = evaluator.evaluate_all(test_data)
    
    #get average precision, recall, f1_score, total_score
    avg_precision = statistics.mean([res['precision'] for res in evaluation_results])
    avg_recall = statistics.mean([res['recall'] for res in evaluation_results])
    avg_f1_score = statistics.mean([res['f1_score'] for res in evaluation_results])
    avg_total_score = statistics.mean([res['total_score'] for res in evaluation_results])

    global_metrics = evaluator.compute_global_metrics(evaluation_results)

    print("\n--- Global Metrics ---")
    print(f"True Positives: {global_metrics['true_positives']}")
    print(f"False Positives: {global_metrics['false_positives']}")
    print(f"False Negatives: {global_metrics['false_negatives']}")
    print(f"Global Precision: {global_metrics['precision']:.4f}")
    print(f"Global Recall: {global_metrics['recall']:.4f}")
    print(f"Global F1 Score: {global_metrics['f1_score']:.4f}")
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"evaluation/evaluation4/metadata_enchancer/metadata_enhanced_evaluation_{timestamp}.json", 'w') as f:
        json.dump([{
            'query': res['query'],
            'expected_uris': res['expected_uris'],
            'retrieved_uris': res['retrieved_uris'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1_score': res['f1_score'],
            'total_score': res['total_score']
        } for res in evaluation_results], f, indent=4)
        #add average precision, recall, f1_score, total_score to the end of the json file
        json.dump({
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'average_f1_score': avg_f1_score,
            'average_total_score': avg_total_score
        }, f, indent=4)
        json.dump(global_metrics, f, indent=4)



    print(f"Evaluation results saved to metadata_enhanced_evaluation_{timestamp}.json")