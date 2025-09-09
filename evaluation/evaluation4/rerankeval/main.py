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
    
    def metadata_guided_search(self, query: str, metadata: LegalMetadata, namespace: str = "chunks", top_k: int = 5, reranker = "no_rerank") -> Dict:
        """Perform metadata-guided search with multiple strategies"""
        enhanced_queries = self.enhance_query_with_metadata(query, metadata)
        
        all_results = []
        seen_uris = set()
        
        # Weight queries based on metadata confidence and type
        query_weights = [1.2, 1.0, 0.9, 0.8, 0.7, 0.6]  # Decreasing weights
        
        for i, enhanced_query in enumerate(enhanced_queries):
            try:
                results = self.baseline_search(enhanced_query, namespace, 3, reranker)
                weight = query_weights[i] if i < len(query_weights) else 0.5
                
                for hit in results['result']['hits']:
                    uri = hit['fields']['uri']
                    if uri not in seen_uris:
                        hit['_score'] = hit['_score'] * weight
                        hit['metadata_enhanced'] = True
                        all_results.append(hit)
                        seen_uris.add(uri)
                        
            except Exception as e:
                print("Error:", e)
                continue
        
        # Sort by weighted scores
        all_results.sort(key=lambda x: x['_score'], reverse=True)
        
        return {'result': {'hits': all_results[:top_k]}}
    
    def baseline_search(self, query: str, namespace: str, top_k: int = 5, reranker ="cohere-rerank-3.5") -> Dict:
        """Baseline search without reranking"""
        if reranker != "no_rerank":
            results = self.rerank_search(query, namespace, top_k, reranker)
        else:
            results = self.index.search(
                namespace=namespace,
                query={"inputs": {"text": query}, "top_k": top_k},
                fields=["uri", "text", "name"]
            )
        return results
    
    def rerank_search(self, query: str, namespace: str, top_k: int = 5 , reranker ="cohere-rerank-3.5"):
        """
        Query the Pinecone index with the given query string and return the top_k results.
        """
        if reranker == "pinecone-rerank-v0":
            query = query[:450] 
        results = self.index.search(
            namespace=namespace, 
            query={"inputs": {"text": query}, "top_k": top_k + 10}, 
            rerank={
                "model": reranker,
                "top_n": top_k,
                "rank_fields": ["text"]
            },
            fields=["uri", "text", "name"])
        return  results
    
    def multi_query_search(self, queries: List[str], namespace: str, top_k: int, reranker ="cohere-rerank-3.5") -> Dict:
        """Search with multiple queries and combine results"""
        all_results = []
        seen_uris = set()
        
        for query in queries:
            try:
                results = self.baseline_search(query, namespace, 3, reranker)
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
    
    def calculate_f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_precision_recall(self, expected_uris, retrieved_uris):
        if not expected_uris:
            return 0.0, 0.0
        
        true_positive = len(set(expected_uris) & set(retrieved_uris))
        false_positive = len(set(retrieved_uris) - set(expected_uris))
        false_negative = len(set(expected_uris) - set(retrieved_uris))
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        
        return precision, recall
    
    def evaluate_technique(self, data: List[Dict], technique_name: str, technique_func, reranker, top_k) -> List[Dict]:
        """Evaluate a specific metadata-based technique"""
        results = []
        
        for item in data:
            try:
                query = item['scenario'] + " " + item['question']
                metadata = self.extract_metadata_with_llm(item['scenario'], item['question'], item.get('law_type', ''))
                retrieved_results = technique_func(query, metadata, reranker=reranker, top_k=top_k)
                
                # Extract URIs and calculate metrics
                retrieved_uris = [hit['fields']["uri"] for hit in retrieved_results['result']['hits']]
                expected_uris = [case['uri'] for case in item['relevant_cases'] if case["uri"] != ""]
                
                precision, recall = self.calculate_precision_recall(expected_uris, retrieved_uris)
                f1_score = self.calculate_f1_score(precision, recall)
                
                results.append({
                    'query': item['scenario'] + " " + item['question'],
                    'expected_uri': expected_uris,
                    'retrieved_uri': retrieved_uris,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'technique': technique_name
                })
                
                print(f"Processed query {len(results)}/{len(data)} for {technique_name}")
                
            except Exception as e:
                print(f"Error evaluating item {len(results)+1} with {technique_name}: {e}")
                results.append({
                    'query': item['scenario'] + " " + item['question'],
                    'expected_uri': [case['uri'] for case in item['relevant_cases'] if case["uri"] != ""],
                    'retrieved_uri': [],
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'technique': technique_name
                })
        
        return results


if __name__ == "__main__":
    print("Starting Metadata-Based RAG Evaluation...")
    
    # Initialize evaluator
    evaluator = MetadataRAGEvaluator()
    
    # Load evaluation data
    data = load_json('evaluation/evaluation3/eval3.json')
    
    # Define metadata-based techniques to test
    techniques = {
        "metadata_guided": evaluator.metadata_guided_search,
    }
    rerankers = ["no_rerank", "bge-reranker-v2-m3", "cohere-rerank-3.5", "pinecone-rerank-v0"]
    k_values = [3, 5, 7, 10]
    
    # Store metadata information for analysis
    metadata_info = {
        'extraction_start_time': datetime.now(),
        'total_queries': len(data),
        'metadata_quality': {}
    }
    
    print(f"Evaluating {len(techniques)} techniques on {len(data)} queries...")
    
    # Evaluate each technique
    all_results = {}
    for reranker in rerankers:
        for top_k in k_values:
            for technique_name, technique_func in techniques.items():
                print(f"\n{'='*50}")
                print(f"Using reranker: {reranker}, Top K: {top_k}")
                print(f"{'='*50}")
                
                start_time = time.time()
                # Evaluate technique for current reranker and k
                results = evaluator.evaluate_technique(data, technique_name, technique_func, reranker, top_k)
                end_time = time.time()
                
                # Store results in a nested dictionary by reranker and k value
                if reranker not in all_results:
                    all_results[reranker] = {}
                all_results[reranker][top_k] = results
                
                # Calculate summary metrics for the current combination
                avg_f1 = np.mean([r['f1_score'] for r in results])
                avg_precision = np.mean([r['precision'] for r in results])
                avg_recall = np.mean([r['recall'] for r in results])
                success_rate = sum(1 for r in results if r['f1_score'] > 0) / len(results)
                
                print(f"\nResults for {technique_name} with reranker {reranker} at top_k={top_k}:")
                print(f"  F1-Score: {avg_f1:.4f}")
                print(f"  Precision: {avg_precision:.4f}")
                print(f"  Recall: {avg_recall:.4f}")
                print(f"  Success Rate: {success_rate:.1%}")
                print(f"  Processing Time: {end_time - start_time:.1f}s")
            
            # Generate comprehensive report for each reranker and k value combination
            print(f"\n{'='*50}")
            print("Generating comprehensive analysis...")
            print(f"{'='*50}")
            
            metadata_info['extraction_end_time'] = datetime.now()
            
            # Save detailed results for each reranker and k value
            with open(f'evaluation/evaluation4/rerankeval/metadata_rag_detailed_results_{reranker}_k{top_k}.json', 'w') as f:
                json.dump({
                    technique: {
                        'results': results,
                        'summary': {
                            'avg_precision': np.mean([r['precision'] for r in results]),
                            'avg_recall': np.mean([r['recall'] for r in results]),
                            'avg_f1': np.mean([r['f1_score'] for r in results]),
                            'success_rate': sum(1 for r in results if r['f1_score'] > 0) / len(results),
                            'std_f1': np.std([r['f1_score'] for r in results])
                        }
                    } for technique, results in all_results[reranker].items()
                }, f, indent=2)
            
            print(f"\nEvaluation for reranker {reranker} at k={top_k} complete!")
    
    print(f"\nMetadata-based RAG evaluation complete!")
    print(f"Files generated:")
    print(f"  - metadata_rag_detailed_results_*.json")
    
    # Print final summary
    best_technique = max(all_results.keys(), key=lambda x: np.mean([r['f1_score'] for r in all_results[x]]))
    best_f1 = np.mean([r['f1_score'] for r in all_results[best_technique]])
    
    print(f"\nBest performing technique: {best_technique}")
    print(f"Best F1-score achieved: {best_f1:.4f}")
