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
    
    def metadata_guided_search(self, query: str, metadata: LegalMetadata, namespace: str = "summary", top_k: int = 5) -> Dict:
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
    
    def llm_query_classification_search(self, scenario: str, question: str, law_type: str = "") -> Dict:
        """Use LLM to classify query intent and search accordingly"""
        
        classification_prompt = f"""
        Analyze this legal query and classify the search intent:
        
        Scenario: {scenario}
        Question: {question}
        Law Type: {law_type}
        
        Classify the query and suggest 3-5 specific search terms that would best find relevant case law.
        Consider:
        1. What type of legal precedent is needed?
        2. What specific legal tests or principles apply?
        3. What factual patterns are most relevant?
        
        Return JSON format:
        {{
            "priority_terms": ["term1", "term2", "term3"],
            "alternative_terms": ["alt1", "alt2"],
            "focus_area": "specific area to focus search"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a legal research specialist expert in case law search strategies."},
                    {"role": "user", "content": classification_prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            classification_json = response.choices[0].message.content.strip()
            if "```json" in classification_json:
                classification_json = classification_json.split("```json")[1].split("```")[0]
            elif "```" in classification_json:
                classification_json = classification_json.split("```")[1]
                
            classification = json.loads(classification_json)
            
            # Build queries based on classification
            queries = []
            base_query = f"{scenario} {question}"
            
            # Priority terms query
            if classification.get("priority_terms"):
                priority_query = f"{' '.join(classification['priority_terms'])} {base_query}"
                queries.append(priority_query)
            
            # Focus area query
            if classification.get("focus_area"):
                focus_query = f"{classification['focus_area']} {base_query}"
                queries.append(focus_query)
            
            # Alternative terms queries
            if classification.get("alternative_terms"):
                for alt_term in classification["alternative_terms"][:2]:
                    alt_query = f"{alt_term} {base_query}"
                    queries.append(alt_query)
            
            # Original query
            queries.append(base_query)
            
            # Search with all queries
            return self.multi_query_search(queries, "summary", 5)
            
        except Exception as e:
            print(f"Error in LLM classification search: {e}")
            # Fallback to regular search
            return self.baseline_search(f"{scenario} {question}", "summary", 5)
    
    def semantic_metadata_filtering(self, query: str, metadata: LegalMetadata) -> Dict:
        """Filter search results using semantic metadata matching"""
        
        # First, get broader results
        initial_results = self.baseline_search(query, "summary", 10)
        
        if not initial_results['result']['hits']:
            return initial_results
        
        # Use LLM to score relevance based on metadata
        scoring_prompt = f"""
        You are evaluating case law relevance. Given this query metadata:
        
        Domain: {metadata.domain.value}
        Query Type: {metadata.query_type.value}
        Key Concepts: {', '.join(metadata.key_concepts)}
        Legal Principles: {', '.join(metadata.legal_principles)}
        Parties: {', '.join(metadata.parties)}
        
        Score each case text for relevance (0-10 scale) based on how well it matches the query metadata.
        Focus on legal concepts, domain match, and principle alignment.
        
        Cases to score:
        """
        
        scored_results = []
        
        try:
            # Score results in batches
            for i, hit in enumerate(initial_results['result']['hits']):
                case_text = hit['fields'].get('text', '')  # Truncate for API limits
                individual_prompt = f"{scoring_prompt}\nCase {i+1}: {case_text}\n\nRelevance score (0-10):"
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a legal expert scoring case relevance. Return only a number 0-10."},
                        {"role": "user", "content": individual_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                
                try:
                    score = float(response.choices[0].message.content.strip())
                    hit['llm_relevance_score'] = score
                    hit['_score'] = hit['_score'] * (score / 10.0)  # Adjust original score
                    scored_results.append(hit)
                except ValueError:
                    # If LLM doesn't return a number, use original score
                    hit['llm_relevance_score'] = 5.0
                    scored_results.append(hit)
                
                # Rate limiting
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error in semantic filtering: {e}")
            scored_results = initial_results['result']['hits']
        
        # Sort by adjusted scores
        scored_results.sort(key=lambda x: x['_score'], reverse=True)
        
        return {'result': {'hits': scored_results[:5]}}
    
    def adaptive_metadata_search(self, scenario: str, question: str, law_type: str = "") -> Dict:
        """Adaptive search that combines multiple metadata techniques"""
        
        # Extract metadata
        metadata = self.extract_metadata_with_llm(scenario, question, law_type)
        query = f"{scenario} {question}"
        
        # Strategy selection based on metadata
        if metadata.case_complexity == "complex" or len(metadata.key_concepts) > 5:
            # Use comprehensive approach for complex cases
            results1 = self.metadata_guided_search(query, metadata, "summary", 3)
            results2 = self.llm_query_classification_search(scenario, question, law_type)
            
            # Combine results
            combined_uris = set()
            combined_hits = []
            
            # Add results with weighting
            for hit in results1['result']['hits']:
                uri = hit['fields']['uri']
                if uri not in combined_uris:
                    hit['_score'] = hit['_score'] * 1.1  # Boost metadata-guided
                    combined_hits.append(hit)
                    combined_uris.add(uri)
            
            for hit in results2['result']['hits']:
                uri = hit['fields']['uri']
                if uri not in combined_uris:
                    combined_hits.append(hit)
                    combined_uris.add(uri)
            
            combined_hits.sort(key=lambda x: x['_score'], reverse=True)
            return {'result': {'hits': combined_hits[:5]}}
            
        elif metadata.case_complexity == "simple":
            # Use simpler approach for straightforward cases
            return self.metadata_guided_search(query, metadata, "summary", 5)
        
        else:
            # Medium complexity - use classification-based approach
            return self.llm_query_classification_search(scenario, question, law_type)
    
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
    
    def evaluate_technique(self, data: List[Dict], technique_name: str, technique_func) -> List[Dict]:
        """Evaluate a specific metadata-based technique"""
        results = []
        
        for item in data:
            try:
                if technique_name in ["metadata_guided", "semantic_filtering"]:
                    # These need metadata extraction first
                    query = item['scenario'] + " " + item['question']
                    metadata = self.extract_metadata_with_llm(item['scenario'], item['question'], item.get('law_type', ''))
                    
                    if technique_name == "metadata_guided":
                        retrieved_results = technique_func(query, metadata)
                    else:  # semantic_filtering
                        retrieved_results = technique_func(query, metadata)
                        
                elif technique_name in ["llm_classification", "adaptive_metadata"]:
                    retrieved_results = technique_func(item['scenario'], item['question'], item.get('law_type', ''))
                else:
                    # Baseline techniques
                    query = item['scenario'] + " " + item['question']
                    if technique_name == "baseline_chunks":
                        retrieved_results = technique_func(query, "chunks")
                    elif technique_name == "baseline_summary":
                        retrieved_results = technique_func(query, "summary")
                    else:
                        retrieved_results = technique_func(query, "summary")
                
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


def create_metadata_visualizations(all_results: Dict[str, List[Dict]], metadata_info: Dict[str, Any]):
    """Create visualizations for metadata-based RAG techniques"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Metadata-Based RAG Techniques Evaluation', fontsize=16, fontweight='bold')
    
    # Prepare data
    techniques = list(all_results.keys())
    technique_labels = [t.replace('_', ' ').title() for t in techniques]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(techniques)))
    
    # 1. Performance Comparison
    ax1 = axes[0, 0]
    f1_scores = [np.mean([r['f1_score'] for r in all_results[t]]) for t in techniques]
    precisions = [np.mean([r['precision'] for r in all_results[t]]) for t in techniques]
    recalls = [np.mean([r['recall'] for r in all_results[t]]) for t in techniques]
    
    x = np.arange(len(techniques))
    width = 0.25
    
    ax1.bar(x - width, precisions, width, label='Precision', alpha=0.8, color='lightblue')
    ax1.bar(x, recalls, width, label='Recall', alpha=0.8, color='lightcoral')
    ax1.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Techniques')
    ax1.set_ylabel('Score')
    ax1.set_title('Metadata-Based Techniques Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(technique_labels, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Success Rate
    ax2 = axes[0, 1]
    success_rates = [sum(1 for r in all_results[t] if r['f1_score'] > 0) / len(all_results[t]) for t in techniques]
    bars = ax2.bar(techniques, success_rates, color=colors, alpha=0.8)
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate by Technique')
    ax2.set_xticklabels(technique_labels, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=8)
    
    # 3. Metadata Complexity vs Performance
    ax3 = axes[0, 2]
    if 'complexity_scores' in metadata_info:
        complexity_scores = metadata_info['complexity_scores']
        metadata_f1s = [np.mean([r['f1_score'] for r in all_results[t]]) for t in techniques if 'metadata' in t or 'llm' in t or 'adaptive' in t]
        
        ax3.scatter(complexity_scores[:len(metadata_f1s)], metadata_f1s, alpha=0.7, s=100)
        ax3.set_xlabel('Query Complexity Score')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('Complexity vs Performance')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Complexity data\nnot available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Query Complexity Analysis')
    
    # 4. Improvement over Baseline
    ax4 = axes[1, 0]
    baseline_f1 = np.mean([r['f1_score'] for r in all_results.get('baseline_chunks', all_results[techniques[0]])])
    improvements = []
    
    for technique in techniques:
        avg_f1 = np.mean([r['f1_score'] for r in all_results[technique]])
        if baseline_f1 > 0:
            improvement = ((avg_f1 - baseline_f1) / baseline_f1) * 100
        else:
            improvement = 0
        improvements.append(improvement)
    
    colors_imp = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]
    bars = ax4.bar(techniques, improvements, color=colors_imp, alpha=0.7)
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Performance vs Baseline')
    ax4.set_xticklabels(technique_labels, rotation=45, ha='right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 5. F1-Score Distribution
    ax5 = axes[1, 1]
    all_f1_data = []
    technique_names = []
    
    for technique, results in all_results.items():
        f1_scores_list = [r['f1_score'] for r in results]
        all_f1_data.extend(f1_scores_list)
        technique_names.extend([technique.replace('_', ' ').title()] * len(f1_scores_list))
    
    df = pd.DataFrame({'F1-Score': all_f1_data, 'Technique': technique_names})
    
    if len(df) > 0:
        unique_techniques = df['Technique'].unique()
        box_data = [df[df['Technique'] == tech]['F1-Score'].values for tech in unique_techniques]
        box = ax5.boxplot(box_data, patch_artist=True)
        
        ax5.set_ylabel('F1-Score Distribution')
        ax5.set_title('Score Distribution by Technique')
        ax5.set_xticklabels(unique_techniques, rotation=45, ha='right', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
    
    # 6. Technique Ranking
    ax6 = axes[1, 2]
    sorted_techniques = sorted(techniques, key=lambda x: np.mean([r['f1_score'] for r in all_results[x]]), reverse=True)
    sorted_f1s = [np.mean([r['f1_score'] for r in all_results[t]]) for t in sorted_techniques]
    sorted_labels = [t.replace('_', ' ').title() for t in sorted_techniques]
    
    bars = ax6.barh(range(len(sorted_techniques)), sorted_f1s, 
                    color=[colors[techniques.index(t)] for t in sorted_techniques], alpha=0.8)
    ax6.set_xlabel('Average F1-Score')
    ax6.set_title('Technique Ranking')
    ax6.set_yticks(range(len(sorted_techniques)))
    ax6.set_yticklabels(sorted_labels, fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, sorted_f1s)):
        width = bar.get_width()
        ax6.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('evaluation/evaluation4/generative_rag_results_SUMMARY.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting Metadata-Based RAG Evaluation...")
    
    # Initialize evaluator
    evaluator = MetadataRAGEvaluator()
    
    # Load evaluation data
    data = load_json('evaluation/evaluation3/eval3.json')
    
    # Define metadata-based techniques to test
    techniques = {
        "baseline_chunks": evaluator.baseline_search,
        "baseline_summary": evaluator.baseline_search,
        "metadata_guided": evaluator.metadata_guided_search,
        "llm_classification": evaluator.llm_query_classification_search,
        "semantic_filtering": evaluator.semantic_metadata_filtering,
        "adaptive_metadata": evaluator.adaptive_metadata_search,
    }
    
    # Store metadata information for analysis
    metadata_info = {
        'extraction_start_time': datetime.now(),
        'total_queries': len(data),
        'metadata_quality': {}
    }
    
    print(f"Evaluating {len(techniques)} techniques on {len(data)} queries...")
    
    # Evaluate each technique
    all_results = {}
    for technique_name, technique_func in techniques.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {technique_name}...")
        print(f"{'='*50}")
        
        start_time = time.time()
        results = evaluator.evaluate_technique(data, technique_name, technique_func)
        end_time = time.time()
        
        all_results[technique_name] = results
        
        # Calculate summary metrics
        avg_f1 = np.mean([r['f1_score'] for r in results])
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        success_rate = sum(1 for r in results if r['f1_score'] > 0) / len(results)
        
        print(f"\nResults for {technique_name}:")
        print(f"  F1-Score: {avg_f1:.4f}")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall: {avg_recall:.4f}")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Processing Time: {end_time - start_time:.1f}s")
    
    # Generate comprehensive report
    print(f"\n{'='*50}")
    print("Generating comprehensive analysis...")
    print(f"{'='*50}")
    
    metadata_info['extraction_end_time'] = datetime.now()

    # Create visualizations
    create_metadata_visualizations(all_results, metadata_info)
    
    # Save detailed results
    with open('evaluation/evaluation4/generative_detailed_results_SUMMARY.json', 'w') as f:
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
            } for technique, results in all_results.items()
        }, f, indent=2)
    
    print(f"\nMetadata-based RAG evaluation complete!")
    print(f"Files generated:")
    print(f"  - metadata_rag_results.png")
    print(f"  - metadata_rag_detailed_results.json")
    
    # Print final summary
    best_technique = max(all_results.keys(), key=lambda x: np.mean([r['f1_score'] for r in all_results[x]]))
    best_f1 = np.mean([r['f1_score'] for r in all_results[best_technique]])
    
    print(f"\nBest performing technique: {best_technique}")
    print(f"Best F1-score achieved: {best_f1:.4f}")