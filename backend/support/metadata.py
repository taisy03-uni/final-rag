import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from routers.chatgpt import get_model_response


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

class MetadataEnhancer:
        
    async def extract_metadata_with_llm(self, scenario: str) -> LegalMetadata:
        """Extract structured metadata using OpenAI"""
        
        prompt = f"""
        You are a legal expert tasked with extracting structured metadata from legal scenarios and questions.
        
        Scenario: {scenario}
        
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
            metadata_json = await get_model_response(prompt=prompt, reasoning="low")
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
                print("Error:", e)
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