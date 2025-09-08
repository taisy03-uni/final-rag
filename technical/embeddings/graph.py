import spacy
import networkx as nx
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import pandas as pd
from dataclasses import dataclass
from neo4j import GraphDatabase
import sqlite3

@dataclass
class LegalEntity:
    """Represents a legal entity (case, statute, judge, etc.)"""
    id: str
    type: str  # 'case', 'statute', 'judge', 'court', 'legal_concept'
    name: str
    metadata: Dict = None

@dataclass
class LegalRelationship:
    """Represents a relationship between legal entities"""
    source: str
    target: str
    relationship_type: str  # 'cites', 'overrules', 'follows', 'distinguishes', etc.
    context: str = None
    strength: float = 1.0

class LegalKnowledgeGraphBuilder:
    def __init__(self):
        # Load spaCy model (you might want to use a legal-specific one)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            
        self.entities = {}
        self.relationships = []
        self.graph = nx.DiGraph()
        
        # Legal citation patterns
        self.citation_patterns = {
            'uk_case': [
                r'\[(\d{4})\]\s*([A-Z]+)\s*(\d+)',  # [2020] UKSC 15
                r'(\d{4})\s+(\d+)\s+([A-Z]+)\s+(\d+)',  # 2020 1 WLR 123
                r'\((\d{4})\)\s*([A-Z]+)\s*(\d+)',  # (2020) EWCA Civ 123
            ],
            'statute': [
                r'([A-Z][a-zA-Z\s]+Act)\s+(\d{4})',  # Human Rights Act 1998
                r'([A-Z][a-zA-Z\s]+)\s+\(([A-Z][a-zA-Z\s]+)\)\s+Act\s+(\d{4})',
            ]
        }
        
        # Legal relationship indicators
        self.relationship_indicators = {
            'cites': ['referred to', 'cited', 'mentioned', 'see', 'following'],
            'overrules': ['overruled', 'overturned', 'reversed'],
            'follows': ['followed', 'applied', 'adopted'],
            'distinguishes': ['distinguished', 'different', 'unlike'],
            'considers': ['considered', 'examined', 'reviewed'],
            'affirms': ['affirmed', 'confirmed', 'upheld']
        }

    def extract_legal_entities(self, text: str, document_id: str = None) -> List[LegalEntity]:
        """Extract legal entities from text"""
        entities = []
        
        # Extract case citations
        for pattern in self.citation_patterns['uk_case']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation = match.group(0)
                entity = LegalEntity(
                    id=f"case_{citation.replace(' ', '_')}",
                    type='case',
                    name=citation,
                    metadata={'pattern': pattern, 'document_source': document_id}
                )
                entities.append(entity)
        
        # Extract statute references
        for pattern in self.citation_patterns['statute']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                statute = match.group(0)
                entity = LegalEntity(
                    id=f"statute_{statute.replace(' ', '_')}",
                    type='statute',
                    name=statute,
                    metadata={'document_source': document_id}
                )
                entities.append(entity)
        
        # Extract named entities (judges, courts, etc.)
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                # Simple heuristics to identify legal entities
                if any(word in ent.text.lower() for word in ['court', 'tribunal', 'commission']):
                    entity_type = 'court'
                elif any(title in ent.text.lower() for title in ['lord', 'lady', 'judge', 'justice']):
                    entity_type = 'judge'
                else:
                    entity_type = 'organization'
                
                entity = LegalEntity(
                    id=f"{entity_type}_{ent.text.replace(' ', '_')}",
                    type=entity_type,
                    name=ent.text,
                    metadata={'spacy_label': ent.label_, 'document_source': document_id}
                )
                entities.append(entity)
        
        return entities

    def extract_relationships(self, text: str, entities: List[LegalEntity], 
                            document_id: str = None) -> List[LegalRelationship]:
        """Extract relationships between legal entities"""
        relationships = []
        
        # Create entity mention map
        entity_mentions = {}
        for entity in entities:
            # Find all mentions of this entity in text
            mentions = []
            for match in re.finditer(re.escape(entity.name), text, re.IGNORECASE):
                mentions.append((match.start(), match.end()))
            entity_mentions[entity.id] = mentions
        
        # Find relationships based on proximity and linguistic patterns
        for source_entity in entities:
            for target_entity in entities:
                if source_entity.id == target_entity.id:
                    continue
                
                relationships.extend(
                    self._find_relationship_between_entities(
                        text, source_entity, target_entity, document_id
                    )
                )
        
        return relationships

    def _find_relationship_between_entities(self, text: str, source: LegalEntity, 
                                          target: LegalEntity, document_id: str) -> List[LegalRelationship]:
        """Find specific relationships between two entities"""
        relationships = []
        
        # Look for relationship indicators between entity mentions
        for rel_type, indicators in self.relationship_indicators.items():
            for indicator in indicators:
                # Find sentences containing both entities and the relationship indicator
                sentences = text.split('.')
                for sentence in sentences:
                    if (source.name.lower() in sentence.lower() and 
                        target.name.lower() in sentence.lower() and
                        indicator.lower() in sentence.lower()):
                        
                        relationship = LegalRelationship(
                            source=source.id,
                            target=target.id,
                            relationship_type=rel_type,
                            context=sentence.strip(),
                            strength=self._calculate_relationship_strength(sentence, indicator)
                        )
                        relationships.append(relationship)
        
        return relationships

    def _calculate_relationship_strength(self, context: str, indicator: str) -> float:
        """Calculate the strength of a relationship based on context"""
        # Simple heuristic - could be much more sophisticated
        base_strength = 1.0
        
        # Increase strength for explicit language
        if any(word in context.lower() for word in ['clearly', 'explicitly', 'directly']):
            base_strength += 0.3
        
        # Decrease strength for uncertain language
        if any(word in context.lower() for word in ['may', 'might', 'possibly', 'arguably']):
            base_strength -= 0.2
        
        return max(0.1, min(1.0, base_strength))

    def build_knowledge_graph(self, documents: List[Dict[str, str]]):
        """Build knowledge graph from a collection of legal documents"""
        print("Building legal knowledge graph...")
        
        all_entities = {}
        all_relationships = []
        
        for i, doc in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}: {doc.get('filename', 'Unknown')}")
            
            text = doc.get('full_text', '') or doc.get('text', '')
            doc_id = doc.get('filename') or doc.get('id') or f"doc_{i}"
            
            # Extract entities
            entities = self.extract_legal_entities(text, doc_id)
            
            # Merge entities (same entity might appear in multiple documents)
            for entity in entities:
                if entity.id in all_entities:
                    # Merge metadata
                    if all_entities[entity.id].metadata:
                        all_entities[entity.id].metadata.update(entity.metadata or {})
                else:
                    all_entities[entity.id] = entity
            
            # Extract relationships
            relationships = self.extract_relationships(text, entities, doc_id)
            all_relationships.extend(relationships)
        
        # Build NetworkX graph
        self.entities = all_entities
        self.relationships = all_relationships
        
        # Add nodes
        for entity_id, entity in all_entities.items():
            self.graph.add_node(entity_id, 
                              name=entity.name, 
                              type=entity.type,
                              metadata=entity.metadata)
        
        # Add edges
        for rel in all_relationships:
            if rel.source in self.graph and rel.target in self.graph:
                self.graph.add_edge(rel.source, rel.target,
                                  relationship=rel.relationship_type,
                                  context=rel.context,
                                  strength=rel.strength)
        
        print(f"Knowledge graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

    def query_knowledge_graph(self, query: str, top_k: int = 10) -> List[Dict]:
        """Query the knowledge graph"""
        results = []
        
        # Simple keyword matching - could be much more sophisticated
        query_lower = query.lower()
        
        # Find relevant nodes
        relevant_nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            if query_lower in node_data['name'].lower():
                relevant_nodes.append((node_id, node_data))
        
        # Get subgraph around relevant nodes
        for node_id, node_data in relevant_nodes:
            # Get neighbors
            neighbors = list(self.graph.neighbors(node_id))
            predecessors = list(self.graph.predecessors(node_id))
            
            result = {
                'entity': node_data,
                'entity_id': node_id,
                'outgoing_relationships': [],
                'incoming_relationships': []
            }
            
            # Get outgoing relationships
            for neighbor in neighbors:
                edge_data = self.graph.edges[node_id, neighbor]
                result['outgoing_relationships'].append({
                    'target': self.graph.nodes[neighbor]['name'],
                    'relationship': edge_data['relationship'],
                    'context': edge_data.get('context', ''),
                    'strength': edge_data.get('strength', 1.0)
                })
            
            # Get incoming relationships
            for predecessor in predecessors:
                edge_data = self.graph.edges[predecessor, node_id]
                result['incoming_relationships'].append({
                    'source': self.graph.nodes[predecessor]['name'],
                    'relationship': edge_data['relationship'],
                    'context': edge_data.get('context', ''),
                    'strength': edge_data.get('strength', 1.0)
                })
            
            results.append(result)
        
        return results[:top_k]

    def save_to_neo4j(self, uri: str, username: str, password: str):
        """Save knowledge graph to Neo4j database"""
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create nodes
            for entity_id, entity in self.entities.items():
                session.run(
                    "CREATE (n:LegalEntity {id: $id, name: $name, type: $type, metadata: $metadata})",
                    id=entity_id,
                    name=entity.name,
                    type=entity.type,
                    metadata=json.dumps(entity.metadata or {})
                )
            
            # Create relationships
            for rel in self.relationships:
                session.run(
                    """
                    MATCH (a:LegalEntity {id: $source})
                    MATCH (b:LegalEntity {id: $target})
                    CREATE (a)-[r:LEGAL_RELATIONSHIP {
                        type: $rel_type,
                        context: $context,
                        strength: $strength
                    }]->(b)
                    """,
                    source=rel.source,
                    target=rel.target,
                    rel_type=rel.relationship_type,
                    context=rel.context,
                    strength=rel.strength
                )
        
        driver.close()
        print("Knowledge graph saved to Neo4j")

    def save_to_json(self, filepath: str):
        """Save knowledge graph to JSON file"""
        graph_data = {
            'entities': {eid: {
                'name': e.name,
                'type': e.type,
                'metadata': e.metadata
            } for eid, e in self.entities.items()},
            'relationships': [{
                'source': r.source,
                'target': r.target,
                'relationship_type': r.relationship_type,
                'context': r.context,
                'strength': r.strength
            } for r in self.relationships]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"Knowledge graph saved to {filepath}")

    def get_graph_statistics(self) -> Dict:
        """Get statistics about the knowledge graph"""
        stats = {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'entity_types': {},
            'relationship_types': {},
            'most_connected_entities': []
        }
        
        # Count entity types
        for entity in self.entities.values():
            stats['entity_types'][entity.type] = stats['entity_types'].get(entity.type, 0) + 1
        
        # Count relationship types
        for rel in self.relationships:
            rel_type = rel.relationship_type
            stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
        
        # Find most connected entities
        node_degrees = dict(self.graph.degree())
        most_connected = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        stats['most_connected_entities'] = [
            {'entity_id': eid, 'name': self.entities[eid].name, 'connections': degree}
            for eid, degree in most_connected if eid in self.entities
        ]
        
        return stats

# Example usage
if __name__ == "__main__":
    # Sample legal documents
    sample_docs = [
        {
            'filename': 'case1.txt',
            'full_text': '''
            In R v Smith [2020] UKSC 15, the Supreme Court considered the application of the Human Rights Act 1998.
            The court followed the decision in Brown v Jones [2019] EWCA Civ 123, which distinguished the earlier
            case of Wilson v Davis [2018] EWHC 456. Lord Justice Williams delivered the judgment, noting that
            the European Court of Human Rights had previously ruled on similar matters.
            '''
        },
        {
            'filename': 'case2.txt',
            'full_text': '''
            The Court of Appeal in Brown v Jones [2019] EWCA Civ 123 overruled the High Court decision in 
            Wilson v Davis [2018] EWHC 456. Lady Justice Thompson noted that the Human Rights Act 1998 
            required a different interpretation. The case cited extensively from European jurisprudence.
            '''
        }
    ]
    
    # Build knowledge graph
    kg_builder = LegalKnowledgeGraphBuilder()
    kg_builder.build_knowledge_graph(sample_docs)
    
    # Query the graph
    results = kg_builder.query_knowledge_graph("Human Rights Act")
    print("\nQuery results for 'Human Rights Act':")
    for result in results:
        print(f"Entity: {result['entity']['name']}")
        print(f"Type: {result['entity']['type']}")
        if result['outgoing_relationships']:
            print("Outgoing relationships:")
            for rel in result['outgoing_relationships']:
                print(f"  -> {rel['target']} ({rel['relationship']})")
        print("---")
    
    # Get statistics
    stats = kg_builder.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total relationships: {stats['total_relationships']}")
    print(f"Entity types: {stats['entity_types']}")
    print(f"Relationship types: {stats['relationship_types']}")
    
    # Save to JSON
    kg_builder.save_to_json("legal_knowledge_graph.json")