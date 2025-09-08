from neo4j import GraphDatabase
import json
from dotenv import load_dotenv
import os

class Neo4jGraphBuilder:
    def __init__(self):
        self.uri = "neo4j+s://3b8fbcad.databases.neo4j.io"
        self.user = "neo4j"
        self.password = ""
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
    def create_constraints(self):
        with self.driver.session() as session:
            # Create uniqueness constraints
            session.run("CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
            session.run("CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE")
    
    def process_chunks(self, chunks):
        with self.driver.session() as session:
            # First pass - create case and organizations
            case_data = chunks[0]
            session.execute_write(self._create_case, case_data)
            
            # Process parties from chunk 2
            if len(chunks) > 1:
                session.execute_write(self._process_parties, chunks[1])
                
            # Process legal representatives from chunk 3
            if len(chunks) > 2:
                session.execute_write(self._process_legal_team, chunks[2])
    
    @staticmethod
    def _create_case(tx, chunk):
        case_id = chunk['_id'].split('#')[0]
        
        # First check if case exists
        result = tx.run("MATCH (c:Case {id: $case_id}) RETURN c", case_id=case_id)
        if result.single():
            print(f"Case {case_id} already exists - updating")
            tx.run("""
            MATCH (c:Case {id: $case_id})
            SET c.name = $case_name,
                c.date = $judgment_date,
                c.neutral_citation = $neutral_citation,
                c.case_number = $case_number,
                c.uri = $uri
            """, 
            case_id=case_id,
            case_name=chunk['name'],
            judgment_date=chunk['judgment_date'],
            neutral_citation=chunk['text'].split('\n')[3].strip('[]'),
            case_number="IPT/19/197/C",
            uri=chunk['uri'])
        else:
            print(f"Creating new case {case_id}")
            tx.run("""
            CREATE (c:Case {
                id: $case_id,
                name: $case_name,
                date: $judgment_date,
                neutral_citation: $neutral_citation,
                case_number: $case_number,
                uri: $uri
            })
            """, 
            case_id=case_id,
            case_name=chunk['name'],
            judgment_date=chunk['judgment_date'],
            neutral_citation=chunk['text'].split('\n')[3].strip('[]'),
            case_number="IPT/19/197/C",
            uri=chunk['uri'])
        
    @staticmethod
    def _process_parties(tx, chunk):
        case_id = chunk['_id'].split('#')[0]
        lines = [line.strip() for line in chunk['text'].split('\n') if line.strip()]
        
        # Identify complainant
        if "Complainant" in lines[1]:
            tx.run("""
            MATCH (c:Case {id: $case_id})
            MERGE (p:Person {name: $name})
            MERGE (p)-[r:IS_COMPLAINANT_IN]->(c)
            """, case_id=case_id, name=lines[1].split('- v -')[0].strip())
        
        # Process respondents
        for org in lines[3:]:
            if org.strip():
                tx.run("""
                MATCH (c:Case {id: $case_id})
                MERGE (o:Organization {name: $name})
                MERGE (o)-[r:IS_RESPONDENT_IN]->(c)
                """, case_id=case_id, name=org.strip())
    
    @staticmethod
    def _process_legal_team(tx, chunk):
        case_id = chunk['_id'].split('#')[0]
        text = chunk['text']
        
        # Process complainant's legal team
        if "for the Complainant" in text:
            team_part = text.split("for the Complainant")[0]
            lawyers = [l.strip() for l in team_part.split("and") if l.strip()]
            for lawyer in lawyers:
                tx.run("""
                MATCH (c:Case {id: $case_id})<-[:IS_COMPLAINANT_IN]-(p)
                MERGE (l:Person {name: $name})
                MERGE (l)-[r:REPRESENTS]->(p)
                SET r.role = 'Complainant Counsel'
                """, case_id=case_id, name=lawyer.split("(instructed by")[0].strip())
        
        # Process respondents' legal team
        if "for the Respondents" in text:
            team_part = text.split("for the Respondents")[0].split("for the Complainant")[-1]
            lawyers = [l.strip() for l in team_part.split("and") if l.strip()]
            for lawyer in lawyers:
                tx.run("""
                MATCH (c:Case {id: $case_id})<-[:IS_RESPONDENT_IN]-(o)
                MERGE (l:Person {name: $name})
                MERGE (l)-[r:REPRESENTS]->(o)
                SET r.role = 'Respondent Counsel'
                """, case_id=case_id, name=lawyer.split("(instructed by")[0].strip())

# Usage example
if __name__ == "__main__":
    # Load your chunks
    with open("data/chunked/chunked_file.json", "r") as f:
        chunks = json.load(f)
    
    # Initialize and build the graph
    builder = Neo4jGraphBuilder()
    try:
        builder.clear_database()
        builder.create_constraints()
        builder.process_chunks(chunks[:3])  # Process first 3 chunks
        print("Neo4j database populated successfully!")
    finally:
        builder.close()
        
