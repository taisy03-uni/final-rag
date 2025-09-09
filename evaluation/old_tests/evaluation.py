import json
from pinecone import Pinecone
from openai import OpenAI

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

import openai

def generate_noun_phrase_collocations(query, api_key, model="gpt-4.o-mini", temperature=0.1):
    """
    Generate noun phrase collocations (variations) from a query using OpenAI's API.
    Returns:
        list: A list of noun phrase variations/collocations.
    """
    # Set the API key
    client = OpenAI(api_key=api_key)
    
    # Create a prompt to generate noun phrase variations
    prompt = f"""
    Generate a list of noun phrase variations or collocations related to the following query.
    These should be meaningful alternatives that could help in search or retrieval tasks.
    Focus on maintaining the core meaning while offering different phrasings.
    
    Query: "{query}"
    
    Return the variations as a comma-separated list without numbers or bullet points.
    """
    
    try:
        response = client.responses.create(
            model= model,
            input=f"You are a helpful assistant that generates useful noun phrase variations for search queries. Prompt: {prompt}"
        )
        print(response.output_text)
        return response.output_text
    
    except Exception as e:
        print(f"Error generating noun phrase collocations: {e}")
        return []
    

def query_pinecone(query, rerank=True, reranker ="cohere-rerank-3.5"):
    """
    Query the Pinecone index with the given query string and return the top_k results.
    """
    pc = Pinecone(api_key="XXX")
    index = pc.Index(host="")
    if rerank:
        #limit query to 1024 tokens
        if reranker == "pinecone-rerank-v0":
            query = query[:450] 
        else:
            query = query[:1024] #1024 for bge-reranker-v2-m3s
        results = index.search(
        namespace="chunks", 
        query={"inputs": {"text": query}, "top_k": 10}, 
        rerank={
            "model": reranker,
            "top_n": 10,
            "rank_fields": ["text"]
        },
        fields=["file_path", "text", "name"])
    else:
        results = index.search(
            namespace="chunks",
            query= {"inputs": {"text": query}, "top_k": 10},
            fields=["file_path", "text", "name"])
    return  results

def evaluate_retrieval(query, expected_uri, reranker = None):
    """
    Evaluate the retrieval results against expected results.
    """
    if reranker == None:
        retrieved_results = query_pinecone(query, rerank = False)
    else:
        retrieved_results = query_pinecone(query, rerank = True, reranker = reranker)
    # Extracting "uri" and "name" from the retrieved results

    # check if the expected URI is in the retrieved results
    retrieved_uris = [hit['fields']["file_path"] for hit in retrieved_results['result']['hits']]
    scores = [hit['_score'] for hit in retrieved_results['result']['hits']]
    is_correct = expected_uri in retrieved_uris
    #get score if is_correct else None
    score = scores[retrieved_uris.index(expected_uri)] if is_correct else None

    print(f"Expected URI: {expected_uri}")
    print(f"Retrieved URIs: {retrieved_uris}")
    return is_correct, score, retrieved_uris

def evaluate_all(data, reranker):
    """
    Evaluate all queries in the provided data.
    """
    results = []
    for item in data:
        query = item['query']

        print(f"Query: {query}")
        expected_uri = item['original_file_path']
        if reranker == "no rerank":
            rerank = False
            is_correct, score, retrieved_uri = evaluate_retrieval(query, expected_uri)
        else: 
            rerank = True
            is_correct, score, retrieved_uri = evaluate_retrieval(query, expected_uri, reranker)
        results.append({
            'query': query,
            'expected_uri': expected_uri,
            'retrieved_uri': retrieved_uri,
            'is_correct': is_correct,
            'score': score
        })

    return results

if __name__ == "__main__":
    # Load the data from the JSON file
    data = load_json('evaluation/evaluation1/processed_evaluation_dataset2_with_queries.json')
    rerankers = ["no rerank", "bge-reranker-v2-m3", "cohere-rerank-3.5", "pinecone-rerank-v0"]
    # Evaluate all queries
    for reranker in rerankers:
        print(f"Evaluating with reranker: {reranker}")
        evaluation_results = evaluate_all(data, reranker) 
        # Print the evaluation results
        for result in evaluation_results:
            print(f"Expected URI: {result['expected_uri']}")
            print(f"Is Correct: {result['is_correct']}")
            print(f"Score: {result['score']}\n")

        # count total correct and incorrect results
        total_correct = sum(1 for result in evaluation_results if result['is_correct'])
        total_incorrect = len(evaluation_results) - total_correct
        print(f"Total Correct: {total_correct}")
        print(f"Total Incorrect: {total_incorrect}")
        #total sum of scores
        total_score = sum(result['score'] for result in evaluation_results if result['score'] is not None)
        print(f"Total Score: {total_score}")
        # Save the evaluation results to a JSON file
        filename = f'evaluation/evaluation1/PHrasecollisions_with_{reranker}_n_10.txt'
        with open(filename, 'w') as file:
            for result in evaluation_results:
                file.write(f"Expected URI: {result['expected_uri']}\n")
                file.write(f"Retrieved URIs: {result['retrieved_uri']}\n")
                file.write(f"Is Correct: {result['is_correct']}\n")
                file.write(f"Score: {result['score']}\n\n")
            # add total correct and incorrect to the file
            file.write(f"\nTotal Correct: {total_correct}\n")
            file.write(f"Total Incorrect: {total_incorrect}\n")
            file.write(f"Total Score: {total_score}\n")