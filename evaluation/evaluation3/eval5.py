# You 

import json
from pinecone import Pinecone
from openai import OpenAI

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def query_pinecone_chunks(query, rerank=True, reranker ="cohere-rerank-3.5"):
    """
    Query the Pinecone index with the given query string and return the top_k results.
    """
    pc = Pinecone(api_key="xxx")
    index = pc.Index(host="xxx")
    if rerank:
        #limit query to 1024 tokens
        if reranker == "pinecone-rerank-v0":
            query = query[:450] 
        else:
            query = query[:1024] #1024 for bge-reranker-v2-m3s
        results = index.search(
        namespace="chunks", 
        query={"inputs": {"text": query}, "top_k": 5}, 
        rerank={
            "model": reranker,
            "top_n": 5,
            "rank_fields": ["text"]
        },
        fields=["uri", "text", "name"])
    else:
        results = index.search(
            namespace="chunks",
            query= {"inputs": {"text": query}, "top_k": 5},
            fields=["uri", "text", "name"])
    return  results

def query_pinecone_summary(query, rerank=True, reranker ="cohere-rerank-3.5"):
    """
    Query the Pinecone index with the given query string and return the top_k results.
    """
    pc = Pinecone(api_key="")
    index = pc.Index(host="")
    if rerank:
        #limit query to 1024 tokens
        if reranker == "pinecone-rerank-v0":
            query = query[:450] 
        else:
            query = query[:1024] #1024 for bge-reranker-v2-m3s
        results = index.search(
        namespace="summary", 
        query={"inputs": {"text": query}, "top_k": 5}, 
        rerank={
            "model": reranker,
            "top_n": 5,
            "rank_fields": ["text"]
        },
        fields=["uri", "text", "name"])
    else:
        results = index.search(
            namespace="summary",
            query= {"inputs": {"text": query}, "top_k": 5},
            fields=["uri", "text", "name"])
    return  results

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_precision_recall(expected_uris, retrieved_uris):
    """
    Calculate precision and recall based on expected and retrieved URIs.
    """
    if not expected_uris:
        return 0.0, 0.0

    true_positive = len(set(expected_uris) & set(retrieved_uris))
    false_positive = len(set(retrieved_uris) - set(expected_uris))
    false_negative = len(set(expected_uris) - set(retrieved_uris))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

    return precision, recall

def calculate_total_score(expected_uris, retrieved_uris, scores):
    """
    Calculate the total score based on expected URIs, retrieved URIs, and their scores.
    """
    total_score = 0
    for uri in expected_uris:
        if uri in retrieved_uris:
            index = retrieved_uris.index(uri)
            total_score += scores[index] if index < len(scores) else 0
    return total_score
def evaluate_retrieval(query, expected_uri, reranker = None):
    """
    Evaluate the retrieval results against expected results.
    """
    if reranker == None:
        retrieved_results = query_pinecone_chunks(query, rerank = False)
    else:
        retrieved_results = query_pinecone_chunks(query, rerank = True, reranker = reranker)

    # check if the expected URI is in the retrieved results
    retrieved_uris = [hit['fields']["uri"] for hit in retrieved_results['result']['hits']]
    # calculate precision and recall
    precision, recall = calculate_precision_recall(expected_uri, retrieved_uris)
    f1_score = calculate_f1_score(precision, recall)
    total_score = calculate_total_score(expected_uri, retrieved_uris, [hit['_score'] for hit in retrieved_results['result']['hits']])
    print(f"Expected URI: {expected_uri}")
    print(f"Retrieved URIs: {retrieved_uris}")
    return retrieved_uris, precision, recall, f1_score, total_score

def evaluate_all(data, reranker):
    """
    Evaluate all queries in the provided data.
    """
    results = []
    for item in data:
        #Number 1 
        query = item['scenario'] + " " + item['question']
        # ALT 
        #MID
        #query = item['scenario'] + " " +  item['law_type'] +  " " +  item['end']

        # MID
        #query = item['scenario'] +  " " + item['law_type'] +  " " + item['end'] + " " + item['question']

        #query = item['scenario']

        print(f"Query: {query}")
        cases = item['relevant_cases']
        expected_uri = [case['uri'] for case in cases if case["uri"] != ""]  
        if reranker == "no rerank":
            rerank = False
            retrieved_uris, precision, recall, f1_score, total_score = evaluate_retrieval(query, expected_uri)
        else: 
            rerank = True
            retrieved_uris, precision, recall, f1_score, total_score= evaluate_retrieval(query, expected_uri, reranker)
        results.append({
            'query': query,
            'expected_uri': expected_uri,
            'retrieved_uri': retrieved_uris,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_score': total_score
        })

    return results

if __name__ == "__main__":
    # Load the data from the JSON file
    data = load_json('evaluation/evaluation3/eval3.json')
    rerankers = ["no rerank", "bge-reranker-v2-m3", "cohere-rerank-3.5", "pinecone-rerank-v0"]
    # Evaluate all queries
    for reranker in rerankers:
        print(f"Evaluating with reranker: {reranker}")
        evaluation_results = evaluate_all(data, reranker) 
        # Print the evaluation results
        for result in evaluation_results:
            print(f"Expected URI: {result['expected_uri']}")
            print(f"Retrieved URIs: {result['retrieved_uri']}")
            print(f"Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1 Score: {result['f1_score']:.4f}, Total Score: {result['total_score']:.4f}")

        # count aberage precision, recall, f1 score
        total_precision = sum(result['precision'] for result in evaluation_results if result['precision'] is not None)
        total_recall = sum(result['recall'] for result in evaluation_results if result['recall'] is not None)
        total_f1_score = sum(result['f1_score'] for result in evaluation_results if result['f1_score'] is not None)
        average_precision = total_precision / len(evaluation_results) if evaluation_results else 0
        average_recall = total_recall / len(evaluation_results) if evaluation_results else 0
        average_f1_score = total_f1_score / len(evaluation_results) if evaluation_results else 0
        print(f"Average Precision: {average_precision:.4f}, Average Recall: {average_recall:.4f}, Average F1 Score: {average_f1_score:.4f}")
        # count total correct and incorrect results
        total_correct = sum(1 for result in evaluation_results if result['precision'] > 0)
        total_incorrect = len(evaluation_results) - total_correct
        print(f"Total Correct: {total_correct}")
        print(f"Total Incorrect: {total_incorrect}")

        #total sum of scores
        total_score = sum(result['total_score'] for result in evaluation_results if result['total_score'] is not None)
        print(f"Total Score: {total_score}")
        # Save the evaluation results to a JSON file
        filename = f'code/evaluation/evaluation3/TEST{reranker}__n5.txt'
        with open(filename, 'w') as file:
            for result in evaluation_results:
                file.write(f"Expected URI: {result['expected_uri']}\n")
                file.write(f"Retrieved URIs: {result['retrieved_uri']}\n")
                file.write(f"Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1 Score: {result['f1_score']:.4f}, Total Score: {result['total_score']:.4f}\n")
                file.write("\n")
            file.write(f"Average Precision: {average_precision:.4f}, Average Recall: {average_recall:.4f}, Average F1 Score: {average_f1_score:.4f}\n")
            file.write(f"Total Correct: {total_correct}\n")
            file.write(f"Total Incorrect: {total_incorrect}\n")
            file.write(f"Total Score: {total_score}\n")
            print(f"Evaluation results saved to {filename}")
            # Save the evaluation results to a JSON file