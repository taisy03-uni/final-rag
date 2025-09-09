import json
import copy
from pinecone import Pinecone
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def query_pinecone_chunks(query, rerank=True, reranker ="cohere-rerank-3.5", top_k=5):
    """
    Query the Pinecone index with the given query string and return the top_k results.
    """
    pc = Pinecone(api_key="")
    index = pc.Index(host="")
    
    # First get results without reranking (get more for reranking)
    initial_k = top_k * 3 if rerank else top_k  # Get 3x more for reranking
    results = index.search(
        namespace="chunks",
        query= {"inputs": {"text": query}, "top_k": initial_k},
        fields=["uri", "text", "name"])
    
    # Truncate document text before reranking to avoid token limits
    if 'result' in results and 'hits' in results['result']:
        for hit in results['result']['hits']:
            if 'fields' in hit and 'text' in hit['fields']:
                # More aggressive truncation for pinecone-rerank-v0 (512 token limit total)
                if rerank and reranker == "pinecone-rerank-v0":
                    hit['fields']['text'] = hit['fields']['text'][:200]  # Conservative for 512 total limit
                else:
                    hit['fields']['text'] = hit['fields']['text'][:500]
    
    # Now apply reranking if requested
    if rerank:
        #limit query to avoid token limits
        if reranker == "pinecone-rerank-v0":
            query = query[:450] 
        else:
            query = query[:1024]  # For cohere-rerank-3.5
            
        # Rerank the truncated results
        rerank_results = index.search(
            namespace="chunks", 
            query={"inputs": {"text": query}, "top_k": initial_k}, 
            rerank={
                "model": reranker,
                "top_n": top_k,
                "rank_fields": ["text"]
            },
            fields=["uri", "text", "name"])
        
        # Apply same truncation to reranked results
        if 'result' in rerank_results and 'hits' in rerank_results['result']:
            for hit in rerank_results['result']['hits']:
                if 'fields' in hit and 'text' in hit['fields']:
                    if reranker == "pinecone-rerank-v0":
                        hit['fields']['text'] = hit['fields']['text'][:200]  # Conservative for 512 total limit
                    else:
                        hit['fields']['text'] = hit['fields']['text'][:500]
        
        return rerank_results
    
    # Return top_k from non-reranked results
    if 'result' in results and 'hits' in results['result']:
        results['result']['hits'] = results['result']['hits'][:top_k]
    
    return results

def query_pinecone_summary(query, rerank=True, reranker ="cohere-rerank-3.5", top_k=5):
    """
    Query the Pinecone index with the given query string and return the top_k results.
    """
    pc = Pinecone(api_key="")
    index = pc.Index(host="")
    
    # First get results without reranking (get more for reranking)
    initial_k = top_k * 3 if rerank else top_k  # Get 3x more for reranking
    results = index.search(
        namespace="summary",
        query= {"inputs": {"text": query}, "top_k": initial_k},
        fields=["uri", "text", "name"])
    
    # Truncate document text before reranking to avoid token limits
    if 'result' in results and 'hits' in results['result']:
        for hit in results['result']['hits']:
            if 'fields' in hit and 'text' in hit['fields']:
                # More aggressive truncation for pinecone-rerank-v0 (512 token limit total)
                if rerank and reranker == "pinecone-rerank-v0":
                    hit['fields']['text'] = hit['fields']['text'][:200]  # Conservative for 512 total limit
                else:
                    hit['fields']['text'] = hit['fields']['text'][:1000]
    
    # Now apply reranking if requested
    if rerank:
        #limit query to avoid token limits
        if reranker == "pinecone-rerank-v0":
            query = query[:450] 
        else:
            query = query[:1024]  # For cohere-rerank-3.5
            
        # Rerank the truncated results
        rerank_results = index.search(
            namespace="summary", 
            query={"inputs": {"text": query}, "top_k": initial_k}, 
            rerank={
                "model": reranker,
                "top_n": top_k,
                "rank_fields": ["text"]
            },
            fields=["uri", "text", "name"])
        
        # Apply same truncation to reranked results
        if 'result' in rerank_results and 'hits' in rerank_results['result']:
            for hit in rerank_results['result']['hits']:
                if 'fields' in hit and 'text' in hit['fields']:
                    if reranker == "pinecone-rerank-v0":
                        hit['fields']['text'] = hit['fields']['text'][:200]  # Conservative for 512 total limit
                    else:
                        hit['fields']['text'] = hit['fields']['text'][:300]
        
        return rerank_results
    
    # Return top_k from non-reranked results
    if 'result' in results and 'hits' in results['result']:
        results['result']['hits'] = results['result']['hits'][:top_k]
    
    return results

def query_hybrid_rrf(query, rerank=True, reranker="cohere-rerank-3.5"):
    """
    Query both chunks and summaries namespaces and combine using Reciprocal Rank Fusion @20, return top 10.
    """
    # Get results from both namespaces with top_k=20 each
    chunks_results = query_pinecone_chunks(query, rerank=rerank, reranker=reranker, top_k=5)
    summary_results = query_pinecone_summary(query, rerank=rerank, reranker=reranker, top_k=5)
    
    # Combine using RRF and return top 10
    rrf_results = reciprocal_rank_fusion([chunks_results, summary_results])
    
    return rrf_results

def reciprocal_rank_fusion(results_list, k=60):
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    Args:
        results_list: List of result lists, each containing dicts with 'fields' containing 'uri'
        k: RRF parameter (default 60)
    
    Returns:
        List of tuples (uri, rrf_score) sorted by RRF score descending
    """
    rrf_scores = defaultdict(float)
    doc_info = {}
    
    for results in results_list:
        if 'result' in results and 'hits' in results['result']:
            hits = results['result']['hits']
        else:
            hits = results
            
        for rank, hit in enumerate(hits, 1):
            uri = hit['fields']['uri']
            rrf_scores[uri] += 1 / (k + rank)
            if uri not in doc_info:
                doc_info[uri] = hit
    
    # Sort by RRF score and return top 10 (after retrieving @20)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return in the same format as original results
    fused_hits = []
    for uri, score in sorted_results:
        if uri in doc_info and doc_info[uri] is not None:
            original_hit = doc_info[uri]
            hit = {
                'fields': original_hit['fields'],
                '_score': score
            }
            fused_hits.append(hit)
    
    return {'result': {'hits': fused_hits}}

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
def evaluate_retrieval_rrf(query, expected_uri, reranker = None):
    """
    Evaluate the hybrid RRF retrieval results against expected results.
    """
    if reranker == None:
        retrieved_results = query_hybrid_rrf(query, rerank = False)
    else:
        retrieved_results = query_hybrid_rrf(query, rerank = True, reranker = reranker)

    # check if the expected URI is in the retrieved results
    retrieved_uris = [hit['fields']["uri"] for hit in retrieved_results['result']['hits']]
    #do top 10k
    # calculate precision and recall
    precision, recall = calculate_precision_recall(expected_uri, retrieved_uris)
    f1_score = calculate_f1_score(precision, recall)
    total_score = calculate_total_score(expected_uri, retrieved_uris, [hit['_score'] for hit in retrieved_results['result']['hits']])
    print(f"Expected URI: {expected_uri}")
    print(f"Retrieved URIs: {retrieved_uris}")
    return retrieved_uris, precision, recall, f1_score, total_score

def evaluate_retrieval(query, expected_uri, reranker = None):
    """
    Evaluate the retrieval results against expected results (original chunks-only version).
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

def evaluate_all_rrf(data, reranker):
    """
    Evaluate all queries in the provided data using RRF hybrid retrieval.
    """
    results = []
    for item in data:
        query = item['scenario'] + " " + item['question']
        print(f"Query: {query}")
        cases = item['relevant_cases']
        expected_uri = [case['uri'] for case in cases if case["uri"] != ""]  
        if reranker == "no rerank":
            retrieved_uris, precision, recall, f1_score, total_score = evaluate_retrieval_rrf(query, expected_uri)
        else: 
            retrieved_uris, precision, recall, f1_score, total_score = evaluate_retrieval_rrf(query, expected_uri, reranker)
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

def evaluate_all(data, reranker):
    """
    Evaluate all queries in the provided data (original chunks-only version).
    """
    results = []
    for item in data:
        query = item['scenario'] + " " + item['question']
        print(f"Query: {query}")
        cases = item['relevant_cases']
        expected_uri = [case['uri'] for case in cases if case["uri"] != ""]  
        if reranker == "no rerank":
            retrieved_uris, precision, recall, f1_score, total_score = evaluate_retrieval(query, expected_uri)
        else: 
            retrieved_uris, precision, recall, f1_score, total_score = evaluate_retrieval(query, expected_uri, reranker)
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
    rerankers = ["no rerank"]

    # Evaluate all queries using RRF hybrid retrieval
    for reranker in rerankers:
        print(f"Evaluating RRF@20 hybrid retrieval with reranker: {reranker}")
        evaluation_results = evaluate_all_rrf(data, reranker) 
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
        # Save the evaluation results to a file
        filename = f'evaluation/evaluation3/RRF20_HYBRID_{reranker}_@5.txt'
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