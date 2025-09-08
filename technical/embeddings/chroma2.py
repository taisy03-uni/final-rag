from transformers import AutoTokenizer, AutoModel
import chromadb
import numpy as np
import torch
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Load LegalBERT tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

def embed_text(text: str):
    """Get embedding vector from LegalBERT for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy().astype(np.float32)

# 2. Connect to Chroma
chroma_client = chromadb.PersistentClient(path="embeddings/chroma_store")
collection = chroma_client.get_or_create_collection(name="legal_embeddings")

# 3. Query Chroma with an example
def load_json(filepath="data/task1_test_labels_2025.json"):
    with open(filepath, "r") as f:
        data = json.load(f)

    examples = []
    for i, (query_doc, responses) in enumerate(data.items()):
        if i >= 20: 
            break
        # responses is already a list
        examples.append({"query": query_doc, "responses": responses})
    return examples


def get_text(query_document, base_dir="data/task2025_test"):
    filepath = os.path.join(base_dir, query_document)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()



precisions, recalls, f1s = [], [], []

examples = load_json()  
with open("data/task1_test_labels_2025.json", "r") as f:
    data = json.load(f)

all_examples = [{"query": q, "responses": r} for q, r in list(data.items())]

for ex in all_examples:
    query_doc = ex["query"]
    ground_truth = set(ex["responses"])
    # get query text
    try:
        query_text = get_text(query_doc)
    except FileNotFoundError:
        continue  # skip if missing file
    # embed and query
    query_embedding = embed_text(query_text)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=15
    )
    # retrieve file names from metadata
    retrieved_docs = {meta["file_name"] for meta in results["metadatas"][0]}
    # compute metrics
    tp = len(retrieved_docs & ground_truth)
    fp = len(retrieved_docs - ground_truth)
    fn = len(ground_truth - retrieved_docs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

print(precisions)
print(recall)
print(f1)

# 6. Report Mean Metrics
print(f"Mean Precision: {np.mean(precisions):.3f}")
print(f"Mean Recall:    {np.mean(recalls):.3f}")
print(f"Mean F1:        {np.mean(f1s):.3f}")

