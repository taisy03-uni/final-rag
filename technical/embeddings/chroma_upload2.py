import json
import time
from pinecone import Pinecone
from google.cloud import storage
import pandas as pd
from tqdm import tqdm  # optional, for a nice progress bar
import os
import re
import uuid
import torch
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
import onnxruntime as ort

def get_data_from_filepath(path="data/task2025_train", output_file="data/task2025.json"):
    data = []
    
    # Loop through all files in the folder
    i = 1
    for filename in os.listdir(path):
        print(i)
        i +=1
        if filename.endswith(".txt"):
            filepath = os.path.join(path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            data.append({"file_name": filename, "text": text})
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save as JSON
    df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(df)} files to {output_file}")

def load_json(filepath="data/task2025.json"):
    """
    Load a JSONL file into a pandas DataFrame.
    """
    df = pd.read_json(filepath, orient="records", lines=True)
    print(f"Loaded {len(df)} records from {filepath}")
    return df

def chunking(text, size = 1000):
        if len(text) < size:
            return [text]
        
        # First split by the separator pattern
        chunks = []
        parts = text.split('- - - - - - -')
        for part in parts:
            if part.strip():  # Skip empty/whitespace-only parts
                chunks.append(part.strip())
        new_chunks = []

        for chunk in chunks:
            if len(chunk) < size:
                new_chunks.append(chunk)
                continue
            # Find all matches of numbered items (e.g., "\n3. " or "\n3.\n")
            number_pattern = r'\n(\d+)\.\s*|\[(\d+)\]'
            matches = list(re.finditer(number_pattern, chunk))
            
            # Filter valid sequential numbers (1, 2, 3, ...)
            valid_starts = []
            expected_num = 1
            for match in matches:
                current_num = int(match.group(1) or match.group(2))
                if current_num == expected_num:
                    valid_starts.append(match.start())
                    expected_num += 1

            # Extract chunks between valid numbers
            prev_end = 0
            for start in valid_starts:
                chunk1 = chunk[prev_end:start].strip()
                if chunk1:
                    new_chunks.append(chunk1)
                prev_end = start
            # Add the remaining text after the last number
            if prev_end < len(chunk):
                new_chunks.append(chunk[prev_end:].strip())
        
        #if any chunk is still larger than 1000 characters, split it further
        final_chunks = []
        for chunk in new_chunks:
            if len(chunk) <= size:
                final_chunks.append(chunk)
            else:
                # Split by sentences first
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current_part = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    
                    # If adding this sentence would exceed limit
                    if current_length + sentence_length > size:
                        if current_part:  # Add what we have so far
                            final_chunks.append(' '.join(current_part))
                            current_part = []
                            current_length = 0
                        
                        # If single sentence is too long, split it at spaces
                        if sentence_length > size:
                            words = sentence.split()
                            current_words = []
                            words_length = 0
                            
                            for word in words:
                                word_len = len(word) + (1 if current_words else 0)
                                if words_length + word_len > size:
                                    if current_words:
                                        final_chunks.append(' '.join(current_words))
                                        current_words = []
                                        words_length = 0
                                    # Handle extremely long words
                                    if len(word) > size:
                                        for i in range(0, len(word), size):
                                            final_chunks.append(word[i:i+size])
                                    else:
                                        current_words.append(word)
                                        words_length = len(word)
                                else:
                                    current_words.append(word)
                                    words_length += word_len
                            
                            if current_words:
                                final_chunks.append(' '.join(current_words))
                            continue
                    
                    current_part.append(sentence)
                    current_length += sentence_length
                
                # Add any remaining sentences
                if current_part:
                    final_chunks.append(' '.join(current_part))              
        return final_chunks 


def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts using LegalBERT.
    
    Args:
        texts (list of str): Texts to encode
        device (str): 'cpu' or 'cuda'
    
    Returns:
        torch.Tensor: Embeddings tensor (num_texts x hidden_size)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    embeddings = []
    length = len(texts)
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text, 
                padding="max_length", 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**encoded)
            # Use the [CLS] token representation as embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
            embeddings.append(cls_embedding.cpu())

    return torch.vstack(embeddings)

def process_and_save_to_chroma(df_files, chunk_size=1000, batch_size=16):
    """
    Process all files, chunk them, generate embeddings, 
    and store results directly in ChromaDB.
    """
    for idx, row in df_files.iterrows():
        file_name = row["file_name"]
        text = row["text"]

        print(f"\nProcessing file {idx+1}/{len(df_files)}: {file_name}")

        # Chunk text
        chunks = chunking(text, size=chunk_size)

        # Build chunk objects
        all_chunks = []
        for chunk_id, chunk_text in enumerate(chunks, start=1):
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "file_name": file_name,
                "chunk_id": chunk_id,
            })

        # Generate embeddings in batches
        texts = [c["text"] for c in all_chunks]
        embeddings_list = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            batch_embeddings = generate_embeddings(batch_texts)  # your GPU fn
            embeddings_list.extend(batch_embeddings.cpu().tolist())

        # Insert into ChromaDB
        collection.add(
            ids=[c["id"] for c in all_chunks],
            embeddings=embeddings_list,
            documents=[c["text"] for c in all_chunks],
            metadatas=[{
                "file_name": c["file_name"],
                "chunk_id": c["chunk_id"]
            } for c in all_chunks]
        )

        print(f"✅ Stored {len(all_chunks)} chunks from {file_name} into ChromaDB.")
def open_chroma_collection(path="embeddings/chroma_store", collection_name="legal_embeddings_768"):
    """
    Open an existing ChromaDB collection or create a new one if it doesn't exist.
    
    Args:
        path (str): Path to the Chroma persistent store.
        collection_name (str): Name of the collection to open/create.
    
    Returns:
        chromadb.api.models.Collection: The Chroma collection object.
    """
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=collection_name)
    print(f"✅ Chroma collection '{collection_name}' opened at '{path}'")
    return collection, client

def resume_and_save_to_chroma(df_files, start_file_name="028840.txt", chunk_size=1000, batch_size=16):
    """
    Resume processing after a given file name.
    
    Args:
        df_files (pd.DataFrame): DataFrame with file_name and text
        start_file_name (str): The last successfully processed file
        chunk_size (int): Chunk size for splitting text
        batch_size (int): Batch size for embedding generation
    """
    # Find index of the file we last processed
    try:
        start_idx = df_files.index[df_files["file_name"] == start_file_name][0] + 1
        print(f"start_idx: {start_idx}")
    except IndexError:
        print(f"❌ File {start_file_name} not found in dataframe. Starting from beginning.")
        return None

    print(f"➡️ Resuming from index {start_idx} ({len(df_files) - start_idx} files left).")

    # Process the rest
    for idx in range(start_idx, len(df_files)):
        row = df_files.iloc[idx]
        file_name = row["file_name"]
        text = row["text"]

        print(f"\nProcessing file {idx+1}/{len(df_files)}: {file_name}")

        # Chunk text
        chunks = chunking(text, size=chunk_size)

        # Build chunk objects
        all_chunks = []
        for chunk_id, chunk_text in enumerate(chunks, start=1):
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "file_name": file_name,
                "chunk_id": chunk_id,
            })
        
        # Generate embeddings in batches
        texts = [c["text"] for c in all_chunks]
        embeddings_list = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            batch_embeddings = generate_embeddings(batch_texts)
            embeddings_list.extend(batch_embeddings.cpu().tolist())

        # Insert into ChromaDB
        collection.add(
            ids=[c["id"] for c in all_chunks],
            embeddings=embeddings_list,
            documents=[c["text"] for c in all_chunks],
            metadatas=[{
                "file_name": c["file_name"],
                "chunk_id": c["chunk_id"]
            } for c in all_chunks]
        )

        print(f"✅ Stored {len(all_chunks)} chunks from {file_name} into ChromaDB.")

    
if __name__ == "__main__":

    # Load LegalBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    
    #load chroma
    chroma_client = chromadb.PersistentClient(path="embeddings/chroma_store")
    collection = chroma_client.get_or_create_collection(name="legal_embeddings")

    print(f"✅ Connected to Chroma collection '{collection}' at '{chroma_client}'")

    total_embeddings = collection.count()
    print(f"Total embeddings in collection: {total_embeddings}")    

    #get_data_from_filepath()
       
    df_files = load_json()
    resume_and_save_to_chroma(df_files)
    total_embeddings = collection.count()
    print(f"Total embeddings in collection: {total_embeddings}") 
    #process_and_save_to_chroma(df_files)



