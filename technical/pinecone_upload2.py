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


def save_chunks(df_files, chunk_size=1000, output_path="data/chunked/chunked_file2.json"):
    """
    Process all files, chunk them, and save the chunks to a JSON file.

    Args:
        df_files (pd.DataFrame): DataFrame with columns 'file_name' and 'text'.
        chunk_size (int): Maximum size of each text chunk.
        output_path (str): Path to save the chunked JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_records = []

    for idx, row in df_files.iterrows():
        file_name = row["file_name"]
        text = row["text"]

        print(f"\nProcessing file {idx+1}/{len(df_files)}: {file_name}")

        # Chunk text
        chunks = chunking(text, size=chunk_size)

        # Create records for each chunk
        for chunk_number, chunk_text in enumerate(chunks, start=1):
            record = {
                "_id": str(uuid.uuid4()),
                "text": chunk_text,
                "file_path": file_name,
                "chunk_number": chunk_number
            }
            all_records.append(record)

    # Save all chunks to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(all_records)} chunks to '{output_path}'")

def upload_to_pinecone(key):
    path = "data/chunked/chunked_file2.json"
    pc = Pinecone(api_key=key)
    index = pc.Index(host="https://experiment2-cpullh2.svc.apu-57e2-42f6.pinecone.io")
    
    # Process in chunks
    chunk_size = 96  # Adjust based on your memory constraints
    records = []
    rate_limit_delay = 60 
    data = []
    with open(path, 'r') as f:
        # Load the file incrementally
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
        print(len(data))
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            try:
                index.upsert_records("chunks", chunk)
                time.sleep(1)
                print(f"Uploaded records {i} to {i+len(chunk)}")
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    print(f"Rate limit hit, waiting {rate_limit_delay} seconds...")
                    time.sleep(rate_limit_delay)
                    # Retry the same chunk
                    i -= chunk_size
                else:
                    print(e)
    
if __name__ == "__main__":

    #get_data_from_filepath()

    #df_files = load_json()
    #save_chunks(df_files)
    upload_to_pinecone("")




