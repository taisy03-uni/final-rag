import json
import time
from pinecone import Pinecone

def upload_to_pinecone(key):
    path = "data/chunked/chunked_file.json"
    pc = Pinecone(api_key=key)
    index = pc.Index(host="")
    
    # Process in chunks
    chunk_size = 96  # Adjust based on your memory constraints
    records = []
    rate_limit_delay = 60 
    
    with open(path, 'r') as f:
        # Load the file incrementally
        data = json.load(f)
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
    api_key = "Blanked"
    upload_to_pinecone(api_key)