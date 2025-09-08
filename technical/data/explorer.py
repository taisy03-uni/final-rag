# File used to understand key information about the dataset, chunks and embeddings
from data.data_retrieval import DataDownload
import json
from collections import defaultdict
import numpy as np
import numpy as np
import xml.etree.ElementTree as ET

def get_total_documets(file_paths):
    # Step 1: Output number of files
    print("Number of documents:")
    print(len(file_paths), end='\n')
    print("==================================================",  end='\n')

def get_data_by_year(data):
    print("Data by Year:")
    for year in range(2000,2026):
        paths = data.get_file_paths(year = str(year))
        print(f'Year {year}: {len(paths)} documents')
    print("\n")
    print("==================================================",  end='\n')

def get_data_by_court(data):
    tags_court = data.tags_court
    print("Number of documents by Court", end='\n')
    for tag in tags_court:
        for key, value in tag.items():
            if isinstance(value, list):  # This means it's a court/tribunal with subdivisions
                print(f"\n--- {key.upper()} (Sub-divisions) ---")
                for sub_entry in value:
                    for sub_key, sub_value in sub_entry.items():
                        print(f"\n  {sub_value} ({sub_key}):")
                        paths = data.get_file_paths(court=sub_key)
                        count = len(paths)
                        print(f"Count: {count} files")
            else:  # This is a direct court/tribunal
                print(f"\n--- {value} ({key}) ---")
                paths = data.get_file_paths(court=key)
                count = len(paths)
                print(f"Count: {count} files")

def get_data_by_tribunals(data):
    tags_tribunal = data.tags_tribunals
    print("Number of documents by Court/Tribunals", end='\n')
    for tag in tags_tribunal:
        for key, value in tag.items():
            if isinstance(value, list):  # This means it's a court/tribunal with subdivisions
                print(f"\n--- {key.upper()} (Sub-divisions) ---")
                for sub_entry in value:
                    for sub_key, sub_value in sub_entry.items():
                        print(f"\n  {sub_value} ({sub_key}):")
                        paths = data.get_file_paths(court=sub_key)
                        count = len(paths)
                        print(f"Count: {count} files")
            else:  # This is a direct court/tribunal
                print(f"\n--- {value} ({key}) ---")
                paths = data.get_file_paths(court=key)
                count = len(paths)
                print(f"Count: {count} files")

def get_number_of_chunks():
    try:
        with open(chunks_path, 'r') as f:
            chunk_data = json.load(f)
        print(f"\nNumber of chunks: {len(chunk_data)}")
    except FileNotFoundError:
        print(f"\nChunk file not found at {chunks_path}")
    except json.JSONDecodeError:
        print(f"\nError reading chunk file at {chunks_path}")


def get_avg_document_length(file_paths):
    lengths = []
    
    for path in file_paths:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            text = " ".join(root.itertext())  # extract all text content
            word_count = len(text.split())
            lengths.append(word_count)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    
    if not lengths:
        return None
    
    lengths = np.array(lengths)
    stats = {
        "min": int(np.min(lengths)),
        "mean": float(np.mean(lengths)),
        "max": int(np.max(lengths)),
        "75_percentile": float(np.percentile(lengths, 75)),
        "25_percentile": float(np.percentile(lengths, 25)),
        "5_percentile": float(np.percentile(lengths, 5)),
        "95_percentile": float(np.percentile(lengths, 95))
    }
    
    return stats

# CD into lawai 
#RUN AS MODULES 
# python3 -m data.explorer 

print("data download")
data = DataDownload()
print("getting paths")
file_paths = data.get_file_paths()
print("got paths")
print(file_paths)
print(get_avg_document_length(file_paths))
"""chunks_path = "data/chunked/chunked_file.json"
get_data_by_tribunals(data)
get_number_of_chunks()"""
