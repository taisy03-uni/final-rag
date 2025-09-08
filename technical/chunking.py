from data.data_retrieval import DataDownload
from bs4 import BeautifulSoup
import re
import random  
from pinecone import Pinecone
import os
import json

class Chunker():
    def __init__(self, method):
        self.data = DataDownload()
        self.methods = ["fixed_size", "overlap"]
        self.file_paths = self.data.get_file_paths()
        self.chunks = []


    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        #remove any line that begins with #judgment
        text = re.sub(r'(?m)^#judgment.*\n?', '', text)
        return text
    
    def get_metadata(self, soup, path):
        frbr_expression = soup.find('FRBRWork')
        metadata = {}
        metadata["path"] = path  # Store the file path in metadata
        if frbr_expression:
            # Extract values from child tags
            frbr_uri = frbr_expression.find('FRBRuri')
            if frbr_uri and 'value' in frbr_uri.attrs:
                metadata['uri'] = frbr_uri['value']
            
            frbr_date = frbr_expression.find('FRBRdate')
            if frbr_date and 'date' in frbr_date.attrs:
                metadata['judgment_date'] = frbr_date['date']

            frbr_name = frbr_expression.find('FRBRname')
            if frbr_name and 'value' in frbr_name.attrs:
                metadata['name'] = frbr_name['value']

            frbr_author = frbr_expression.find('FRBRauthor')
            if frbr_author and 'href' in frbr_author.attrs:
                metadata['author'] = frbr_author['href']
            
        return metadata
    
    def chunking(self, text):
        if len(text) < 1000:
            return [text]
        
        # First split by the separator pattern
        chunks = []
        parts = text.split('- - - - - - -')
        for part in parts:
            if part.strip():  # Skip empty/whitespace-only parts
                chunks.append(part.strip())
        new_chunks = []

        for chunk in chunks:
            if len(chunk) < 1000:
                new_chunks.append(chunk)
                continue
            # Find all matches of numbered items (e.g., "\n3. " or "\n3.\n")
            number_pattern = r'\n(\d+)\.\s*\n?'
            matches = list(re.finditer(number_pattern, chunk))
            
            # Filter valid sequential numbers (1, 2, 3, ...)
            valid_starts = []
            expected_num = 1
            for match in matches:
                current_num = int(match.group(1))
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
            if len(chunk) <= 1000:
                final_chunks.append(chunk)
            else:
                # Split by sentences first
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current_part = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    
                    # If adding this sentence would exceed limit
                    if current_length + sentence_length > 1000:
                        if current_part:  # Add what we have so far
                            final_chunks.append(' '.join(current_part))
                            current_part = []
                            current_length = 0
                        
                        # If single sentence is too long, split it at spaces
                        if sentence_length > 1000:
                            words = sentence.split()
                            current_words = []
                            words_length = 0
                            
                            for word in words:
                                word_len = len(word) + (1 if current_words else 0)
                                if words_length + word_len > 1000:
                                    if current_words:
                                        final_chunks.append(' '.join(current_words))
                                        current_words = []
                                        words_length = 0
                                    # Handle extremely long words
                                    if len(word) > 1000:
                                        for i in range(0, len(word), 1000):
                                            final_chunks.append(word[i:i+1000])
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


    def get_file_text(self, path):
        """Open and parse XML file, extracting both text content and metadata."""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'xml')  # Use 'xml' parser for XML files
                # Extract metadata
                metadata = self.get_metadata(soup,path)
                # Get the main text content (excluding metadata)
                text = self.clean_text(soup.get_text())

                #chunk text
                chunks = self.chunking(text)
                return metadata, chunks
                
                
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return None

    
    def get_largest_tokensize(self) -> int:
        filepaths = self.data.get_file_paths()
        max = 0 
        for path in filepaths:
            text = self.get_file_text(path)
            if len(text)/4 > max:
                print(f"Processing file: {path}")
                max = len(text)/4
                print(f"Current max token size: {max}")
        return max

    def output_random_file(self):
        files = self.file_paths
        #choose rnadom file from file_paths  list
        random_file = random.choice(files)
        print(f"Randomly selected file: {random_file}")
        metadata, data = chunker.get_file_text(random_file)
        # output metadata and chunks to output.txt
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(f"Metadata: {metadata}\n")
            f.write("Chunks:\n")
            for chunk in data:
                print('Chunk length:', len(chunk))  
                f.write(chunk + "\n\n\n\n\n\n\n\n\n\n")
        
    def chunk_files(self):
        files = chunker.file_paths
        chunks = []
        n = len(files)
        for i, file in enumerate(files): 
            id1 = "file" + str(i+1)
            metadata, data = chunker.get_file_text(file)
            print(f"File {i}/{n}: {metadata}")
            if not data:
                print(f"Skipping empty file: {file}")
                continue
            if not metadata.get("uri"):
                print(f"Warning: Empty metadata URI - skipping file: {file}")
                continue
            for j,chunk in enumerate(data):
                record = {
                "_id": str(metadata["uri"]) + "#" + str(j+1),
                "text": chunk,
                "file_path": metadata["path"],
                "judgment_date": metadata.get("judgment_date", ""),
                "name": metadata.get("name", ""),
                "author": metadata.get("author", ""),
                "uri": metadata.get("uri", ""),
            }
                chunks.append(record)
        output_filename = f"data/chunked/chunked_file.json"
        with open(output_filename, 'w') as f:
            json.dump(chunks, f, indent=2)
        print(f"Saved chunked files to {output_filename}")
        return output_filename

    def upload_to_pinecone(self,api_key):
        files = chunker.file_paths
        pc = Pinecone(api_key="PINECONE_API")
        index = pc.Index(host="https://ragproject-kjuem0t.svc.aped-4627-b74a.pinecone.io")
        for i, file in enumerate(files): 
            id1 = "file" + str(i+1)
            metadata, data = chunker.get_file_text(file)
            print(metadata)
            if not data:
                print(f"Skipping empty file: {file}")
                continue
            for j,chunk in enumerate(data):
                index.upsert_records(
                    "chunks",
                    [
                    {
                    "_id": str(metadata["uri"]) + "#" + str(j+1),
                    "text": chunk,
                    "file_path": metadata["path"],
                    "judgment_date": metadata.get("judgment_date", ""),
                    "name": metadata.get("name", ""),
                    "author": metadata.get("author", ""),
                    "uri": metadata.get("uri", ""),
                    }]
                )
                print(f"Uploaded chunk {j+1} of file {id1} to Pinecone.")
    
    def clean_and_metadata_text(self):
        """
        Process all files: clean text, extract metadata, and save full case text (no chunking).
        Saves output to data/cleaned_data.json.
        """
        cleaned_cases = []

        for i, path in enumerate(self.file_paths):
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'xml')
                    metadata = self.get_metadata(soup, path)
                    text = self.clean_text(soup.get_text())

                    if not metadata.get("uri"):
                        print(f"Skipping file with empty URI: {path}")
                        continue

                    record = {
                        "_id": metadata.get("uri", f"file{i+1}"),
                        "text": text,
                        "file_path": metadata.get("path", path),
                        "judgment_date": metadata.get("judgment_date", ""),
                        "name": metadata.get("name", ""),
                        "author": metadata.get("author", ""),
                        "uri": metadata.get("uri", "")
                    }

                    cleaned_cases.append(record)
                    print(f"Processed file {i+1}/{len(self.file_paths)}: {path}")

            except Exception as e:
                print(f"Error processing file {path}: {e}")
                continue

        output_file = "data/cleaned_data.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_cases, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(cleaned_cases)} cleaned cases to {output_file}")
        return output_file

if __name__ == "__main__":
    chunker = Chunker("fixed_size")
    #chunker.output_random_file()
    files = chunker.file_paths
    print(f"Total files to process: {len(files)}")
    file = chunker.clean_and_metadata_text()

    """with open("output.txt", "w", encoding="utf-8") as out_file:
        for file in files:
            metadata, chunks = chunker.get_file_text(file)
            
            # Write file header
            out_file.write(f"\n\n=== File: {file} ===\n")
            out_file.write(f"=== Metadata: {metadata} ===\n\n")
            
            # Write each chunk with separation
            for i, chunk in enumerate(chunks, 1):
                print(f"{i} lenght:")
                print(len(chunk))
                out_file.write(f"--- Chunk {i} ---\n")
                out_file.write(chunk)
                out_file.write("\n\n" + "="*50 + "\n\n")  # Visual separator
            
            out_file.write("\n\n" + "*"*80 + "\n\n")  # File separator
    
    print("Chunks have been written to output.txt with separators")
    """