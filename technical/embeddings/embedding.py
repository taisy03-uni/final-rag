import json
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer

class Embedding:
    """
    A class to handle text embeddings using different models.
    Supported models: 'sentence-transformers' and 'transformers' models.
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', device: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the model to use. Default is 'all-mpnet-base-v2'.
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize the appropriate model
        if "sentence-transformers" in model_name.lower():
            self.model = SentenceTransformer(model_name, device=self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        if hasattr(self, 'tokenizer'):  # Transformer model
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        else:  # Sentence Transformer model
            embedding = self.model.encode(text, convert_to_numpy=True)
        
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if hasattr(self, 'tokenizer'):  # Transformer model
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        else:  # Sentence Transformer model
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        return embeddings.tolist()

def add_embeddings_to_data(input_path: str, output_path: str, model_name: str = 'all-mpnet-base-v2'):
    """
    Add embeddings to JSON data and save to a new file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to save output JSON file with embeddings
        model_name: Name of the embedding model to use
    """
    # Load the data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Initialize embedding model
    embedder = Embeddings(model_name=model_name)
    
    # Extract texts for batch processing
    texts = [item['text'] for item in data]
    
    # Generate embeddings in batches for efficiency
    embeddings = embedder.embed_batch(texts)
    
    # Add embeddings to each item in the data
    for item, embedding in zip(data, embeddings):
        item['embeddings'] = embedding
    
    # Save the data with embeddings
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Successfully added embeddings to {len(data)} items and saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "data/chunked/chunked_file.json"
    output_file = "data/embeddings/embedded_file.json"
    
    add_embeddings_to_data(input_file, output_file, model_name='all-mpnet-base-v2')