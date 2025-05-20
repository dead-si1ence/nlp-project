#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embed text chunks using sentence-transformers.
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    A class for embedding text chunks using sentence-transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the TextEmbedder.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise Exception(f"Failed to load embedding model '{model_name}': {e}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text chunks.
        
        Args:
            texts: List of text chunks to embed
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.array([])
            
        try:
            # Generate embeddings for all texts at once (more efficient)
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            raise Exception(f"Embedding generation failed: {e}")
    
    def embedDocumentChunks(self, chunked_doc: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Embed all chunks in a chunked document.
        
        Args:
            chunked_doc: Dict mapping page numbers to lists of text chunks
            
        Returns:
            Dict mapping page numbers to dicts containing:
              - 'chunks': original text chunks
              - 'embeddings': numpy array of embeddings
        """
        result = {}
        
        for page_num, chunks in chunked_doc.items():
            if not chunks:
                continue
                
            embeddings = self.embed(chunks)
            
            result[page_num] = {
                'chunks': chunks,
                'embeddings': embeddings
            }
            
        return result


if __name__ == "__main__":
    # Sample usage
    embedder = TextEmbedder()
    
    test_texts = [
        "This is a sample sentence to embed.",
        "Embeddings are vector representations of text.",
        "We use sentence-transformers to create these embeddings."
    ]
    
    embeddings = embedder.embed(test_texts)
    
    print(f"Embedding dimension: {embedder.embedding_dimension}")
    print(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape}")
    
    # Calculate similarities between embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    
    print("\nSimilarities between sentences:")
    for i in range(len(test_texts)):
        for j in range(i+1, len(test_texts)):
            print(f"Similarity between sentence {i+1} and {j+1}: {similarities[i][j]:.4f}")
