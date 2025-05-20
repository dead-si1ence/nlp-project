#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embed text using sentence-transformers.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    A class for embedding text using sentence-transformers.
    """
    
    def __init__(self, modelName: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the TextEmbedder.
        
        Args:
            modelName: Name of the sentence-transformer model to use
        """
        self.modelName = modelName
        self.model = SentenceTransformer(modelName)
        self.embeddingDimension = self.model.get_sentence_embedding_dimension()
    
    @property
    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Dimension of the embeddings
        """
        return self.embeddingDimension
    
    def embedText(self, text: Union[str, List[str]], 
                 batchSize: int = 32, 
                 showProgress: bool = False) -> np.ndarray:
        """
        Embed text using the model.
        
        Args:
            text: Text or list of texts to embed
            batchSize: Batch size for embedding
            showProgress: Whether to show a progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]
        
        # Filter out empty strings to avoid errors
        text = [t for t in text if t.strip()]
        
        if not text:
            return np.array([])
        
        # Encode the texts
        embeddings = self.model.encode(
            text,
            batch_size=batchSize,
            show_progress_bar=showProgress,
            convert_to_numpy=True
        )
        
        return embeddings
