#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Store and retrieve embeddings using FAISS.
"""

import os
import pickle
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import faiss


class DocumentIndex:
    """
    A class for indexing and retrieving document embeddings using FAISS.
    """
    
    def __init__(self, dimension: int = 384) -> None:
        """
        Initialize the DocumentIndex.
        
        Args:
            dimension: Dimension of the embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index (normalized for cosine similarity)
        
        # Store metadata separately
        self.document_map = {}  # Maps document_id to document info
        self.id_to_metadata = {}  # Maps vector ID to metadata (doc_id, page, chunk_id)
        self.next_id = 0
    
    def addDocument(self, 
                   doc_id: str, 
                   doc_info: Dict[str, Any], 
                   embedded_chunks: Dict[str, Dict[str, Any]]) -> None:
        """
        Add a document's embedded chunks to the index.
        
        Args:
            doc_id: Unique identifier for the document
            doc_info: Document metadata (filename, path, etc.)
            embedded_chunks: Dict mapping page numbers to dicts containing:
              - 'chunks': original text chunks
              - 'embeddings': numpy array of embeddings
        """
        # Store document info
        self.document_map[doc_id] = doc_info
        
        # Process each page
        for page_num, page_data in embedded_chunks.items():
            chunks = page_data['chunks']
            embeddings = page_data['embeddings']
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to the index
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = self.next_id
                
                # Store metadata
                self.id_to_metadata[vector_id] = {
                    'doc_id': doc_id,
                    'page': page_num,
                    'chunk_id': i,
                    'text': chunk
                }
                
                # Add to FAISS index
                self.index.add(np.array([embedding]))
                
                self.next_id += 1
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using a query embedding.
        
        Args:
            query_embedding: Embedding of the query text
            top_k: Number of results to return
            
        Returns:
            List of dicts containing:
              - 'score': similarity score
              - 'doc_id': document ID
              - 'page': page number
              - 'chunk_id': chunk ID
              - 'text': chunk text
              - 'doc_info': document info
        """
        if self.index.ntotal == 0:
            return []
            
        # Normalize query embedding
        query_embedding_np = np.array([query_embedding])
        faiss.normalize_L2(query_embedding_np)
        
        # Search the index
        scores, ids = self.index.search(query_embedding_np, min(top_k, self.index.ntotal))
        
        results = []
        for score, vector_id in zip(scores[0], ids[0]):
            if vector_id < 0 or vector_id not in self.id_to_metadata:
                continue
                
            metadata = self.id_to_metadata[vector_id]
            doc_id = metadata['doc_id']
            
            result = {
                'score': float(score),
                'doc_id': doc_id,
                'page': metadata['page'],
                'chunk_id': metadata['chunk_id'],
                'text': metadata['text'],
                'doc_info': self.document_map.get(doc_id, {})
            }
            
            results.append(result)
            
        return results
    
    def saveIndex(self, file_path: str) -> None:
        """
        Save the index and metadata to a file.
        
        Args:
            file_path: Path to save the index and metadata
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save metadata
        metadata = {
            'dimension': self.dimension,
            'document_map': self.document_map,
            'id_to_metadata': self.id_to_metadata,
            'next_id': self.next_id
        }
        
        # Save FAISS index
        faiss.write_index(self.index, f".{file_path}.faiss")
        
        # Save metadata separately
        with open(f"{file_path}.meta", 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def loadIndex(cls, file_path: str) -> 'DocumentIndex':
        """
        Load an index and metadata from a file.
        
        Args:
            file_path: Path to load the index and metadata
            
        Returns:
            Loaded DocumentIndex instance
        """
        # Check if files exist
        if not (os.path.exists(f"{file_path}.faiss") and 
                os.path.exists(f"{file_path}.meta")):
            raise FileNotFoundError(f"Index files not found at {file_path}")
            
        # Load metadata
        with open(f"{file_path}.meta", 'rb') as f:
            metadata = pickle.load(f)
            
        # Load FAISS index
        index = faiss.read_index(f"{file_path}.faiss")
        
        # Create instance
        instance = cls(dimension=metadata['dimension'])
        instance.index = index
        instance.document_map = metadata['document_map']
        instance.id_to_metadata = metadata['id_to_metadata']
        instance.next_id = metadata['next_id']
        
        return instance
    
    def reset(self) -> None:
        """
        Reset the index and metadata.
        """
        self.index = faiss.IndexFlatIP(self.dimension)
        self.document_map = {}
        self.id_to_metadata = {}
        self.next_id = 0
    
    def getStats(self) -> Dict[str, Any]:
        """
        Get stats about the index.
        
        Returns:
            Dict containing index statistics
        """
        return {
            'dimensions': self.dimension,
            'total_vectors': self.index.ntotal,
            'document_count': len(self.document_map),
            'chunk_count': len(self.id_to_metadata)
        }


if __name__ == "__main__":
    # Sample usage
    import numpy as np
    
    # Create a DocumentIndex
    index = DocumentIndex(dimension=3)
    
    # Create some test embeddings
    doc1_embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.1, 0.1]
    ])
    
    doc2_embeddings = np.array([
        [0.1, 0.8, 0.1],
        [0.0, 0.0, 1.0]
    ])
    
    # Normalize embeddings
    faiss.normalize_L2(doc1_embeddings)
    faiss.normalize_L2(doc2_embeddings)
    
    # Add documents to the index
    index.addDocument(
        doc_id="doc1",
        doc_info={"filename": "doc1.pdf", "path": "/path/to/doc1.pdf"},
        embedded_chunks={
            "0": {"chunks": ["This is document 1, chunk 1", "This is document 1, chunk 2"], 
                  "embeddings": doc1_embeddings}
        }
    )
    
    index.addDocument(
        doc_id="doc2",
        doc_info={"filename": "doc2.pdf", "path": "/path/to/doc2.pdf"},
        embedded_chunks={
            "0": {"chunks": ["This is document 2, chunk 1", "This is document 2, chunk 2"], 
                  "embeddings": doc2_embeddings}
        }
    )
    
    # Test search
    query_embedding = np.array([0.9, 0.1, 0.0])
    faiss.normalize_L2(np.array([query_embedding]))
    
    results = index.search(query_embedding, top_k=2)
    
    print("Search results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Document: {result['doc_id']} ({result['doc_info']['filename']})")
        print(f"  Text: {result['text']}")
        print("-" * 40)
    
    # Save and load the index
    index.saveIndex("test_index")
    
    loaded_index = DocumentIndex.loadIndex("test_index")
    print(f"Loaded index stats: {loaded_index.getStats()}")
    
    # Clean up test files
    import os
    if os.path.exists("test_index.faiss"):
        os.remove("test_index.faiss")
    if os.path.exists("test_index.meta"):
        os.remove("test_index.meta")
