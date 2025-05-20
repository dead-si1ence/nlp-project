#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Query module for retrieving relevant documents and answering questions.
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
from .embed import TextEmbedder
from .index import DocumentIndex
from .qa import QuestionAnswerer


class QueryEngine:
    """
    A class for querying the document index and answering questions.
    """
    
    def __init__(self, 
                 index: DocumentIndex,
                 embedder: TextEmbedder = None,
                 qa: QuestionAnswerer = None) -> None:
        """
        Initialize the QueryEngine.
        
        Args:
            index: Document index for retrieving relevant chunks
            embedder: Text embedder for embedding queries
            qa: Question answerer for answering questions
        """
        self.index = index
        
        # Initialize embedder if not provided
        if embedder is None:
            self.embedder = TextEmbedder()
        else:
            self.embedder = embedder
        
        # Initialize QA if not provided
        if qa is None:
            self.qa = QuestionAnswerer()
        else:
            self.qa = qa
    
    def query(self, 
              queryText: str, 
              topK: int = 5,
              rerank: bool = True,
              returnSourceDocs: bool = True) -> Dict[str, Any]:
        """
        Query the index and answer a question.
        
        Args:
            queryText: The query or question text
            topK: Number of relevant documents to retrieve
            rerank: Whether to rerank results by QA score
            returnSourceDocs: Whether to include source documents in response
            
        Returns:
            Dict containing:
                - answer: The answer to the question
                - score: Confidence score for the answer
                - source_docs: List of source documents (if return_source_docs=True)
        """
        # Embed the query
        queryEmbedding = self.embedder.embedText(queryText)
        
        # Search the index
        distances, indices, metadataList = self.index.search(queryEmbedding, k=topK)
        
        # Filter out invalid results
        validResults = []
        for i, (dist, idx, meta) in enumerate(zip(distances, indices, metadataList)):
            if idx != -1 and meta:  # Skip invalid results
                meta["score"] = float(dist)
                validResults.append(meta)
        
        if not validResults:
            return {
                "answer": "No relevant documents found to answer the question.",
                "score": 0.0,
                "source_docs": [] if returnSourceDocs else None
            }
        
        # Get context texts
        contexts = [result["text"] for result in validResults]
        
        # Concatenate contexts for better context
        combined_context = " ".join(contexts)
        
        # Answer the question
        qaResults = self.qa.answerQuestion(queryText, combined_context, topK=1)
        
        if isinstance(qaResults, list):
            bestResult = qaResults[0] if qaResults else {"answer": "Could not find an answer in the document.", "score": 0.0, "context": ""}
        else:
            bestResult = qaResults
            
        # If answer is empty, provide a fallback
        if not bestResult["answer"].strip():
            bestResult["answer"] = "Could not find a specific answer in the document. Please try rewording your question."
        
        # Prepare the response
        response = {
            "answer": bestResult["answer"],
            "score": bestResult["score"]
        }
        
        if returnSourceDocs:
            response["source_docs"] = validResults
        
        return response
    
    def simpleSearch(self, 
                    queryText: str, 
                    topK: int = 5) -> List[Dict[str, Any]]:
        """
        Simple semantic search without QA.
        
        Args:
            queryText: The query text
            topK: Number of relevant documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        # Embed the query
        queryEmbedding = self.embedder.embedText(queryText)
        
        # Search the index
        distances, indices, metadataList = self.index.search(queryEmbedding, k=topK)
        
        # Filter out invalid results
        validResults = []
        for i, (dist, idx, meta) in enumerate(zip(distances, indices, metadataList)):
            if idx != -1 and meta:  # Skip invalid results
                meta["score"] = float(dist)
                validResults.append(meta)
        
        return validResults
