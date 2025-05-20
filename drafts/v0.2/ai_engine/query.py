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
              query_text: str, 
              top_k: int = 5,
              rerank: bool = True,
              return_source_docs: bool = True) -> Dict[str, Any]:
        """
        Query the index and answer a question.
        
        Args:
            query_text: The query or question text
            top_k: Number of relevant documents to retrieve
            rerank: Whether to rerank results by QA score
            return_source_docs: Whether to include source documents in response
            
        Returns:
            Dict containing:
              - 'query': original query text
              - 'answer': best answer text
              - 'answer_score': confidence score of answer
              - 'source_document': source document info (if return_source_docs is True)
              - 'sources': list of source documents used (if return_source_docs is True)
        """
        # Embed the query
        query_embedding = self.embedder.embed([query_text])[0]
        
        # Search for relevant chunks
        retrieved_chunks = self.index.search(query_embedding, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                'query': query_text,
                'answer': "No relevant documents found to answer this question.",
                'answer_score': 0.0
            }
        
        # Answer the question
        answers = self.qa.answerWithRetrievedContext(
            question=query_text,
            retrieved_documents=retrieved_chunks
        )
        
        # Construct response
        response = {
            'query': query_text,
            'answer': answers[0]['answer'],
            'answer_score': float(answers[0]['score'])
        }
        
        # Include source documents if requested
        if return_source_docs:
            response['source_document'] = {
                'doc_id': answers[0]['doc_id'],
                'page': answers[0]['page'],
                'chunk_id': answers[0]['chunk_id'],
                'context': answers[0]['context'],
                'info': answers[0].get('doc_info', {})
            }
            
            response['sources'] = [
                {
                    'doc_id': chunk['doc_id'],
                    'page': chunk['page'],
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'score': chunk['score'],
                    'info': chunk['doc_info']
                }
                for chunk in retrieved_chunks
            ]
        
        return response


if __name__ == "__main__":
    # Sample usage (mocking parts that would require model loading)
    import faiss
    import numpy as np
    from unittest.mock import MagicMock
    
    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = np.array([[0.9, 0.1, 0.0]])
    
    # Create test index
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
    
    # Mock QA
    mock_qa = MagicMock()
    mock_qa.answerWithRetrievedContext.return_value = [{
        'answer': 'This is a mock answer',
        'score': 0.95,
        'context': 'This is document 1, chunk 1',
        'start': 8,
        'end': 25,
        'doc_id': 'doc1',
        'page': '0',
        'chunk_id': 0,
        'retrieval_score': 0.9,
        'doc_info': {"filename": "doc1.pdf", "path": "/path/to/doc1.pdf"}
    }]
    
    # Create query engine
    query_engine = QueryEngine(
        index=index,
        embedder=mock_embedder,
        qa=mock_qa
    )
    
    # Test query
    result = query_engine.query("What is this document about?", top_k=2)
    
    print("Query result:")
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Answer score: {result['answer_score']}")
    print(f"Source document: {result['source_document']['doc_id']} "
          f"(Page {result['source_document']['page']}, "
          f"Chunk {result['source_document']['chunk_id']})")
