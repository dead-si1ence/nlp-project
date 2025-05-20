#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command line interface for the document QA system.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Union, Any
import uuid

from .extract import DocumentExtractor
from .chunk import TextChunker
from .embed import TextEmbedder
from .index import DocumentIndex
from .qa import QuestionAnswerer
from .query import QueryEngine


class DocumentQA:
    """
    Main class for the document QA system.
    """
    
    def __init__(self, 
                 index_path: str = "index",
                 embedder_model: str = "all-MiniLM-L6-v2",
                 qa_model: str = "deepset/roberta-base-squad2") -> None:
        """
        Initialize the DocumentQA system.
        
        Args:
            index_path: Path to store/load the document index
            embedder_model: Name of the sentence-transformer model to use
            qa_model: Name of the QA model to use
        """
        self.index_path = index_path
        self.embedder_model = embedder_model
        self.qa_model = qa_model
        
        self.extractor = DocumentExtractor()
        self.chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        self.embedder = TextEmbedder(model_name=embedder_model)
        
        # Try to load existing index or create a new one
        try:
            self.index = DocumentIndex.loadIndex(index_path)
        except FileNotFoundError:
            # Create a new index if not found
            self.index = DocumentIndex(dimension=self.embedder.embedding_dimension)
        
        self.qa = QuestionAnswerer(model_name=qa_model)
        self.query_engine = QueryEngine(index=self.index, embedder=self.embedder, qa=self.qa)
    
    def processDocument(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and add it to the index.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing processing results
        """
        try:
            # Generate a document ID
            doc_id = str(uuid.uuid4())
            
            # Extract doc info
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            doc_info = {
                "filename": filename,
                "path": os.path.abspath(file_path),
                "size": file_size
            }
            
            # Extract text from document
            extracted_text = self.extractor.extract(file_path)
            
            # Chunk the text - use "auto" method to detect markdown structure
            chunked_doc = self.chunker.processDocument(extracted_text, method="auto")
            
            # Embed the chunks
            embedded_chunks = self.embedder.embedDocumentChunks(chunked_doc)
            
            # Add to index
            self.index.addDocument(doc_id, doc_info, embedded_chunks)
            
            # Save index after adding document
            self.index.saveIndex(self.index_path)
            
            # Get stats
            stats = self.index.getStats()
            
            # Prepare result
            page_count = len(extracted_text)
            chunk_count = sum(len(chunks) for chunks in chunked_doc.values())
            
            result = {
                "success": True,
                "doc_id": doc_id,
                "filename": filename,
                "page_count": page_count,
                "chunk_count": chunk_count,
                "index_stats": stats
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def askQuestion(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to answer
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Dict containing answer information
        """
        try:
            result = self.query_engine.query(question, top_k=top_k)
            result["success"] = True
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def listDocuments(self) -> Dict[str, Any]:
        """
        List all documents in the index.
        
        Returns:
            Dict containing list of documents
        """
        try:
            documents = []
            
            for doc_id, doc_info in self.index.document_map.items():
                documents.append({
                    "doc_id": doc_id,
                    **doc_info
                })
            
            return {
                "success": True,
                "documents": documents,
                "count": len(documents),
                "index_stats": self.index.getStats()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def resetIndex(self) -> Dict[str, bool]:
        """
        Reset the index, removing all documents.
        
        Returns:
            Dict indicating success or failure
        """
        try:
            self.index.reset()
            # Save the reset index to disk
            self.index.saveIndex(self.index_path)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


def main() -> None:
    """
    Main function for CLI.
    """
    parser = argparse.ArgumentParser(description="Document QA CLI")
    
    # Global arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--output", type=str, help="Path to save output", default="answer.txt")
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload and process a document")
    upload_parser.add_argument("file_path", help="Path to the document file")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of relevant chunks to retrieve")
    
    # List command
    subparsers.add_parser("list", help="List all documents in the index")
    
    # Reset command
    subparsers.add_parser("reset", help="Reset the index, removing all documents")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create DocumentQA instance
    qa_system = DocumentQA()
    
    # Execute command
    if args.command == "upload":
        result = qa_system.processDocument(args.file_path)
    elif args.command == "ask":
        result = qa_system.askQuestion(args.question, args.top_k)
    elif args.command == "list":
        result = qa_system.listDocuments()
    elif args.command == "reset":
        result = qa_system.resetIndex()
    else:
        parser.print_help()
        sys.exit(1)
    
    # Save full result to output file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    # Print based on debug mode
    if args.debug:
        # Debug mode: print full JSON result
        print(json.dumps(result, indent=2))
    elif args.command == "ask" and result.get("success", False):
        # Normal mode for ask command: only print the answer
        print(result.get("answer", "No answer found"))


if __name__ == "__main__":
    main()
