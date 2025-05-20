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

    def __init__(
        self,
        indexPath: str = "./data/index/index",
        embedderModel: str = "all-MiniLM-L6-v2",
        qaModel: str = "deepset/roberta-base-squad2",
    ) -> None:
        """
        Initialize the DocumentQA system.

        Args:
            indexPath: Path to store/load the document index
            embedderModel: Name of the sentence-transformer model to use
            qaModel: Name of the QA model to use
        """
        self.indexPath = indexPath
        self.embedderModel = embedderModel
        self.qaModel = qaModel

        self.extractor = DocumentExtractor()
        self.chunker = TextChunker(chunkSize=512, chunkOverlap=50)
        self.embedder = TextEmbedder(modelName=embedderModel)

        # Try to load existing index or create a new one
        try:
            self.index = DocumentIndex.loadIndex(indexPath)
        except FileNotFoundError:
            # Create a new index if not found
            self.index = DocumentIndex(dimension=self.embedder.embedding_dimension)

        self.qa = QuestionAnswerer(modelName=qaModel)
        self.queryEngine = QueryEngine(
            index=self.index, embedder=self.embedder, qa=self.qa
        )

    def processDocument(self, filePath: str) -> Dict[str, Any]:
        """
        Process a document and add it to the index.

        Args:
            filePath: Path to the document to process

        Returns:
            Dict with document info and status
        """
        try:
            # Extract text from the document
            pages = self.extractor.extractText(filePath)

            # Create a document ID
            docId = str(uuid.uuid4())

            # Chunk the document
            chunks = self.chunker.chunkDocument(pages)

            # Prepare chunks with metadata
            chunkTexts = []
            chunkMetadata = []

            for pageNum, chunkText in chunks:
                if not chunkText.strip():
                    continue

                chunkTexts.append(chunkText)
                chunkMetadata.append(
                    {
                        "doc_id": docId,
                        "file_path": filePath,
                        "file_name": os.path.basename(filePath),
                        "page": pageNum,
                        "text": chunkText,
                    }
                )

            # Embed the chunks
            embeddings = self.embedder.embedText(chunkTexts)

            # Add to the index
            ids = self.index.addDocuments(embeddings, chunkMetadata)

            # Save the index
            self.index.saveIndex(self.indexPath)

            return {
                "status": "success",
                "document_id": docId,
                "filename": os.path.basename(filePath),
                "pages": len(pages),
                "chunks": len(chunks),
                "message": f"Successfully processed document: {os.path.basename(filePath)}",
            }

        except Exception as e:
            return {
                "status": "error",
                "filename": os.path.basename(filePath),
                "message": f"Error processing document: {str(e)}",
            }

    def askQuestion(self, question: str, topK: int = 5) -> Dict[str, Any]:
        """
        Ask a question about the processed documents.

        Args:
            question: The question to ask
            topK: Number of relevant documents to retrieve

        Returns:
            Dict with answer and source documents
        """
        result = self.queryEngine.query(
            queryText=question, topK=topK, rerank=True, returnSourceDocs=True
        )

        return result

    def listDocuments(self) -> Dict[str, Any]:
        """
        List all documents in the index.

        Returns:
            Dict with document info
        """
        # Collect unique document IDs and file paths
        documents = {}

        for idx, metadata in self.index.metadata.items():
            doc_id = metadata.get("doc_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "file_path": metadata.get("file_path"),
                    "file_name": metadata.get("file_name"),
                    "chunks": 0,
                }

            if doc_id:
                documents[doc_id]["chunks"] += 1

        return {
            "status": "success",
            "document_count": len(documents),
            "documents": list(documents.values()),
        }

    def resetIndex(self) -> Dict[str, Any]:
        """
        Reset the index, removing all documents.

        Returns:
            Dict with status info
        """
        self.index.reset()
        self.index.saveIndex(self.indexPath)

        return {"status": "success", "message": "Index has been reset"}


def main() -> None:
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="Document QA System CLI")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Upload command
    upload_parser = subparsers.add_parser(
        "upload", help="Upload and process a document"
    )
    upload_parser.add_argument("file_path", help="Path to the document to process")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the documents")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of documents to retrieve"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all documents in the index")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset the index")

    # Parse arguments
    args = parser.parse_args()

    # Create the DocumentQA instance
    qa_system = DocumentQA()

    # Execute the appropriate command
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
        return

    # Print the result as JSON
    print("\x1bc")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
