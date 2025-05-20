#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract text from PDF files using PyMuPDF (fitz).
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import fitz  # PyMuPDF


class DocumentExtractor:
    """
    A class for extracting text content from various document formats.
    Currently supports PDF files using PyMuPDF.
    """
    
    def __init__(self) -> None:
        """
        Initialize the DocumentExtractor.
        """
        pass
    
    def extractFromPDF(self, file_path: str) -> Dict[int, str]:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict mapping page numbers (0-indexed) to text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File {file_path} is not a PDF")
            
        result = {}
        
        try:
            # Open the PDF
            with fitz.open(file_path) as doc:
                # Iterate through pages
                for page_num, page in enumerate(doc):
                    # Extract text from the page
                    text = page.get_text()
                    result[page_num] = text
        except Exception as e:
            raise Exception(f"Error extracting text from {file_path}: {e}")
            
        return result
    
    def extractFromText(self, file_path: str) -> Dict[int, str]:
        """
        Extract text from a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dict with a single key 0 mapped to the file content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return {0: text}
        except Exception as e:
            raise Exception(f"Error extracting text from {file_path}: {e}")
    
    def extract(self, file_path: str) -> Dict[int, str]:
        """
        Extract text from a document based on its file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict mapping page numbers to text content
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extractFromPDF(file_path)
        elif file_ext in ('.txt', '.text'):
            return self.extractFromText(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")


if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract.py <file_path>")
        sys.exit(1)
        
    extractor = DocumentExtractor()
    try:
        content = extractor.extract(sys.argv[1])
        for page_num, text in content.items():
            print(f"--- Page {page_num + 1} ---")
            print(text[:300] + "...\n")  # Print first 300 chars
    except Exception as e:
        print(f"Error: {e}")
