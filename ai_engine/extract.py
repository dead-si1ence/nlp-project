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

    def extractFromPDF(self, filePath: str) -> Dict[int, str]:
        """
        Extract text from a PDF file.

        Args:
            filePath: Path to the PDF file

        Returns:
            Dict mapping page numbers (0-indexed) to text content
        """
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"File not found: {filePath}")

        if not filePath.lower().endswith(".pdf"):
            raise ValueError(f"File {filePath} is not a PDF")

        result = {}

        try:
            # Open the PDF
            with fitz.open(filePath) as doc:
                # Iterate through pages
                for pageNum, page in enumerate(doc):
                    # Extract text from the page
                    text = page.get_text()
                    result[pageNum] = text
            return result
        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {str(e)}")

    def extractFromText(self, filePath: str) -> Dict[int, str]:
        """
        Extract text from a plain text file.

        Args:
            filePath: Path to the text file

        Returns:
            Dict mapping page numbers (0-indexed) to text content
            For text files, all content is considered as page 0
        """
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"File not found: {filePath}")

        try:
            with open(filePath, "r", encoding="utf-8") as f:
                content = f.read()
            return {0: content}
        except Exception as e:
            raise RuntimeError(f"Error extracting text from file: {str(e)}")

    def extractText(self, filePath: str) -> Dict[int, str]:
        """
        Extract text from a file based on its extension.

        Args:
            filePath: Path to the file

        Returns:
            Dict mapping page numbers to text content
        """
        if filePath.lower().endswith(".pdf"):
            return self.extractFromPDF(filePath)
        elif filePath.lower().endswith((".txt", ".md", ".csv")):
            return self.extractFromText(filePath)
        else:
            raise ValueError(f"Unsupported file format: {filePath}")
