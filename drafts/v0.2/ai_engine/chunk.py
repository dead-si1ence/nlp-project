#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunk text into smaller pieces for embedding and retrieval.
"""

import re
from typing import Dict, List, Tuple, Union
import nltk
from nltk.tokenize import sent_tokenize


class TextChunker:
    """
    A class for breaking text into smaller chunks suitable for embedding.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """
        Initialize the TextChunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def chunkByParagraph(self, text: str) -> List[str]:
        """
        Split text into paragraphs and then into chunks of specified size.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Split by double newlines to get paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for para in paragraphs:
            # If paragraph is too long, split it into sentences
            if len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_length = 0
                
                # Split paragraph into sentences
                sentences = sent_tokenize(para)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) + 1 <= self.chunk_size:
                        if temp_chunk:
                            temp_chunk += " "
                        temp_chunk += sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sentence
                
                if temp_chunk:
                    chunks.append(temp_chunk)
            
            # If adding paragraph doesn't exceed chunk size
            elif current_length + len(para) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                    current_length += 2
                current_chunk += para
                current_length += len(para)
            else:
                chunks.append(current_chunk)
                current_chunk = para
                current_length = len(para)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunkBySentences(self, text: str) -> List[str]:
        """
        Split text into sentences and then into chunks of specified size.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            # If sentence is longer than chunk_size, split it into smaller pieces
            if len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_length = 0
                
                # Split long sentence into smaller chunks
                words = sentence.split()
                temp_chunk = ""
                
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= self.chunk_size:
                        if temp_chunk:
                            temp_chunk += " "
                        temp_chunk += word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = word
                
                if temp_chunk:
                    chunks.append(temp_chunk)
            
            # If adding sentence doesn't exceed chunk size
            elif current_length + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " "
                    current_length += 1
                current_chunk += sentence
                current_length += len(sentence)
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
                current_length = len(sentence)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunkBySection(self, text: str) -> List[str]:
        """
        Split text by markdown section headers (## headers).
        
        This is especially useful for markdown documents with structured sections 
        containing lists of items.
        
        Args:
            text: Text to chunk (markdown formatted)
            
        Returns:
            List of text chunks, with each chunk containing a complete section
        """
        # If no section headers, treat as one section
        if not re.search(r'^##\s+', text, re.MULTILINE):
            return [text]
        
        # Split by section headers (## headers)
        section_pattern = r'^(##\s+.*?)(?=^##\s+|\Z)'
        sections = re.findall(section_pattern, text, re.MULTILINE | re.DOTALL)
        
        # Add the title/intro if it exists (content before first ##)
        intro_match = re.match(r'^(.*?)##\s+', text, re.DOTALL)
        if intro_match and intro_match.group(1).strip():
            sections = [intro_match.group(1).strip()] + sections
            
        # Clean up sections
        sections = [s.strip() for s in sections if s.strip()]
        
        # Group smaller sections if needed
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if not current_chunk:
                current_chunk = section
            elif len(current_chunk) + len(section) + 2 <= self.chunk_size:
                # Combine sections if they're small
                current_chunk += "\n\n" + section
            else:
                # Section would make chunk too large
                chunks.append(current_chunk)
                current_chunk = section
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def chunkWithOverlap(self, text: str, method: str = "paragraph") -> List[str]:
        """
        Chunk text with overlap between chunks.
        
        Args:
            text: Text to chunk
            method: Chunking method, either "paragraph" or "sentence"
            
        Returns:
            List of text chunks with overlap
        """
        if method == "paragraph":
            initial_chunks = self.chunkByParagraph(text)
        elif method == "section":
            initial_chunks = self.chunkBySection(text)
        else:
            initial_chunks = self.chunkBySentences(text)
        
        # If chunk_overlap is 0 or we have very few chunks, return as is
        if self.chunk_overlap == 0 or len(initial_chunks) <= 1:
            return initial_chunks
        
        # Create chunks with overlap
        overlapping_chunks = []
        
        for i, chunk in enumerate(initial_chunks):
            if i == 0:
                overlapping_chunks.append(chunk)
                continue
            
            prev_chunk = initial_chunks[i-1]
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) >= self.chunk_overlap else prev_chunk
            
            # Add overlap to beginning of current chunk
            new_chunk = overlap_text + " " + chunk
            overlapping_chunks.append(new_chunk)
        
        return overlapping_chunks
    
    def processDocument(self, doc_content: Dict[int, str], method: str = "paragraph") -> Dict[str, List[str]]:
        """
        Process an entire document and create chunks.
        
        Args:
            doc_content: Dict mapping page numbers to text content
            method: Chunking method, "paragraph", "section", or "sentence"
            
        Returns:
            Dict mapping page numbers to lists of chunks
        """
        result = {}
        
        for page_num, text in doc_content.items():
            page_key = str(page_num)
            
            # Auto-detect if text has markdown sections for better chunking
            if method == "auto" or (method == "paragraph" and re.search(r'^##\s+', text, re.MULTILINE)):
                # Use section-based chunking for markdown content with headers
                result[page_key] = self.chunkWithOverlap(text, "section")
            else:
                result[page_key] = self.chunkWithOverlap(text, method)
            
        return result


if __name__ == "__main__":
    # Sample usage
    text = """
    This is a sample paragraph. It contains several sentences.
    It should be chunked based on the specified method.
    
    This is another paragraph. The chunker should handle multiple paragraphs.
    It should also respect paragraph boundaries when chunking.
    
    If a paragraph is very long, it should be split into multiple chunks.
    However, we want to ensure that the chunks make semantic sense and don't
    cut sentences in the middle.
    """
    
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunkWithOverlap(text, "paragraph")
    
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} (length: {len(chunk)}):")
        print(chunk)
        print("-" * 40)
