#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split documents into manageable chunks for embedding and retrieval.
"""

import re
from typing import Dict, List, Tuple, Union, Any


class TextChunker:
    """
    A class for splitting text into chunks for embedding and retrieval.
    """

    def __init__(self, chunkSize: int = 512, chunkOverlap: int = 50) -> None:
        """
        Initialize the TextChunker.

        Args:
            chunkSize: The target size of each chunk in characters
            chunkOverlap: The amount of overlap between chunks in characters
        """
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap

    def _splitIntoSentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.

        Args:
            text: Text to split into sentences

        Returns:
            List of sentences
        """
        # Simple sentence splitter based on common sentence endings
        # This is not as accurate as NLTK's sent_tokenize but doesn't require external resources
        sentence_endings = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunkText(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # If text is shorter than chunk size, return as a single chunk
        if len(text) <= self.chunkSize:
            return [text]

        sentences = self._splitIntoSentences(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If a single sentence is longer than the chunk size, split it further
            if sentence_length > self.chunkSize:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long sentence into smaller parts
                words = sentence.split()
                current_part = []
                current_part_length = 0

                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    if current_part_length + word_length <= self.chunkSize:
                        current_part.append(word)
                        current_part_length += word_length
                    else:
                        if current_part:
                            chunks.append(" ".join(current_part))
                        current_part = [word]
                        current_part_length = word_length

                if current_part:
                    chunks.append(" ".join(current_part))

            # Normal case: add sentence to chunk if it fits
            elif current_length + sentence_length + 1 <= self.chunkSize:  # +1 for space
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length

        # Add the last chunk if there is one
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Create overlapping chunks if needed
        if self.chunkOverlap > 0 and len(chunks) > 1:
            return self._createOverlappingChunks(chunks)

        return chunks

    def _createOverlappingChunks(self, chunks: List[str]) -> List[str]:
        """
        Create overlapping chunks from a list of chunks.

        Args:
            chunks: The non-overlapping chunks

        Returns:
            List of overlapping chunks
        """
        overlapping_chunks = []

        for i in range(len(chunks)):
            overlapping_chunks.append(chunks[i])

            # Skip creating an overlapping chunk for the last chunk
            if i < len(chunks) - 1:
                # Get the end of the current chunk and the start of the next chunk
                current_words = chunks[i].split()
                next_words = chunks[i + 1].split()

                # Calculate how many words to take from each chunk
                current_overlap_size = min(self.chunkOverlap // 2, len(current_words))
                next_overlap_size = min(self.chunkOverlap // 2, len(next_words))

                # Create an overlapping chunk
                overlap_chunk = " ".join(
                    current_words[-current_overlap_size:]
                    + next_words[:next_overlap_size]
                )

                if overlap_chunk.strip():
                    overlapping_chunks.append(overlap_chunk)

        return overlapping_chunks

    def chunkDocument(self, pages: Dict[int, str]) -> List[Tuple[int, str]]:
        """
        Chunk a document represented as a dictionary of page numbers to text.

        Args:
            pages: Dict mapping page numbers to text content

        Returns:
            List of tuples (page_number, chunk_text)
        """
        result = []

        for page_num, page_text in pages.items():
            page_chunks = self.chunkText(page_text)
            for chunk in page_chunks:
                result.append((page_num, chunk))

        return result

        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If a single sentence is longer than the chunk size, split it further
            if sentence_length > self.chunkSize:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long sentence into smaller parts
                words = sentence.split()
                current_part = []
                current_part_length = 0

                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    if current_part_length + word_length <= self.chunkSize:
                        current_part.append(word)
                        current_part_length += word_length
                    else:
                        if current_part:
                            chunks.append(" ".join(current_part))
                        current_part = [word]
                        current_part_length = word_length

                if current_part:
                    chunks.append(" ".join(current_part))

            # Normal case: add sentence to chunk if it fits
            elif current_length + sentence_length + 1 <= self.chunkSize:  # +1 for space
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length

        # Add the last chunk if there is one
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Create overlapping chunks if needed
        if self.chunkOverlap > 0 and len(chunks) > 1:
            return self._createOverlappingChunks(chunks)

        return chunks

    def _createOverlappingChunks(self, chunks: List[str]) -> List[str]:
        """
        Create overlapping chunks from a list of chunks.

        Args:
            chunks: The non-overlapping chunks

        Returns:
            List of overlapping chunks
        """
        overlapping_chunks = []

        for i in range(len(chunks)):
            overlapping_chunks.append(chunks[i])

            # Skip creating an overlapping chunk for the last chunk
            if i < len(chunks) - 1:
                # Get the end of the current chunk and the start of the next chunk
                current_words = chunks[i].split()
                next_words = chunks[i + 1].split()

                # Calculate how many words to take from each chunk
                current_overlap_size = min(self.chunkOverlap // 2, len(current_words))
                next_overlap_size = min(self.chunkOverlap // 2, len(next_words))

                # Create an overlapping chunk
                overlap_chunk = " ".join(
                    current_words[-current_overlap_size:]
                    + next_words[:next_overlap_size]
                )

                if overlap_chunk.strip():
                    overlapping_chunks.append(overlap_chunk)

        return overlapping_chunks

    def chunkDocument(self, pages: Dict[int, str]) -> List[Tuple[int, str]]:
        """
        Chunk a document represented as a dictionary of page numbers to text.

        Args:
            pages: Dict mapping page numbers to text content

        Returns:
            List of tuples (page_number, chunk_text)
        """
        result = []

        for page_num, page_text in pages.items():
            page_chunks = self.chunkText(page_text)
            for chunk in page_chunks:
                result.append((page_num, chunk))

        return result
