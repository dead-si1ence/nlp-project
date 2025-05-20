#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Index and retrieve document chunks using FAISS.
"""

import os
import faiss
import numpy as np
import pickle
from typing import Dict, List, Tuple, Union, Any


class DocumentIndex:
    """
    A class for indexing and retrieving document chunks using FAISS.
    """

    def __init__(self, dimension: int = 384) -> None:
        """
        Initialize the DocumentIndex.

        Args:
            dimension: Dimension of the embeddings to index
        """
        self.dimension = dimension
        # Use IndexIDMap to support adding with IDs
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

        # Metadata storage mapping indices to document information
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.nextId = 0

    def addDocuments(
        self, embeddings: np.ndarray, metadataList: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add document embeddings and metadata to the index.

        Args:
            embeddings: Document embeddings as a numpy array
            metadataList: List of metadata dictionaries for each embedding

        Returns:
            List of assigned IDs
        """
        if len(embeddings) != len(metadataList):
            raise ValueError(
                "Number of embeddings must match number of metadata entries"
            )

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Get the number of embeddings
        numEmbeddings = embeddings.shape[0]

        # Generate IDs for the embeddings
        ids = np.arange(self.nextId, self.nextId + numEmbeddings, dtype=np.int64)

        # Add embeddings to the index
        self.index.add_with_ids(embeddings, ids)

        # Add metadata
        for i, id_val in enumerate(ids):
            self.metadata[int(id_val)] = metadataList[i]

        # Update next ID
        self.nextId += numEmbeddings

        return ids.tolist()

    def search(
        self, query: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Search the index for similar documents.

        Args:
            query: Query embedding
            k: Number of results to return

        Returns:
            Tuple of (distances, indices, metadata)
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize query for cosine similarity
        faiss.normalize_L2(query)

        # Search the index
        distances, indices = self.index.search(query, k)

        # Collect metadata for the results
        metadata = []
        for idx in indices[0]:
            if (
                idx != -1 and idx in self.metadata
            ):  # FAISS returns -1 for padded results
                metadata.append(self.metadata[int(idx)])
            else:
                metadata.append({})

        return distances[0], indices[0], metadata

    def saveIndex(self, indexPath: str) -> None:
        """
        Save the index and metadata to disk.

        Args:
            indexPath: Path to save the index, without extension
        """
        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(indexPath) if os.path.dirname(indexPath) else ".",
            exist_ok=True,
        )

        # Save the FAISS index
        faiss.write_index(self.index, f"{indexPath}.faiss")

        # Save the metadata and next ID
        with open(f"{indexPath}.meta", "wb") as f:
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "next_id": self.nextId,
                    "dimension": self.dimension,
                },
                f,
            )

    @classmethod
    def loadIndex(cls, indexPath: str) -> "DocumentIndex":
        """
        Load the index and metadata from disk.

        Args:
            indexPath: Path to load the index from, without extension

        Returns:
            DocumentIndex instance
        """
        # Check if index files exist
        if not os.path.exists(f"{indexPath}.faiss") or not os.path.exists(
            f"{indexPath}.meta"
        ):
            raise FileNotFoundError(f"Index files not found at {indexPath}")

        # Load the metadata and next ID
        with open(f"{indexPath}.meta", "rb") as f:
            metaDict = pickle.load(f)

        # Load the FAISS index
        index = faiss.read_index(f"{indexPath}.faiss")

        # Create a new instance
        instance = cls(dimension=metaDict["dimension"])
        instance.index = index
        instance.metadata = metaDict["metadata"]
        instance.nextId = metaDict["next_id"]

        return instance

    def reset(self) -> None:
        """
        Reset the index to its initial state.
        """
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        self.metadata = {}
        self.nextId = 0
