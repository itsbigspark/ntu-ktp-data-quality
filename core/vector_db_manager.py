"""
Vector Database Manager - ChromaDB Integration
Provides fast semantic search and persistent embeddings storage
"""

import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time


class VectorDBManager:
    """
    Manages ChromaDB vector database for semantic search

    Features:
    - Create/delete collections
    - Add embeddings with metadata
    - Fast similarity search
    - Incremental updates
    - Persistent storage
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client

        Args:
            persist_directory: Directory to store ChromaDB data
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections in the database

        Returns:
            List of collection info dicts
        """
        collections = self.client.list_collections()

        result = []
        for collection in collections:
            try:
                count = collection.count()
                metadata = collection.metadata or {}

                result.append({
                    'name': collection.name,
                    'count': count,
                    'metadata': metadata,
                    'created_at': metadata.get('created_at', 'Unknown')
                })
            except Exception as e:
                result.append({
                    'name': collection.name,
                    'count': 0,
                    'metadata': {},
                    'error': str(e)
                })

        return result

    def create_collection(
        self,
        name: str,
        description: str = "",
        overwrite: bool = False
    ) -> chromadb.Collection:
        """
        Create a new collection

        Args:
            name: Collection name
            description: Collection description
            overwrite: If True, delete existing collection with same name

        Returns:
            ChromaDB collection object
        """
        # Sanitize name (ChromaDB requirements)
        name = name.replace(" ", "_").replace("-", "_").lower()
        name = "".join(c for c in name if c.isalnum() or c == "_")

        if overwrite:
            try:
                self.client.delete_collection(name=name)
            except:
                pass

        collection = self.client.get_or_create_collection(
            name=name,
            metadata={
                "description": description,
                "created_at": pd.Timestamp.now().isoformat()
            }
        )

        return collection

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection

        Args:
            name: Collection name

        Returns:
            True if deleted, False if not found
        """
        try:
            self.client.delete_collection(name=name)
            return True
        except Exception:
            return False

    def add_to_collection(
        self,
        collection_name: str,
        embeddings: np.ndarray,
        texts: List[str],
        ids: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Add embeddings to a collection

        Args:
            collection_name: Name of collection
            embeddings: Numpy array of embeddings (n_samples, embedding_dim)
            texts: List of original texts
            ids: List of unique IDs
            metadata: Optional list of metadata dicts

        Returns:
            Dict with stats
        """
        collection = self.client.get_collection(name=collection_name)

        # Convert embeddings to list of lists
        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = embeddings

        # Add to collection
        start_time = time.time()

        collection.add(
            embeddings=embeddings_list,
            documents=texts,
            ids=ids,
            metadatas=metadata if metadata else None
        )

        elapsed = time.time() - start_time

        return {
            'success': True,
            'count': len(ids),
            'elapsed': elapsed,
            'rate': len(ids) / elapsed if elapsed > 0 else 0
        }

    def query_collection(
        self,
        collection_name: str,
        query_embeddings: Optional[np.ndarray] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict] = None,
        include: List[str] = None
    ) -> Dict[str, Any]:
        """
        Query a collection for similar items

        Args:
            collection_name: Name of collection
            query_embeddings: Query embeddings (if None, must provide query_texts)
            query_texts: Query texts (used if query_embeddings is None)
            n_results: Number of results to return per query
            where: Optional metadata filters
            include: What to include in results ['embeddings', 'documents', 'metadatas', 'distances']

        Returns:
            Query results
        """
        collection = self.client.get_collection(name=collection_name)

        if include is None:
            include = ['documents', 'metadatas', 'distances']

        start_time = time.time()

        if query_embeddings is not None:
            # Convert to list format
            if isinstance(query_embeddings, np.ndarray):
                query_embeddings = query_embeddings.tolist()

            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include
            )
        elif query_texts is not None:
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                include=include
            )
        else:
            raise ValueError("Must provide either query_embeddings or query_texts")

        elapsed = time.time() - start_time

        return {
            'results': results,
            'elapsed': elapsed,
            'n_queries': len(query_embeddings) if query_embeddings else len(query_texts)
        }

    def find_duplicates(
        self,
        collection_name: str,
        threshold: float = 0.85,
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find duplicate entries in a collection

        Args:
            collection_name: Name of collection
            threshold: Similarity threshold (0.0-1.0)
            batch_size: Batch size for queries

        Returns:
            List of duplicate pairs
        """
        collection = self.client.get_collection(name=collection_name)

        # Get all items
        all_items = collection.get(include=['embeddings', 'documents', 'metadatas'])

        if not all_items['ids']:
            return []

        embeddings = np.array(all_items['embeddings'])
        ids = all_items['ids']
        documents = all_items['documents']

        duplicates = []

        # Query each item against the collection
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            batch_embeddings = embeddings[i:batch_end]

            results = collection.query(
                query_embeddings=batch_embeddings.tolist(),
                n_results=5,  # Top 5 similar
                include=['documents', 'distances']
            )

            # Process results
            for j, (result_ids, result_docs, result_distances) in enumerate(
                zip(results['ids'], results['documents'], results['distances'])
            ):
                query_idx = i + j
                query_id = ids[query_idx]
                query_doc = documents[query_idx]

                for k, (res_id, res_doc, distance) in enumerate(
                    zip(result_ids, result_docs, result_distances)
                ):
                    # Skip self-match
                    if res_id == query_id:
                        continue

                    # Convert distance to similarity (ChromaDB uses L2 distance)
                    # For normalized embeddings: similarity ≈ 1 - (distance² / 2)
                    similarity = 1 - (distance * distance / 2)

                    if similarity >= threshold:
                        # Ensure we don't add duplicates (A-B and B-A)
                        pair_key = tuple(sorted([query_id, res_id]))

                        # Check if already added
                        if not any(
                            tuple(sorted([d['id1'], d['id2']])) == pair_key
                            for d in duplicates
                        ):
                            duplicates.append({
                                'id1': query_id,
                                'id2': res_id,
                                'text1': query_doc,
                                'text2': res_doc,
                                'similarity': round(similarity, 3),
                                'distance': round(distance, 3)
                            })

        return duplicates

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection

        Args:
            collection_name: Name of collection

        Returns:
            Statistics dict
        """
        collection = self.client.get_collection(name=collection_name)

        count = collection.count()
        metadata = collection.metadata or {}

        # Sample some items to get embedding dimension
        sample = collection.get(limit=1, include=['embeddings'])
        embedding_dim = len(sample['embeddings'][0]) if sample['embeddings'] is not None and len(sample['embeddings']) > 0 else 0

        # Estimate storage size (rough)
        # Each embedding: embedding_dim * 4 bytes (float32) + overhead
        storage_mb = (count * embedding_dim * 4) / (1024 * 1024)

        return {
            'name': collection_name,
            'count': count,
            'embedding_dimension': embedding_dim,
            'storage_mb': round(storage_mb, 2),
            'metadata': metadata,
            'created_at': metadata.get('created_at', 'Unknown')
        }

    def update_collection(
        self,
        collection_name: str,
        new_embeddings: np.ndarray,
        new_texts: List[str],
        new_ids: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Update collection with new data (incremental)

        Args:
            collection_name: Name of collection
            new_embeddings: New embeddings to add
            new_texts: New texts
            new_ids: New IDs (must be unique)
            metadata: Optional metadata

        Returns:
            Update stats
        """
        # Check for existing IDs
        collection = self.client.get_collection(name=collection_name)
        existing = collection.get(ids=new_ids)

        if existing['ids']:
            return {
                'success': False,
                'error': f'IDs already exist: {existing["ids"]}'
            }

        # Add new data
        return self.add_to_collection(
            collection_name,
            new_embeddings,
            new_texts,
            new_ids,
            metadata
        )

    def export_collection(self, collection_name: str) -> pd.DataFrame:
        """
        Export collection to DataFrame

        Args:
            collection_name: Name of collection

        Returns:
            DataFrame with ids, documents, embeddings, metadata
        """
        collection = self.client.get_collection(name=collection_name)

        data = collection.get(include=['embeddings', 'documents', 'metadatas'])

        df = pd.DataFrame({
            'id': data['ids'],
            'document': data['documents'],
            'embedding': data['embeddings']
        })

        # Add metadata as columns
        if data.get('metadatas'):
            metadata_df = pd.DataFrame(data['metadatas'])
            df = pd.concat([df, metadata_df], axis=1)

        return df

    def get_database_size(self) -> Dict[str, Any]:
        """
        Get total database size and stats

        Returns:
            Database statistics
        """
        collections = self.list_collections()

        total_count = sum(c['count'] for c in collections)
        total_storage = sum(
            (c['count'] * 384 * 4) / (1024 * 1024)  # Assume 384-dim embeddings
            for c in collections
        )

        return {
            'num_collections': len(collections),
            'total_records': total_count,
            'estimated_storage_mb': round(total_storage, 2),
            'persist_directory': str(self.persist_directory),
            'collections': collections
        }
