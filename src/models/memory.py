"""
Memory Management Module.
Handles storage and retrieval of embeddings and reasoning results using Pinecone.
"""
from typing import Dict, List, Optional, Tuple, Union
import pinecone
import torch
import json
import time
from datetime import datetime

class MemoryManager:
    """
    Manages interaction with Pinecone vector database for storing and retrieving embeddings.
    """
    def __init__(self, config: Dict):
        """
        Initialize the memory manager.
        
        Args:
            config: Dictionary containing Pinecone configuration
                - api_key: Pinecone API key
                - environment: Pinecone environment
                - index_name: Name of the Pinecone index
                - dimension: Dimension of embeddings
                - metric: Distance metric to use
        """
        # Initialize Pinecone
        pinecone.init(
            api_key=config['api_key'],
            environment=config['environment']
        )
        
        # Create index if it doesn't exist
        if config['index_name'] not in pinecone.list_indexes():
            pinecone.create_index(
                name=config['index_name'],
                dimension=config['dimension'],
                metric=config['metric']
            )
            
        self.index = pinecone.Index(config['index_name'])
        
    def store(
        self,
        embeddings: torch.Tensor,
        metadata: List[Dict],
        namespace: Optional[str] = None
    ) -> List[str]:
        """
        Store embeddings and metadata in Pinecone.
        
        Args:
            embeddings: Tensor of embeddings to store
            metadata: List of metadata dictionaries for each embedding
            namespace: Optional namespace for the vectors
            
        Returns:
            List of IDs for stored vectors
        """
        # Generate unique IDs based on timestamp
        timestamp = int(time.time() * 1000)
        ids = [f"{timestamp}_{i}" for i in range(len(embeddings))]
        
        # Convert embeddings to list format
        vectors = embeddings.detach().cpu().numpy().tolist()
        
        # Add timestamp to metadata
        for meta in metadata:
            meta['timestamp'] = datetime.fromtimestamp(timestamp/1000).isoformat()
        
        # Upsert to Pinecone
        self.index.upsert(
            vectors=list(zip(ids, vectors, metadata)),
            namespace=namespace
        )
        
        return ids
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        namespace: Optional[str] = None,
        top_k: int = 10,
        filter: Optional[Dict] = None
    ) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve similar vectors and metadata from Pinecone.
        
        Args:
            query_embedding: Query embedding tensor
            namespace: Optional namespace to search in
            top_k: Number of results to return
            filter: Optional metadata filters
            
        Returns:
            Tuple containing:
                - List of metadata dictionaries
                - List of similarity scores
        """
        # Convert query to list
        query = query_embedding.detach().cpu().numpy().tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query,
            namespace=namespace,
            top_k=top_k,
            filter=filter,
            include_metadata=True
        )
        
        # Extract metadata and scores
        metadata = [match.metadata for match in results.matches]
        scores = [float(match.score) for match in results.matches]
        
        return metadata, scores
    
    def retrieve_by_time(
        self,
        start_time: str,
        end_time: str,
        namespace: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve vectors by time range.
        
        Args:
            start_time: ISO format start time
            end_time: ISO format end time
            namespace: Optional namespace to search in
            
        Returns:
            List of metadata dictionaries
        """
        # Create time range filter
        time_filter = {
            'timestamp': {
                '$gte': start_time,
                '$lte': end_time
            }
        }
        
        # Fetch vectors (limit to 10000 to prevent memory issues)
        results = self.index.query(
            vector=[0] * self.index.describe_index_stats().dimension,
            namespace=namespace,
            filter=time_filter,
            top_k=10000,
            include_metadata=True
        )
        
        return [match.metadata for match in results.matches]
    
    def delete_old(
        self,
        before_time: str,
        namespace: Optional[str] = None
    ):
        """
        Delete vectors older than specified time.
        
        Args:
            before_time: ISO format cutoff time
            namespace: Optional namespace to delete from
        """
        # Create time filter
        time_filter = {
            'timestamp': {
                '$lt': before_time
            }
        }
        
        # Delete matching vectors
        self.index.delete(
            filter=time_filter,
            namespace=namespace
        )
    
    def clear(self, namespace: Optional[str] = None):
        """
        Clear all vectors from specified namespace.
        
        Args:
            namespace: Optional namespace to clear
        """
        self.index.delete(delete_all=True, namespace=namespace)
