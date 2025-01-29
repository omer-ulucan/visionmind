"""
Embedding Generation Module.
Handles generation of embeddings for both text and object features.
"""
from typing import Dict, List, Union
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torchvision.models as models

class TextEmbedder:
    """
    Generates embeddings for text using Sentence Transformers.
    """
    def __init__(self, config: Dict):
        """
        Initialize the text embedder.
        
        Args:
            config: Dictionary containing model configuration
                - model_name: Name of the sentence transformer model
                - max_length: Maximum sequence length
        """
        self.model = SentenceTransformer(config['model_name'])
        self.max_length = config['max_length']
        
    def embed(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Tensor of shape (N, embedding_dim) containing text embeddings
        """
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]
            
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            max_length=self.max_length,
            normalize_embeddings=True,
            convert_to_tensor=True
        )
        
        return embeddings

class ObjectEmbedder(nn.Module):
    """
    Generates embeddings for object features using a CNN backbone.
    """
    def __init__(self, config: Dict):
        """
        Initialize the object embedder.
        
        Args:
            config: Dictionary containing model configuration
                - backbone: Name of the CNN backbone
                - feature_dim: Input feature dimension
                - embedding_dim: Output embedding dimension
        """
        super(ObjectEmbedder, self).__init__()
        
        # Feature projection network
        self.projection = nn.Sequential(
            nn.Linear(config['feature_dim'], 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, config['embedding_dim']),
            nn.LayerNorm(config['embedding_dim'])
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate embeddings from object features.
        
        Args:
            features: Tensor of shape (N, feature_dim) containing object features
            
        Returns:
            Tensor of shape (N, embedding_dim) containing object embeddings
        """
        return self.projection(features)
    
    def embed(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate normalized embeddings from object features.
        
        Args:
            features: Tensor of shape (N, feature_dim) containing object features
            
        Returns:
            Tensor of shape (N, embedding_dim) containing normalized embeddings
        """
        with torch.no_grad():
            embeddings = self.forward(features)
            # L2 normalize embeddings
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

class MultiModalEmbedder:
    """
    Combines text and object embedders for unified embedding generation.
    """
    def __init__(self, text_config: Dict, object_config: Dict):
        """
        Initialize the multimodal embedder.
        
        Args:
            text_config: Configuration for text embedder
            object_config: Configuration for object embedder
        """
        self.text_embedder = TextEmbedder(text_config)
        self.object_embedder = ObjectEmbedder(object_config)
        
    def embed_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Generate embeddings for text input."""
        return self.text_embedder.embed(texts)
    
    def embed_objects(self, features: torch.Tensor) -> torch.Tensor:
        """Generate embeddings for object features."""
        return self.object_embedder.embed(features)
    
    def save_object_embedder(self, path: str):
        """Save object embedder weights to disk."""
        torch.save(self.object_embedder.state_dict(), path)
    
    def load_object_embedder(self, path: str):
        """Load object embedder weights from disk."""
        self.object_embedder.load_state_dict(torch.load(path))
