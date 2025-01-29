"""
Dynamic Logical Neural Network (DLNN) Module.
Implements reasoning capabilities based on processed embeddings.
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicLogicalNetwork(nn.Module):
    """
    Neural network for logical reasoning over processed embeddings.
    """
    def __init__(self, config: Dict):
        """
        Initialize the DLNN.
        
        Args:
            config: Dictionary containing model configuration
                - input_dim: Input embedding dimension
                - hidden_dims: List of hidden layer dimensions
                - num_classes: Number of reasoning output classes
                - dropout: Dropout probability
        """
        super(DynamicLogicalNetwork, self).__init__()
        
        # Build dynamic layers
        layers = []
        prev_dim = config['input_dim']
        
        for hidden_dim in config['hidden_dims']:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout']),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Reasoning head
        self.reasoning_head = nn.Linear(prev_dim, config['num_classes'])
        
        # Logical gates for combining evidence
        self.and_gate = nn.Parameter(torch.ones(config['num_classes']))
        self.or_gate = nn.Parameter(torch.ones(config['num_classes']))
        
    def forward(
        self,
        embeddings: torch.Tensor,
        prior_evidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform reasoning over embeddings.
        
        Args:
            embeddings: Input embeddings of shape (batch_size, seq_len, input_dim)
            prior_evidence: Optional tensor of prior reasoning results
            
        Returns:
            Tuple containing:
                - Reasoning logits
                - Attention weights over input embeddings
        """
        batch_size, seq_len, _ = embeddings.size()
        
        # Extract features
        features = self.feature_extractor(embeddings)
        
        # Self-attention over sequence
        attention_weights = torch.matmul(
            features, features.transpose(-2, -1)
        ) / torch.sqrt(torch.tensor(features.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Apply attention
        attended_features = torch.matmul(attention_weights, features)
        
        # Generate reasoning logits
        logits = self.reasoning_head(attended_features)
        
        # Apply logical operations if prior evidence exists
        if prior_evidence is not None:
            # AND operation (both current and prior evidence must be strong)
            and_evidence = torch.min(
                F.sigmoid(logits),
                F.sigmoid(prior_evidence).unsqueeze(1).expand_as(logits)
            ) * self.and_gate.view(1, 1, -1)
            
            # OR operation (either current or prior evidence is sufficient)
            or_evidence = torch.max(
                F.sigmoid(logits),
                F.sigmoid(prior_evidence).unsqueeze(1).expand_as(logits)
            ) * self.or_gate.view(1, 1, -1)
            
            # Combine evidence types
            logits = torch.log(
                (and_evidence + or_evidence) / 2 + 1e-10
            )  # Prevent log(0)
            
        return logits, attention_weights
    
    def interpret_reasoning(
        self,
        logits: torch.Tensor,
        attention_weights: torch.Tensor,
        embeddings: torch.Tensor,
        class_names: List[str]
    ) -> List[Dict]:
        """
        Interpret the reasoning process and generate explanations.
        
        Args:
            logits: Reasoning logits
            attention_weights: Attention weights over input
            embeddings: Original input embeddings
            class_names: List of reasoning class names
            
        Returns:
            List of dictionaries containing reasoning explanations
        """
        batch_size = logits.size(0)
        explanations = []
        
        for b in range(batch_size):
            # Get predictions and confidences
            probs = F.softmax(logits[b], dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
            # Get attention focus for each prediction
            batch_attention = attention_weights[b]
            
            # Generate explanations
            batch_explanations = []
            for t in range(predictions.size(0)):
                pred_idx = predictions[t].item()
                confidence = probs[t, pred_idx].item()
                
                # Get most attended elements
                attention_scores = batch_attention[t]
                top_k = min(3, attention_scores.size(0))
                top_attention_idx = torch.topk(
                    attention_scores, k=top_k
                ).indices.tolist()
                
                explanation = {
                    'prediction': class_names[pred_idx],
                    'confidence': confidence,
                    'evidence_indices': top_attention_idx,
                    'attention_weights': attention_scores[top_attention_idx].tolist()
                }
                batch_explanations.append(explanation)
                
            explanations.append(batch_explanations)
            
        return explanations

class ReasoningModule:
    """
    High-level interface for the reasoning system.
    """
    def __init__(
        self,
        dlnn_config: Dict,
        class_names: List[str]
    ):
        """
        Initialize the reasoning module.
        
        Args:
            dlnn_config: Configuration for the DLNN
            class_names: List of reasoning class names
        """
        self.model = DynamicLogicalNetwork(dlnn_config)
        self.class_names = class_names
        
    def reason(
        self,
        embeddings: torch.Tensor,
        prior_evidence: Optional[torch.Tensor] = None
    ) -> Tuple[List[Dict], torch.Tensor]:
        """
        Perform reasoning and generate explanations.
        
        Args:
            embeddings: Input embeddings
            prior_evidence: Optional prior reasoning results
            
        Returns:
            Tuple containing:
                - List of reasoning explanations
                - Updated evidence tensor for future reasoning
        """
        # Get reasoning outputs
        logits, attention = self.model(embeddings, prior_evidence)
        
        # Generate explanations
        explanations = self.model.interpret_reasoning(
            logits, attention, embeddings, self.class_names
        )
        
        # Return explanations and evidence for future reasoning
        return explanations, logits.detach()
    
    def save(self, path: str):
        """Save model weights to disk."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights from disk."""
        self.model.load_state_dict(torch.load(path))
