"""
Text-or-Non-Text Classifier Module.
This module implements a CNN-based classifier to determine if an image contains text.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple

class TextClassifier(nn.Module):
    """
    A CNN-based classifier that determines whether an image contains text or not.
    Uses a pre-trained ResNet backbone with custom classification head.
    """
    def __init__(self, config: Dict):
        """
        Initialize the text classifier.
        
        Args:
            config: Dictionary containing model configuration
                - model_name: Name of the pretrained model to use
                - num_classes: Number of output classes (2 for text/non-text)
                - pretrained: Whether to use pretrained weights
        """
        super(TextClassifier, self).__init__()
        
        # Load pretrained backbone
        self.backbone = getattr(models, config['model_name'].split('/')[-1])(
            pretrained=config['pretrained']
        )
        
        # Replace the final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config['num_classes'])
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size, num_classes) containing class logits
        """
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on input images.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple containing:
                - Predicted class indices
                - Class probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            return preds, probs

    def save(self, path: str):
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path))
