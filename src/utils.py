"""
Utility functions for the Visual Reasoning AI System.
"""
import os
import logging
from typing import Dict, Optional
import torch
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

def setup_logging(config: Dict, name: str = "visual_reasoning") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Logging configuration dictionary
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(config['level'])
    
    # Create formatter
    formatter = logging.Formatter(config['format'])
    
    # Create and configure handlers
    handlers = [
        logging.StreamHandler(),  # Console handler
        logging.FileHandler(      # File handler
            os.path.join('logs', f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        )
    ]
    
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def load_image(
    path: str,
    target_size: Optional[tuple] = None,
    rgb: bool = True
) -> np.ndarray:
    """
    Load and preprocess an image.
    
    Args:
        path: Path to image file
        target_size: Optional tuple of (height, width) for resizing
        rgb: Whether to convert to RGB color space
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
        
    # Convert color space if needed
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # Resize if target size provided
    if target_size:
        image = cv2.resize(image, target_size[::-1])  # OpenCV uses (width, height)
        
    return image

def save_model(
    model: torch.nn.Module,
    path: str,
    metadata: Optional[Dict] = None
):
    """
    Save model weights and metadata.
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
        metadata: Optional dictionary of metadata to save with model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to disk
    torch.save(save_dict, path)

def load_model(
    model: torch.nn.Module,
    path: str
) -> Dict:
    """
    Load model weights and metadata.
    
    Args:
        model: PyTorch model to load weights into
        path: Path to saved model file
        
    Returns:
        Dictionary containing model metadata
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
        
    # Load save dictionary
    save_dict = torch.load(path)
    
    # Load state dict into model
    model.load_state_dict(save_dict['model_state_dict'])
    
    return save_dict['metadata']

def create_attention_visualization(
    image: np.ndarray,
    attention_weights: np.ndarray,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Create visualization of attention weights overlaid on image.
    
    Args:
        image: Input image
        attention_weights: Attention weight matrix
        alpha: Transparency factor for overlay
        
    Returns:
        Visualization image with attention overlay
    """
    # Normalize attention weights to 0-1
    attention = attention_weights - attention_weights.min()
    attention = attention / attention.max()
    
    # Resize to match image dimensions
    attention = cv2.resize(
        attention,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Create heatmap
    heatmap = cv2.applyColorMap(
        (attention * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Overlay heatmap on image
    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )
    
    return overlay

def format_reasoning_output(reasoning_dict: Dict) -> str:
    """
    Format reasoning output for display.
    
    Args:
        reasoning_dict: Dictionary containing reasoning results
        
    Returns:
        Formatted string representation
    """
    output = []
    
    # Add prediction and confidence
    output.append(f"Prediction: {reasoning_dict['prediction']}")
    output.append(f"Confidence: {reasoning_dict['confidence']:.2%}")
    
    # Add evidence information
    if 'evidence_indices' in reasoning_dict:
        output.append("\nEvidence:")
        for idx, weight in zip(
            reasoning_dict['evidence_indices'],
            reasoning_dict['attention_weights']
        ):
            output.append(f"- Evidence {idx}: {weight:.2%} attention")
            
    return "\n".join(output)

def ensure_directory_exists(path: str):
    """
    Ensure all directories in path exist, creating them if necessary.
    
    Args:
        path: Directory path to ensure exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)
