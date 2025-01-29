"""
Object Detection Module.
Implements YOLO-based object detection with custom post-processing for embedding generation.
"""
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    """
    YOLO-based object detector with custom post-processing for the visual reasoning system.
    """
    def __init__(self, config: Dict):
        """
        Initialize the object detector.
        
        Args:
            config: Dictionary containing model configuration
                - model_name: Path to YOLO model weights
                - conf_threshold: Confidence threshold for detections
                - iou_threshold: IoU threshold for NMS
        """
        self.config = config
        self.model = YOLO(config['model_name'])
        self.conf_threshold = config['conf_threshold']
        self.iou_threshold = config['iou_threshold']
        
    def detect(self, image: np.ndarray) -> Tuple[List[Dict], torch.Tensor]:
        """
        Perform object detection on an input image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Tuple containing:
                - List of detection dictionaries with keys:
                    - bbox: (x1, y1, x2, y2) coordinates
                    - class_id: Class ID of detected object
                    - confidence: Detection confidence
                    - class_name: String name of detected class
                - Tensor of cropped object features for embedding generation
        """
        # Run YOLO inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        detections = []
        object_crops = []
        
        # Process each detection
        for box in results.boxes:
            bbox = box.xyxy[0].cpu().numpy()  # Get bbox coordinates
            conf = float(box.conf[0])         # Get confidence
            cls_id = int(box.cls[0])          # Get class ID
            
            # Get class name
            cls_name = results.names[cls_id]
            
            # Crop object region
            x1, y1, x2, y2 = map(int, bbox)
            object_crop = image[y1:y2, x1:x2]
            
            detections.append({
                'bbox': bbox.tolist(),
                'class_id': cls_id,
                'confidence': conf,
                'class_name': cls_name
            })
            
            object_crops.append(object_crop)
            
        # Convert crops to tensor if any detections found
        if object_crops:
            object_crops = torch.stack([
                self.preprocess_crop(crop) for crop in object_crops
            ])
        else:
            object_crops = torch.empty(0)
            
        return detections, object_crops
    
    @staticmethod
    def preprocess_crop(crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess cropped object region for feature extraction.
        
        Args:
            crop: Numpy array of cropped image region
            
        Returns:
            Preprocessed tensor ready for feature extraction
        """
        # Convert to float and normalize
        crop = crop.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
        
        return crop
    
    def extract_features(self, crops: torch.Tensor) -> torch.Tensor:
        """
        Extract features from cropped object regions.
        
        Args:
            crops: Tensor of shape (N, C, H, W) containing object crops
            
        Returns:
            Tensor of shape (N, feature_dim) containing object features
        """
        # Use YOLO's backbone as feature extractor
        with torch.no_grad():
            features = self.model.model.backbone(crops)
            
        # Global average pooling
        features = torch.mean(features, dim=[2, 3])
        
        return features
