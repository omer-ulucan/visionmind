"""
Visual Reasoning Pipeline.
Implements the complete pipeline for visual reasoning with memory and context.
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import cv2
import pytesseract
from PIL import Image

from models.text_classifier import TextClassifier
from models.object_detection import ObjectDetector
from models.embedding import MultiModalEmbedder
from models.context_processor import ContextProcessor
from models.reasoning import ReasoningModule
from models.memory import MemoryManager

class VisualReasoningPipeline:
    """
    End-to-end pipeline for visual reasoning with memory and context.
    """
    def __init__(
        self,
        config: Dict,
        device: Optional[str] = None
    ):
        """
        Initialize the visual reasoning pipeline.
        
        Args:
            config: Dictionary containing configurations for all components
            device: Optional torch device to use
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Store configuration
        self.config = config
        
        # Initialize components
        self.text_classifier = TextClassifier(config['MODEL_CONFIG']['text_classifier'])
        self.object_detector = ObjectDetector(config['MODEL_CONFIG']['object_detection'])
        self.embedder = MultiModalEmbedder(
            config['MODEL_CONFIG']['text_embedding'],
            config['MODEL_CONFIG']['object_embedding']
        )
        self.context_processor = ContextProcessor(
            config['MODEL_CONFIG']['context_lstm'],
            config['MODEL_CONFIG']['attention']
        )
        self.reasoning = ReasoningModule(
            config['MODEL_CONFIG']['dlnn'],
            class_names=self._get_reasoning_classes()
        )
        self.memory = MemoryManager(config['PINECONE_CONFIG'])
        
        # Move models to device
        self._to_device()
        
    def _to_device(self):
        """Move all models to specified device."""
        self.text_classifier.to(self.device)
        self.embedder.object_embedder.to(self.device)
        self.context_processor.to(self.device)
        self.reasoning.model.to(self.device)
        
    @staticmethod
    def _get_reasoning_classes() -> List[str]:
        """Define reasoning output classes."""
        return [
            "must_stop",              # Stop signs, red lights, etc.
            "can_proceed",            # Green lights, clear path
            "caution_required",       # Yellow lights, pedestrian crossing
            "direction_change",       # Turn signs, lane merges
            "speed_regulation",       # Speed limit signs
            "pedestrian_present",     # Detected pedestrians
            "vehicle_present",        # Detected vehicles
            "traffic_congestion",     # Multiple vehicles, slow traffic
            "road_condition",         # Weather, construction, etc.
            "emergency_situation"     # Emergency vehicles, accidents
        ]
        
    def process_image(
        self,
        image: Union[str, np.ndarray],
        context_window: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Process a single image through the reasoning pipeline.
        
        Args:
            image: Image path or numpy array
            context_window: Optional list of previous reasoning results
            
        Returns:
            Dictionary containing processing results and reasoning
        """
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Classify image content (text vs non-text)
        has_text = self._classify_content(image)
        
        # Process text and objects
        text_embeddings = None
        if has_text:
            text = pytesseract.image_to_string(
                Image.fromarray(image),
                config=self.config['MODEL_CONFIG']['ocr']['config']
            ).strip()
            text_embeddings = self.embedder.embed_text(text)
            
        # Detect and embed objects
        detections, object_crops = self.object_detector.detect(image)
        if len(object_crops) > 0:
            object_features = self.object_detector.extract_features(object_crops)
            object_embeddings = self.embedder.embed_objects(object_features)
            
        # Combine embeddings
        combined_embeddings = self._combine_embeddings(
            text_embeddings, object_embeddings if 'object_embeddings' in locals() else None
        )
        
        # Process context
        if context_window:
            # Retrieve relevant memories
            memories = self._retrieve_relevant_memories(combined_embeddings)
            
            # Process through context LSTM
            processed_embeddings, hidden_state = self.context_processor(
                combined_embeddings.unsqueeze(0),
                torch.stack([m['embedding'] for m in memories]).unsqueeze(0)
                if memories else None
            )
        else:
            processed_embeddings = combined_embeddings.unsqueeze(0)
            memories = None
            
        # Perform reasoning
        reasoning_results, evidence = self.reasoning.reason(
            processed_embeddings,
            context_window[-1]['evidence'] if context_window else None
        )
        
        # Store results in memory
        result_id = self.memory.store(
            combined_embeddings.unsqueeze(0),
            [{
                'reasoning': reasoning_results[0],
                'detections': detections,
                'text_present': has_text
            }]
        )[0]
        
        return {
            'id': result_id,
            'reasoning': reasoning_results[0],
            'detections': detections,
            'text_present': has_text,
            'evidence': evidence,
            'memories_used': memories
        }
        
    def _classify_content(self, image: np.ndarray) -> bool:
        """Classify if image contains text."""
        # Preprocess image for text classifier
        input_tensor = self._preprocess_image(image)
        
        # Get prediction
        with torch.no_grad():
            pred, _ = self.text_classifier.predict(input_tensor)
            
        return bool(pred.item())
        
    @staticmethod
    def _preprocess_image(image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize and normalize
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return tensor
        
    def _combine_embeddings(
        self,
        text_embeddings: Optional[torch.Tensor],
        object_embeddings: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Combine text and object embeddings."""
        embeddings_list = []
        
        if text_embeddings is not None:
            embeddings_list.append(text_embeddings)
            
        if object_embeddings is not None:
            embeddings_list.append(object_embeddings)
            
        if not embeddings_list:
            raise ValueError("No embeddings to combine")
            
        # Concatenate along sequence dimension
        return torch.cat(embeddings_list, dim=0)
        
    def _retrieve_relevant_memories(
        self,
        current_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve relevant memories for current context."""
        metadata, scores = self.memory.retrieve(
            current_embedding.mean(dim=0),
            top_k=top_k
        )
        
        return [
            {**meta, 'similarity': score}
            for meta, score in zip(metadata, scores)
        ]
