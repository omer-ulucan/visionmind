"""
Configuration settings for the Visual Reasoning AI System.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Model configurations
MODEL_CONFIG = {
    "text_classifier": {
        "model_name": "microsoft/resnet-50",
        "num_classes": 2,  # text vs non-text
        "pretrained": True
    },
    "ocr": {
        "lang": "eng",
        "config": "--oem 3 --psm 6"
    },
    "object_detection": {
        "model_name": "yolov8x.pt",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45
    },
    "text_embedding": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "max_length": 512
    },
    "object_embedding": {
        "backbone": "resnet50",
        "feature_dim": 2048,
        "embedding_dim": 512
    },
    "context_lstm": {
        "hidden_size": 512,
        "num_layers": 2,
        "dropout": 0.1,
        "bidirectional": True
    },
    "attention": {
        "num_heads": 8,
        "head_dim": 64,
        "dropout": 0.1
    },
    "dlnn": {
        "input_dim": 512,
        "hidden_dims": [256, 128],
        "num_classes": 10,  # Adjust based on reasoning categories
        "dropout": 0.1
    }
}

# Pinecone configuration
PINECONE_CONFIG = {
    "api_key": os.getenv("PINECONE_API_KEY"),
    "environment": os.getenv("PINECONE_ENVIRONMENT"),
    "index_name": os.getenv("PINECONE_INDEX_NAME", "visual-reasoning-ai"),
    "dimension": 512,  # Must match embedding dimensions
    "metric": "cosine"
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 3e-4,
    "num_epochs": 100,
    "early_stopping_patience": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Logging configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
