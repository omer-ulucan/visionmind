﻿# Visual Reasoning AI System

A prototype AI system that mimics human visual perception and reasoning, inspired by how the human brain processes visual and contextual information. The system analyzes images containing text and objects, reasons about them sequentially, and maintains a memory of past decisions for contextual processing.

## Future Updates: Towards Project NeuroSynth – The Artificial Brain

**VisionMind** is an early prototype focused on **visual perception and reasoning**, but this is only a small step in a much larger journey.

Our ultimate goal is to **fully replicate the human brain**, not just vision but **all cognitive processes**, leading to the creation of Project**NeuroSynth**, an artificial brain capable of human-like perception, memory, and reasoning.

### What's Next?

- **Complete Brain Simulation** – Expanding beyond vision to incorporate **auditory processing, motor control, abstract thinking, and decision-making**.
- **Dynamic Learning & Memory** – Enabling AI to **continuously learn, adapt to new experiences, and refine its reasoning** over time.
- **Symbolic & Neural Hybrid AI** – Combining **deep learning and logical reasoning** to create a system that not only detects patterns but also **understands and explains them**.
- **General AI Framework** – A modular architecture where different cognitive abilities (**vision, reasoning, memory, problem-solving**) work **together like a real brain**.

**VisionMind is just the first experiment.** Future iterations will push the boundaries of AI, leading to a system that **doesn’t just see—it thinks, understands, and interacts like a human.**

## Architecture Overview

The system implements a sophisticated pipeline that processes visual information through several key stages:

1. **Input Processing**

   - Text/Object Classification
   - OCR for text extraction
   - YOLO-based object detection

2. **Embedding Generation**

   - Text embeddings via Sentence Transformers
   - Object feature embeddings via CNN
   - Multimodal embedding fusion

3. **Contextual Processing**

   - Bidirectional LSTM for sequential processing
   - Multi-head attention mechanism
   - Memory retrieval and integration

4. **Reasoning Engine**

   - Dynamic Logical Neural Network (DLNN)
   - Evidence combination through learned gates
   - Explainable reasoning outputs

5. **Memory Management**
   - Vector storage in Pinecone
   - Temporal and similarity-based retrieval
   - Continuous learning through memory updates

## Key Features

- **Multimodal Processing**: Handles both text and visual information
- **Contextual Reasoning**: Considers past decisions and current context
- **Memory Integration**: Stores and retrieves relevant past experiences
- **Explainable Decisions**: Provides reasoning paths and confidence scores
- **Continuous Learning**: Updates knowledge base through vector storage

## Requirements

> ⚠️ **Note:** The model training has not been conducted yet due to resource constraints. The pipeline and processing components are functional, but training and fine-tuning will be completed in future updates.

```
# Core ML/DL Libraries
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
sentence-transformers>=2.2.2

# Computer Vision & OCR
opencv-python>=4.7.0
pytesseract>=0.3.10
ultralytics>=8.0.0  # For YOLO

# Vector Database
pinecone-client>=2.2.2

# Data Processing
numpy>=1.24.0
pandas>=2.0.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0
tqdm>=4.65.0
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/visual-reasoning-ai.git
cd visual-reasoning-ai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
# Create .env file with your Pinecone credentials
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=visual-reasoning-ai
```

## Usage

```python
from src.pipeline import VisualReasoningPipeline
from src.config import MODEL_CONFIG, PINECONE_CONFIG

# Initialize pipeline
pipeline = VisualReasoningPipeline({
    'MODEL_CONFIG': MODEL_CONFIG,
    'PINECONE_CONFIG': PINECONE_CONFIG
})

# Process single image
result = pipeline.process_image(
    'path/to/image.jpg',
    context_window=None  # Optional: previous reasoning results
)

# Access reasoning results
print(f"Reasoning: {result['reasoning']}")
print(f"Detections: {result['detections']}")
```

## Example Scenarios

1. **Traffic Scene Analysis**

   ```python
   # Process traffic scene
   result = pipeline.process_image('traffic_scene.jpg')

   # Example output
   {
       'reasoning': {
           'prediction': 'must_stop',
           'confidence': 0.95,
           'evidence_indices': [0, 2],
           'attention_weights': [0.8, 0.2]
       },
       'detections': [
           {'class_name': 'stop_sign', 'confidence': 0.98},
           {'class_name': 'pedestrian', 'confidence': 0.85}
       ]
   }
   ```

2. **Sequential Processing**
   ```python
   # Process sequence of images with context
   context = []
   for image_path in image_sequence:
       result = pipeline.process_image(image_path, context_window=context)
       context.append(result)
   ```

## Project Structure

```
visual-reasoning-ai/
├── src/
│   ├── config.py           # Configuration settings
│   ├── pipeline.py         # Main pipeline implementation
│   └── models/
│       ├── text_classifier.py     # Text classification model
│       ├── object_detection.py    # Object detection model
│       ├── embedding.py           # Embedding generation
│       ├── context_processor.py   # LSTM and attention
│       ├── reasoning.py           # DLNN implementation
│       └── memory.py             # Memory management
├── data/                  # Data directory
├── models/               # Saved model weights
├── logs/                # Training logs
└── requirements.txt     # Dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{ulucan2025visionmind,
  author       = {Omer Ulucan},
  title        = {VisionMind: Visual Reasoning AI System},
  year         = {2025},
  howpublished = {GitHub Repository},
  url          = {https://github.com/omer-ulucan/visionmind},
  note         = {AI system for visual perception and reasoning. Model training pending due to resource limitations.}
}
```
