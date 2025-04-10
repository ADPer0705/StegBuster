# StegBuster Developer Guide

This guide is designed to help developers understand StegBuster's architecture and contribute to the project effectively.

## Codebase Architecture

StegBuster follows a modular design with clear separation of concerns:

```
stegbuster/
├── stegbuster.py            # Main entry point and CLI interface
├── detectors/               # Traditional detection methods
│   ├── base_detector.py     # Abstract base class for detectors
│   ├── detector_factory.py  # Factory for instantiating correct detectors
│   ├── image_detector.py    # Image steganography detection
│   ├── audio_detector.py    # Audio steganography detection
│   ├── text_detector.py     # Text steganography detection
│   ├── smart_detector.py    # Hybrid ML/traditional detector
│   └── smart_detector_factory.py
├── ml_detectors/            # Machine learning based detection
│   ├── ml_detector_factory.py
│   ├── image_ml_detector.py
│   ├── audio_ml_detector.py
│   └── text_ml_detector.py
├── utils/                   # Utility functions
│   ├── file_utils.py        # File handling utilities
│   ├── model_manager.py     # ML model management
│   └── statistical_analysis.py # Statistical analysis functions
└── visualizers/             # Visualization tools
    └── image_visualizer.py
```

### Key Design Patterns

1. **Factory Pattern**: Used to create the appropriate detector for each file type
2. **Strategy Pattern**: Different detection and extraction strategies encapsulated in separate classes
3. **Abstract Base Classes**: Define interfaces that concrete implementations must follow
4. **Dependency Injection**: Components receive dependencies rather than creating them

## Getting Started with Development

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stegbuster.git
cd stegbuster
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows
```

3. Install dependencies in development mode:
```bash
pip install -e ".[dev]"
# or
pip install -r requirements-dev.txt
```

### Running Tests

We use pytest for testing:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_image_detector.py

# Run with coverage report
pytest --cov=stegbuster tests/
```

### Code Style

We follow PEP 8 guidelines. Use flake8 and black to check and format your code:

```bash
# Check code style
flake8 stegbuster

# Format code
black stegbuster
```

## Core Components

### Base Detector

The `BaseSteganographyDetector` class in `detectors/base_detector.py` defines the interface that all detectors must implement:

```python
from abc import ABC, abstractmethod

class BaseSteganographyDetector(ABC):
    @abstractmethod
    def detect(self, file_path):
        """
        Detect if steganography is present in the file
        Returns the method used if detected, otherwise None
        """
        pass
    
    @abstractmethod
    def extract(self, file_path, method=None):
        """
        Extract hidden data from the file
        Returns the extracted data if successful, otherwise None
        """
        pass
```

When implementing a new detector, you must extend this class and implement these methods.

### Detector Factory

The factory pattern is used to instantiate the appropriate detector based on file type:

```python
def get_detector(file_type):
    """Factory function to get the appropriate detector based on file type"""
    if file_type == 'image':
        return ImageSteganographyDetector()
    elif file_type == 'audio':
        return AudioSteganographyDetector()
    elif file_type == 'text':
        return TextSteganographyDetector()
    else:
        return None
```

### ML Integration

The ML detectors follow a similar pattern but include additional methods for working with machine learning models:

```python
class ImageMLDetector:
    def __init__(self):
        self.model = None
        self.model_name = "image_basic"
        self.confidence_threshold = 0.7
    
    def load_model(self):
        """Load the ML model"""
        # Implementation details
    
    def detect(self, file_path, visualize=False):
        """Detect using ML and return (is_stego_likely, confidence, method, viz_path)"""
        # Implementation details
```

## Adding a New Detector

### Example: Adding a New File Type Detector

1. Create a new detector file (e.g., `pdf_detector.py`):

```python
from detectors.base_detector import BaseSteganographyDetector

class PDFSteganographyDetector(BaseSteganographyDetector):
    """Detector for steganography in PDF files"""
    
    def detect(self, file_path):
        # Implement detection logic
        # Return the detected method or None
        pass
    
    def extract(self, file_path, method=None):
        # Implement extraction logic
        # Return the extracted data or None
        pass
```

2. Update the detector factory:

```python
# In detector_factory.py
from detectors.pdf_detector import PDFSteganographyDetector

def get_detector(file_type):
    # ...existing code...
    elif file_type == 'pdf':
        return PDFSteganographyDetector()
    # ...existing code...
```

3. Update file type identification (in `utils/file_utils.py`):

```python
def identify_file_type(file_path):
    # ...existing code...
    elif mime_type.startswith('application/pdf'):
        return 'pdf'
    # ...existing code...
```

### Adding a New Detection Method

To add a new detection method to an existing detector:

1. Add the method to the appropriate detector class:

```python
def _detect_new_method(self, file_path):
    """Detect using your new method"""
    # Implementation
    return is_stego_detected

def _extract_new_method(self, file_path):
    """Extract data using your new method"""
    # Implementation
    return extracted_data
```

2. Update the main detect/extract methods:

```python
def detect(self, file_path):
    # ...existing code...
    if self._detect_new_method(file_path):
        return "NEW_METHOD"
    # ...existing code...

def extract(self, file_path, method=None):
    # ...existing code...
    if method == "NEW_METHOD" or not method:
        data = self._extract_new_method(file_path)
        if data:
            return data
    # ...existing code...
```

## Working with ML Models

### Adding a New ML Model

1. Define the model in the model manager:

```python
# In utils/model_manager.py
MODEL_REGISTRY = {
    # ...existing code...
    'new_model': {
        'url': 'https://yourserver.com/models/new_model.tflite',
        'hash': '0123456789abcdef0123456789abcdef',
        'size': '15MB'
    }
}
```

2. Add preprocessing function if needed:

```python
def preprocess_for_new_model(file_path):
    """Preprocess data for the new model"""
    # Implementation
    return processed_data
```

3. Update the ML detector to use your new model:

```python
# In ml_detectors/image_ml_detector.py
def set_model(self, model_name):
    """Set which model to use"""
    self.model_name = model_name
    # Reset model to force reloading
    self.model = None
```

## Statistical Analysis Implementation

When implementing new statistical methods:

1. Add the method to `utils/statistical_analysis.py`:

```python
def new_statistical_method(data, **kwargs):
    """
    Implement new statistical analysis method
    
    Args:
        data: Input data array
        **kwargs: Additional parameters
        
    Returns:
        tuple: (statistic_value, p_value, is_significant)
    """
    # Implementation
    return stat_value, p_value, is_significant
```

2. Use the method in your detector:

```python
from utils.statistical_analysis import new_statistical_method

def _detect_using_stats(self, file_path):
    # Load data
    data = load_data(file_path)
    
    # Apply statistical method
    stat, pval, is_significant = new_statistical_method(data)
    
    return is_significant
```

## Optimization Tips

1. **Memory Management**: Process large files in chunks
2. **Caching**: Cache intermediate results for repeated operations
3. **Parallelization**: Use multiprocessing for computationally intensive tasks
4. **Early Termination**: Exit early if detection is confident enough

Example of chunk processing:

```python
def process_large_file(file_path, chunk_size=1024*1024):
    """Process a large file in chunks to save memory"""
    results = []
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            # Process chunk
            chunk_result = analyze_chunk(chunk)
            results.append(chunk_result)
    
    # Combine results
    return combine_results(results)
```

## Error Handling and Logging

Follow these guidelines for error handling:

1. Use specific exceptions:
```python
class SteganographyDetectionError(Exception):
    """Base exception for detection errors"""
    pass

class UnsupportedFileTypeError(SteganographyDetectionError):
    """Raised when file type is not supported"""
    pass
```

2. Implement proper logging:
```python
import logging

# Setup in main module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Usage in modules
logger = logging.getLogger(__name__)
logger.debug("Processing file: %s", file_path)
logger.error("Failed to load model: %s", str(e))
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Implement your changes following the code style guidelines
3. Add tests for new functionality
4. Update documentation as necessary
5. Submit a pull request with a clear description of the changes
6. Respond to code review feedback

## Code Review Checklist

When reviewing code (yours or others'), check:

- [ ] Does it follow PEP 8 style guidelines?
- [ ] Are there appropriate tests?
- [ ] Is documentation updated?
- [ ] Is error handling appropriate?
- [ ] Are there performance concerns?
- [ ] Is the code maintainable?

## Advanced Topics

### Adding Support for GPU Acceleration

```python
def load_model_with_gpu():
    """Load model with GPU acceleration if available"""
    import tensorflow as tf
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # GPU is available
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("Using GPU acceleration")
    else:
        logger.info("No GPU available, using CPU")
    
    # Load model
    # ... implementation ...
```

### Implementing Custom Visualizers

```python
def create_custom_visualization(file_path, results, output_path):
    """Create custom visualization of analysis results"""
    # Implementation using matplotlib, plotly, etc.
    # Save visualization to output_path
    pass
```

## Performance Benchmarking

When optimizing, benchmark your changes:

```python
import time

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function's execution time"""
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    print(f"{func.__name__} took {elapsed:.4f} seconds")
    return result, elapsed
```

## Contributing Documentation

When updating documentation:

1. Update the relevant Markdown file in the `docs/` directory
2. Keep the README.md in sync with detailed documentation
3. Include code examples when appropriate
4. Add docstrings to all functions and classes

## Project Roadmap

Current development priorities:

1. Add support for more file types (PDF, video)
2. Improve ML model accuracy with expanded training data
3. Implement heuristic combination of multiple detection methods
4. Create a web interface for easier use
5. Optimize performance for larger files

Join our development discussions to help shape the future of StegBuster!