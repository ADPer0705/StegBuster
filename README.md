# StegBuster - Universal Steganography Detector and Extractor

StegBuster is a powerful command-line tool designed for detecting and extracting hidden data from various types of files. It combines traditional statistical analysis with machine learning approaches to provide robust steganography detection capabilities.

## Overview

Steganography is the practice of hiding secret data within ordinary, non-secret files or messages. Unlike cryptography, which makes data unreadable, steganography hides the existence of the data itself. StegBuster helps security professionals and digital forensics experts to detect such hidden content in various file formats.

## Features

- **Multiple file format support:** Analyzes images, audio files, and text documents
- **Hybrid detection approach:** Combines traditional statistical methods with machine learning
- **Smart ML integration:** Intelligently decides when to use ML versus traditional methods
- **Multiple technique detection:** Identifies various steganography methods including:
  - **Image:** LSB (Least Significant Bit), DCT (Discrete Cosine Transform)
  - **Audio:** LSB, Echo Hiding, Phase Coding
  - **Text:** Whitespace steganography, Zero-width characters, Lexical methods
- **Automatic file type detection:** Uses file signatures and metadata to identify file types
- **Hidden data extraction:** Recovers concealed data when possible
- **Detailed statistical analysis:** Chi-square, RS analysis, and sample pair analysis
- **Visualization:** Generates visualizations highlighting potential areas with hidden data
- **Performance optimization:** Smart detection to balance speed and accuracy

## Installation

### Prerequisites

StegBuster requires Python 3.10-3.12 (for TensorFlow compatibility) and the following libraries:
- OpenCV (cv2)
- NumPy
- SciPy
- Pillow (PIL)
- python-magic
- TensorFlow (for ML functionality)
- Matplotlib (for visualization)

### Using pyenv (Recommended)

[pyenv](https://github.com/pyenv/pyenv) is the recommended way to manage Python versions:

```bash
# Install pyenv (if not already installed)
# On Linux/macOS:
curl https://pyenv.run | bash

# On Windows (with PowerShell):
# Install pyenv-win from https://github.com/pyenv-win/pyenv-win

# Install Python 3.10
pyenv install 3.10.x

# Set Python 3.10 for this project
cd /path/to/stegbuster
pyenv local 3.10.x

# Create and activate virtual environment
python -m venv steg-env
source steg-env/bin/activate  # On Linux/macOS
# or
.\steg-env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Manual Setup

If you prefer to set up manually, you can install the dependencies with:

```bash
# Make sure you have Python 3.10 installed
python3.10 -m venv steg-env
source steg-env/bin/activate  # On Linux/macOS
# or
.\steg-env\Scripts\activate  # On Windows

pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python stegbuster.py [file_to_analyze]
```

### Command-Line Options

#### Basic Options
- `-o, --output FILE` - Save extracted data to specified file
- `-v, --verbose` - Enable detailed output
- `-a, --all-methods` - Try all extraction methods (slower but more thorough)
- `-c, --chi-square` - Perform detailed chi-square analysis
- `-r, --rs-analysis` - Perform detailed RS (Regular-Singular) analysis
- `--detailed-analysis` - Perform comprehensive steganalysis with all methods
- `--json` - Output results in JSON format (for detailed analysis)

#### Machine Learning Options
- `--use-ml` - Use machine learning for enhanced detection
- `--smart-ml` - Use ML intelligently, only when needed for better performance
- `--ml-confidence FLOAT` - Minimum confidence threshold for ML detection (0-1)
- `--visualize` - Generate visualization of detection results
- `--advanced-model` - Use the advanced ML model (larger, but more accurate)

### Examples

```bash
# Basic analysis
python stegbuster.py suspicious_image.png

# Save extracted data to a file
python stegbuster.py hidden_message.wav -o extracted_message.txt

# Verbose output with all detection methods
python stegbuster.py encoded_text.txt -v -a

# Perform detailed statistical analysis
python stegbuster.py steganography_sample.jpg --detailed-analysis --json

# Use smart ML-based detection with visualization
python stegbuster.py suspicious_image.png --smart-ml --visualize

# Use advanced ML model with custom confidence threshold
python stegbuster.py complex_stego.png --use-ml --advanced-model --ml-confidence 0.85
```

## Technical Implementation

### Project Structure

```
stegbuster/
├── stegbuster.py            # Main CLI entry point
├── demo/                    # Sample files for testing
│   └── test.png
├── detectors/               # Traditional detection methods
│   ├── __init__.py
│   ├── detector_factory.py  # Factory pattern for detector selection
│   ├── base_detector.py     # Abstract base detector class
│   ├── image_detector.py    # Image steganography detection
│   ├── audio_detector.py    # Audio steganography detection
│   ├── text_detector.py     # Text steganography detection
│   ├── smart_detector.py    # Hybrid ML/traditional detector
│   └── smart_detector_factory.py
├── ml_detectors/            # Machine learning based detection
│   ├── __init__.py
│   ├── ml_detector_factory.py
│   ├── image_ml_detector.py
│   ├── audio_ml_detector.py
│   └── text_ml_detector.py
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── file_utils.py        # File handling utilities
│   ├── model_manager.py     # ML model management
│   └── statistical_analysis.py # Statistical analysis functions
├── visualizers/             # Visualization tools
│   ├── __init__.py
│   └── image_visualizer.py
└── docs/                    # Documentation
    └── ml_integration_strategy.md
```

### Detection Methods

#### Traditional Statistical Methods

##### Image Steganography

1. **LSB (Least Significant Bit) Detection:**
   - Examines the distribution of the least significant bits of pixel values
   - Uses chi-square attack to detect non-random patterns
   - Sample Pair Analysis (SPA) to detect anomalies in pairs of values
   - RS Analysis to detect pixel value distortions
   - Extracts bits sequentially and attempts to convert to ASCII text

2. **DCT (Discrete Cosine Transform) Detection:**
   - Analyzes DCT coefficients commonly used in JPEG steganography
   - Looks for anomalies in coefficient distribution
   - Extracts LSB data from mid-frequency coefficients

##### Audio Steganography

1. **LSB Audio Detection:**
   - Similar to image LSB, but works with audio samples
   - Analyzes the randomness of LSBs in audio data

2. **Echo Hiding Detection:**
   - Uses autocorrelation to detect hidden echoes
   - Identifies patterns in echo delays that might encode binary data

3. **Phase Coding Detection:**
   - Analyzes phase information from Fourier transform
   - Looks for unusual patterns or clustering in phase values

##### Text Steganography

1. **Whitespace Steganography:**
   - Detects unusual patterns of spaces and tabs
   - Analyzes trailing whitespace which may encode binary data

2. **Zero-Width Character Detection:**
   - Identifies invisible Unicode characters used to hide data
   - Extracts data encoded in these hidden characters

3. **Lexical Steganography:**
   - Analyzes patterns like first letters of each word/line
   - Detects unusual linguistic patterns that might encode data

#### Machine Learning Methods

StegBuster incorporates machine learning for enhanced detection capabilities:

1. **Deep Learning Models:**
   - Pre-trained models for detecting steganography in different file types
   - Basic and advanced models for different performance/accuracy needs

2. **Classification of Steganographic Methods:**
   - ML models that can identify specific steganography techniques used
   - Provides probability estimates for different embedding methods

3. **Smart ML Integration:**
   - Intelligently decides when to use ML vs traditional detection
   - Combines results from both approaches for optimal detection

4. **Visualization:**
   - Generates heatmaps to visualize suspicious areas in images
   - Highlights potential regions containing hidden data

### ML Integration Strategy

StegBuster uses a smart ML integration approach:

1. **Selective Application:**
   - Uses traditional methods for initial screening
   - Applies ML only when necessary to save resources

2. **Confidence-Based Decision Making:**
   - Configurable confidence thresholds for ML detection
   - Falls back to traditional methods for borderline cases

3. **Model Selection:**
   - Basic models for faster analysis
   - Advanced models for higher accuracy when needed

4. **Result Integration:**
   - Combines results from multiple detection methods
   - Weighted voting based on confidence levels

## How It Works

1. **File Type Identification:**
   - The user provides a file to analyze
   - StegBuster identifies the file type using MIME detection

2. **Detector Selection:**
   - The appropriate detector is selected based on file type
   - If ML is enabled, the detector integrates traditional and ML methods

3. **Steganography Detection:**
   - Detection algorithms analyze the file for steganographic signatures
   - Statistical methods examine data patterns
   - ML models analyze the file for learned steganographic patterns
   - Results are combined based on confidence levels

4. **Data Extraction:**
   - If steganography is detected, extraction algorithms attempt to recover hidden data
   - The most appropriate extraction method is selected based on the detected technique

5. **Result Presentation:**
   - Results are displayed to the user, including:
     - Detected steganographic method
     - Confidence level
     - Extracted content (if available)
     - Visualizations (if requested)

## Performance Considerations

- **Hardware Requirements:**
  - ML functionality works best with at least 4GB of RAM
  - GPU acceleration improves ML performance but is not required

- **Speed vs. Accuracy:**
  - Basic detection is fast but may miss sophisticated steganography
  - `--detailed-analysis` provides more thorough detection but is slower
  - `--smart-ml` balances speed and accuracy

- **File Size Limitations:**
  - Very large files may require significant memory
  - Processing time scales with file size

## Advanced Features

### Detailed Statistical Analysis

The `--detailed-analysis` option provides comprehensive statistical metrics:

- **Chi-Square Analysis:**
  - Tests randomness of LSB values
  - Provides p-values indicating steganography likelihood

- **RS Analysis:**
  - Regular-Singular analysis detects changes in pixel patterns
  - Estimates embedding rates and provides confidence metrics

- **JSON Output:**
  - The `--json` option outputs structured data for integration with other tools
  - Includes detailed metrics and confidence values

### Machine Learning Enhancement

Enable advanced ML features:

- **Smart ML (`--smart-ml`):**
  - Intelligently combines traditional and ML-based detection
  - Adapts analysis based on initial results

- **Advanced Models (`--advanced-model`):**
  - More sophisticated ML models for better detection accuracy
  - Requires more computational resources

## Limitations

- False positives may occur with certain file types or detection methods
- Not all steganographic techniques are detectable or extractable
- Some advanced techniques may require specific knowledge about the encoding method
- Binary data extraction may not always be interpretable as text
- ML detection requires sufficient RAM and compatible Python version (3.10-3.12)

## Future Development

Planned enhancements:
- Support for additional file types (video, PDF)
- Integration with network traffic analysis
- Web interface for easier use
- Enhanced ML model training on larger datasets
- Improved visualization tools

## Contributing

Contributions to improve detection algorithms, add support for new file types, or enhance extraction capabilities are welcome. Please follow standard GitHub pull request procedures.

## License

[MIT License](LICENSE)
