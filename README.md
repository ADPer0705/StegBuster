# StegBuster - Universal Steganography Extractor

StegBuster is a command-line tool designed for detecting and extracting hidden data from various types of files. It analyzes files for steganographic content, identifies the steganographic technique used, and attempts to extract the hidden messages.

## Overview

Steganography is the practice of hiding secret data within ordinary, non-secret files or messages. Unlike cryptography, which makes data unreadable, steganography hides the existence of the data itself. StegBuster helps security professionals and digital forensics experts to detect such hidden content in various file formats.

## Features

- **Multiple file format support:** Analyzes images, audio files, and text documents
- **Multiple technique detection:** Identifies various steganography methods including:
  - **Image:** LSB (Least Significant Bit), DCT (Discrete Cosine Transform)
  - **Audio:** LSB, Echo Hiding, Phase Coding
  - **Text:** Whitespace steganography, Zero-width characters, Lexical methods
- **Automatic file type detection:** Uses file signatures and metadata to identify file types
- **Hidden data extraction:** Recovers concealed data when possible

## Installation

### Prerequisites

StegBuster requires Python 3.6+ and the following libraries:
- OpenCV (cv2)
- NumPy
- SciPy
- Pillow (PIL)
- python-magic

You can install the required dependencies with:

```bash
pip install numpy scipy pillow opencv-python python-magic
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/stegbuster.git
cd stegbuster
```

## Usage

Basic usage:

```bash
python stegbuster.py [file_to_analyze]
```

### Options

- `-o, --output FILE` - Save extracted data to specified file
- `-v, --verbose` - Enable detailed output
- `-a, --all-methods` - Try all extraction methods (slower but more thorough)

### Examples

```bash
# Basic analysis
python stegbuster.py suspicious_image.png

# Save extracted data to a file
python stegbuster.py hidden_message.wav -o extracted_message.txt

# Verbose output
python stegbuster.py encoded_text.txt -v

# Try all methods
python stegbuster.py steganography_sample.jpg -a
```

## Technical Implementation

### Project Structure

```
stegbuster/
├── stegbuster.py          # Main CLI entry point
├── utils/
│   ├── __init__.py
│   └── file_utils.py      # File handling utilities
└── detectors/
    ├── __init__.py
    ├── detector_factory.py # Factory pattern for detector selection
    ├── base_detector.py    # Abstract base detector class
    ├── image_detector.py   # Image steganography detection
    ├── audio_detector.py   # Audio steganography detection
    └── text_detector.py    # Text steganography detection
```

### Detection Methods

#### Image Steganography

1. **LSB (Least Significant Bit) Detection:**
   - Examines the distribution of the least significant bits of pixel values
   - Looks for non-random patterns that indicate data hiding
   - Extracts bits sequentially and attempts to convert to ASCII text

2. **DCT (Discrete Cosine Transform) Detection:**
   - Analyzes DCT coefficients commonly used in JPEG steganography
   - Looks for anomalies in coefficient distribution
   - Extracts LSB data from mid-frequency coefficients

#### Audio Steganography

1. **LSB Audio Detection:**
   - Similar to image LSB, but works with audio samples
   - Analyzes the randomness of LSBs in audio data

2. **Echo Hiding Detection:**
   - Uses autocorrelation to detect hidden echoes
   - Identifies patterns in echo delays that might encode binary data

3. **Phase Coding Detection:**
   - Analyzes phase information from Fourier transform
   - Looks for unusual patterns or clustering in phase values

#### Text Steganography

1. **Whitespace Steganography:**
   - Detects unusual patterns of spaces and tabs
   - Analyzes trailing whitespace which may encode binary data

2. **Zero-Width Character Detection:**
   - Identifies invisible Unicode characters used to hide data
   - Extracts data encoded in these hidden characters

3. **Lexical Steganography:**
   - Analyzes patterns like first letters of each word/line
   - Detects unusual linguistic patterns that might encode data

## How It Works

1. The user provides a file to analyze
2. StegBuster identifies the file type using MIME detection
3. The appropriate detector is selected based on file type
4. Detection algorithms analyze the file for steganographic signatures
5. If steganography is detected, extraction algorithms attempt to recover hidden data
6. Results are displayed to the user, including the detected method and extracted content

## Limitations

- False positives may occur with certain file types or detection methods
- Not all steganographic techniques are detectable or extractable
- Some advanced techniques may require specific knowledge about the encoding method
- Binary data extraction may not always be interpretable as text

## Contributing

Contributions to improve detection algorithms, add support for new file types, or enhance extraction capabilities are welcome.

## License

[MIT License](LICENSE)
