# StegBuster Installation Guide

This guide provides detailed instructions for installing StegBuster and its dependencies on various operating systems.

## System Requirements

- Python 3.10-3.12 (TensorFlow compatibility)
- 4GB RAM minimum (8GB recommended)
- 500MB disk space for base installation
- Additional space for ML models (100MB-500MB depending on models used)

## Operating System Compatibility

StegBuster has been tested on:
- Ubuntu 20.04 LTS and newer
- Windows 10/11
- macOS 11.0 (Big Sur) and newer

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Create a virtual environment
python3 -m venv stegbuster-env

# Activate the virtual environment
# On Linux/macOS:
source stegbuster-env/bin/activate
# On Windows:
.\stegbuster-env\Scripts\activate

# Install StegBuster
pip install stegbuster

# Or install directly from GitHub
pip install git+https://github.com/yourusername/stegbuster.git
```

### Method 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stegbuster.git
cd stegbuster

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Dependency Installation

### Core Dependencies

```bash
pip install numpy scipy pillow opencv-python python-magic
```

### ML Dependencies

```bash
pip install tensorflow matplotlib
```

### Platform-Specific Dependencies

#### Linux (Ubuntu/Debian)

```bash
# For python-magic to work correctly
sudo apt-get install libmagic1
```

#### Windows

On Windows, `python-magic` has some additional requirements:

1. Install the package with binary dependencies:
```bash
pip install python-magic-bin
```

2. Alternatively, download and install the DLL manually:
   - Download [magic.dll](https://github.com/pidydx/libmagicwin64/raw/master/magic1.dll)
   - Rename to `magic.dll`
   - Place in `C:\Windows\System32\`

#### macOS

```bash
# Install libmagic using Homebrew
brew install libmagic
```

## Python Version Management

### Using pyenv (Recommended)

[pyenv](https://github.com/pyenv/pyenv) is useful for managing multiple Python versions:

```bash
# Install pyenv
# On Linux/macOS:
curl https://pyenv.run | bash

# Install specific Python version
pyenv install 3.10.x

# Set local Python version
cd /path/to/stegbuster
pyenv local 3.10.x
```

### Using Conda

```bash
# Create environment with specific Python version
conda create -n stegbuster python=3.10

# Activate environment
conda activate stegbuster

# Install dependencies
pip install -r requirements.txt
```

## Verifying Installation

After installation, verify that StegBuster works correctly:

```bash
# Run with help flag
python stegbuster.py --help

# Test on a sample image
python stegbuster.py demo/test.png
```

## Troubleshooting Common Issues

### TensorFlow Installation Issues

If you encounter issues with TensorFlow installation:

```bash
# Try installing a specific version
pip install tensorflow==2.10.0

# If on older hardware, try:
pip install tensorflow-cpu
```

### Missing python-magic Library

If you get an error about missing magic libraries:

- **Linux**: `sudo apt-get install libmagic1`
- **macOS**: `brew install libmagic`
- **Windows**: Use `python-magic-bin` package instead of `python-magic`

### ImportError: No module named 'cv2'

```bash
# Try reinstalling OpenCV
pip uninstall opencv-python
pip install opencv-python-headless
```

## Updating StegBuster

To update to the latest version:

```bash
# If installed via pip
pip install --upgrade stegbuster

# If installed from git
cd /path/to/stegbuster
git pull
pip install -e .
```

## Next Steps

After successful installation:
- Read the [User Guide](user_guide.md) to learn how to use StegBuster
- Check out [ML Integration](ml_integration_strategy.md) for details on machine learning features
- Try some examples in the [demo](../demo) directory