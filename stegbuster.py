#!/usr/bin/env python3

import argparse
import os
import sys
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed"""
    dependencies = {
        "numpy": "numpy",
        "scipy": "scipy",
        "PIL": "pillow",
        "cv2": "opencv-python",
        "magic": "python-magic"
    }
    
    missing = []
    for module, package in dependencies.items():
        if importlib.util.find_spec(module) is None:
            missing.append(f"{module} ({package})")
    
    if missing:
        print("Warning: The following required dependencies are missing:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nTo install all dependencies, run: pip install -r requirements.txt")
        print("Some functionality may be limited.")
        print()

# Run dependency check at startup
check_dependencies()

from detectors.detector_factory import get_detector
from utils.file_utils import identify_file_type

def main():
    parser = argparse.ArgumentParser(
        description='StegBuster - Universal Steganography Detector and Extractor'
    )
    parser.add_argument('file', help='Path to the file to analyze')
    parser.add_argument('-o', '--output', help='Output file for extracted data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-a', '--all-methods', action='store_true', 
                        help='Try all possible extraction methods (slower)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)
    
    file_type = identify_file_type(args.file)
    if not file_type:
        print(f"Error: Unsupported file type for '{args.file}'")
        sys.exit(1)
        
    if args.verbose:
        print(f"Analyzing {args.file} (detected as: {file_type})")
    
    detector = get_detector(file_type)
    if not detector:
        print(f"Error: No detector available for file type: {file_type}")
        sys.exit(1)
    
    # Check if steganography is present
    steg_method = detector.detect(args.file)
    
    if not steg_method:
        print("No hidden data found.")
        sys.exit(0)
    
    print(f"Steganography detected using {steg_method}. Extracting hidden data...")
    
    # Extract the hidden message
    extracted_data = detector.extract(args.file, method=steg_method)
    
    if not extracted_data:
        print("Extraction failed or no data was found.")
        sys.exit(1)
    
    print(f"Hidden Message: {extracted_data}")
    
    # Save to output file if specified
    if args.output:
        with open(args.output, 'wb') as f:
            f.write(extracted_data if isinstance(extracted_data, bytes) else extracted_data.encode())
        print(f"Extracted data saved to {args.output}")

if __name__ == "__main__":
    main()
