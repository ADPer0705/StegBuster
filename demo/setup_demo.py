#!/usr/bin/env python3
"""
StegBuster Demo Setup Script
===========================

This script sets up all the demonstration files for StegBuster by running the 
individual generators for different steganography types:
- Image steganography (LSB and DCT)
- Audio steganography (LSB and Echo Hiding)
- Text steganography (Whitespace, Zero-width characters, and Lexical)

Run this script to prepare all demo files for your crypto finals presentation.
"""

import os
import sys
import subprocess
import time
import importlib

def print_colored(text, color):
    """Print colored text for better visualization."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'end': '\033[0m'
    }
    
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def check_requirements():
    """Check if required packages are installed."""
    print_colored("Checking required packages...", "blue")
    
    required_packages = [
        ("numpy", "numpy"),
        ("pillow", "PIL"),  # PIL is the import name for pillow
        ("opencv-python", "cv2"),  # cv2 is the import name for opencv-python
        ("scipy", "scipy"),
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            importlib.import_module(import_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name} is not installed")
    
    if missing_packages:
        print_colored("\nSome packages are missing. Install them with:", "yellow")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print_colored("All required packages are installed.", "green")
    return True

def run_generator(script_name, description):
    """Run a generator script and handle errors."""
    print_colored(f"\n{description}:", "blue")
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    
    try:
        print(f"Running {script_name}...")
        subprocess.run([sys.executable, script_path], check=True)
        print_colored(f"✓ Successfully generated {description.lower()}", "green")
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"✗ Error running {script_name}: {e}", "red")
        return False
    except Exception as e:
        print_colored(f"✗ Unexpected error: {e}", "red")
        return False

def main():
    print_colored("=" * 80, "cyan")
    print_colored("StegBuster Demo Setup", "cyan")
    print_colored("=" * 80, "cyan")
    print("This script will generate all steganography demo files for your crypto finals presentation.\n")
    
    if not check_requirements():
        sys.exit(1)
    
    print_colored("\nStarting generators...", "purple")
    
    # Run each generator
    success = True
    success &= run_generator("generate_stego_images.py", "Image Steganography Files")
    success &= run_generator("generate_stego_audio.py", "Audio Steganography Files")
    success &= run_generator("generate_stego_text.py", "Text Steganography Files")
    
    if success:
        print_colored("\n" + "=" * 80, "green")
        print_colored("✓ All demo files generated successfully!", "green")
        print_colored("=" * 80, "green")
        print("\nYou can find the demo files in the following directories:")
        print("- Images: ./demo/images/")
        print("  - LSB: ./demo/images/lsb/")
        print("  - DCT: ./demo/images/dct/")
        print("- Audio: ./demo/audio/")
        print("  - LSB: ./demo/audio/lsb/")
        print("  - Echo: ./demo/audio/echo/")
        print("- Text: ./demo/text/")
        print("  - Whitespace: ./demo/text/whitespace/")
        print("  - Zero-width: ./demo/text/zerowidth/")
        print("  - Lexical: ./demo/text/lexical/")
        
        print("\nTo demonstrate StegBuster with these files, run:")
        print_colored("python stegbuster.py -f <filepath> -a detect", "yellow")
        print("For example:")
        print_colored("python stegbuster.py -f demo/images/lsb/gradient_lsb_short.png -a detect", "yellow")
    else:
        print_colored("\n✗ Some generators encountered errors. Please check the output above.", "red")
        sys.exit(1)
    
if __name__ == "__main__":
    main()