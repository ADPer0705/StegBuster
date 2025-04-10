#!/usr/bin/env python3

import argparse
import os
import sys
import importlib.util
import json
import logging
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_dependencies():
    """Check if all required dependencies are installed"""
    dependencies = {
        "numpy": "numpy",
        "scipy": "scipy",
        "PIL": "pillow",
        "cv2": "opencv-python",
        "magic": "python-magic"
    }
    
    # Add ML dependencies
    ml_dependencies = {
        "tensorflow": "tensorflow",
        "matplotlib": "matplotlib"
    }
    
    # Check Python version
    python_version = platform.python_version()
    python_major, python_minor, _ = map(int, python_version.split('.'))
    tf_compatible = (python_major == 3 and 10 <= python_minor <= 12)
    
    missing = []
    for module, package in dependencies.items():
        if importlib.util.find_spec(module) is None:
            missing.append(f"{module} ({package})")
    
    ml_missing = []
    for module, package in ml_dependencies.items():
        if importlib.util.find_spec(module) is None:
            ml_missing.append(f"{module} ({package})")
    
    if missing:
        print("Warning: The following required dependencies are missing:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nTo install all dependencies, run: pip install -r requirements.txt")
        print("Some functionality may be limited.")
        print()
    
    if ml_missing:
        print("Warning: The following ML dependencies are missing:")
        for pkg in ml_missing:
            print(f"  - {pkg}")
        
        if not tf_compatible:
            print(f"\nYour Python version ({python_version}) is not compatible with TensorFlow.")
            print("TensorFlow requires Python 3.10-3.12. Please use a compatible Python version:")
            print("  1. Create a virtual environment: python3.10 -m venv steg-env-py310")
            print("  2. Activate it: source steg-env-py310/bin/activate (Linux/macOS)")
            print("                  .\\steg-env-py310\\Scripts\\activate (Windows)")
            print("  3. Install dependencies: pip install -r requirements.txt")
        else:
            print("\nTo enable ML functionality, run: pip install tensorflow matplotlib")
        
        print("ML-based detection will be disabled.")
        print()
    
    return len(missing) == 0, len(ml_missing) == 0 and tf_compatible

# Run dependency check at startup
basic_deps_ok, ml_deps_ok = check_dependencies()

from detectors.detector_factory import get_detector
from utils.file_utils import identify_file_type

# Conditionally import ML-related modules
if ml_deps_ok:
    try:
        from ml_detectors.ml_detector_factory import get_ml_detector
        from detectors.smart_detector_factory import get_smart_detector
        ml_imports_ok = True
    except ImportError:
        ml_deps_ok = False
        ml_imports_ok = False
        logging.warning("ML modules could not be imported. ML detection disabled.")
else:
    ml_imports_ok = False

def main():
    parser = argparse.ArgumentParser(
        description='StegBuster - Universal Steganography Detector and Extractor'
    )
    parser.add_argument('file', help='Path to the file to analyze')
    parser.add_argument('-o', '--output', help='Output file for extracted data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-a', '--all-methods', action='store_true', 
                        help='Try all possible extraction methods (slower)')
    parser.add_argument('-c', '--chi-square', action='store_true',
                        help='Perform detailed chi-square analysis')
    parser.add_argument('-r', '--rs-analysis', action='store_true',
                        help='Perform detailed RS (Regular-Singular) analysis')
    parser.add_argument('--detailed-analysis', action='store_true',
                        help='Perform comprehensive steganalysis with all methods')
    parser.add_argument('--json', action='store_true',
                        help='Output results in JSON format (for detailed analysis)')
    
    # Add ML-related arguments
    parser.add_argument('--use-ml', action='store_true', 
                        help='Use machine learning for enhanced detection')
    parser.add_argument('--smart-ml', action='store_true',
                        help='Use ML intelligently, only when needed for better performance')
    parser.add_argument('--ml-confidence', type=float, default=0.7,
                        help='Minimum confidence threshold for ML detection (0-1)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of detection results')
    parser.add_argument('--advanced-model', action='store_true',
                        help='Use the advanced ML model (larger, but more accurate)')
    
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
    
    # Determine whether to use ML
    use_ml = (args.use_ml or args.smart_ml) and ml_deps_ok and ml_imports_ok
    
    # Use the new SmartDetector if smart_ml is enabled
    if args.smart_ml and ml_imports_ok:
        # Configure smart detector options
        smart_options = {
            'ml_confidence': args.ml_confidence,
            'always_use_ml': args.use_ml,
            'visualize': args.visualize,
            'ml_first': True
        }
        
        if args.advanced_model:
            smart_options['model_type'] = 'advanced'
            
        # Create smart detector
        detector = get_smart_detector(file_type, ml_enabled=True, options=smart_options)
        if not detector:
            print(f"Error: No detector available for file type: {file_type}")
            sys.exit(1)
            
        if args.verbose:
            print("Using smart detector with intelligent ML integration")
        
        # Run detection
        steg_method, confidence, ml_used = detector.detect(args.file)
        
        # Report results
        if not steg_method:
            print("No hidden data found.")
            sys.exit(0)
            
        confidence_str = f" (confidence: {confidence:.1%})" if confidence else ""
        ml_str = " using ML analysis" if ml_used else ""
        print(f"Steganography detected using {steg_method}{confidence_str}{ml_str}. Extracting hidden data...")
        
        # Extract the hidden message
        extracted_data = detector.extract(args.file, method=steg_method)
            
        if not extracted_data:
            print("Extraction failed or no data was found.")
            sys.exit(1)
            
        # Try to display the data in a readable format if it's text
        if isinstance(extracted_data, bytes):
            try:
                decoded_text = extracted_data.decode('utf-8')
                print(f"Hidden Message: {decoded_text}")
            except UnicodeDecodeError:
                print(f"Hidden Message: [Binary data of {len(extracted_data)} bytes]")
        else:
            print(f"Hidden Message: {extracted_data}")
        
        # Save to output file if specified
        if args.output:
            mode = 'wb' if isinstance(extracted_data, bytes) else 'w'
            with open(args.output, mode) as f:
                if isinstance(extracted_data, bytes):
                    f.write(extracted_data)
                else:
                    f.write(extracted_data)
            print(f"Extracted data saved to {args.output}")
            
        sys.exit(0)
            
    # Original ML-based detection path
    elif args.use_ml:
        if not ml_deps_ok:
            print("Error: ML dependencies are missing. ML detection is disabled.")
            print("Install requirements with: pip install tensorflow matplotlib")
            sys.exit(1)
            
        try:
            # Get ML detector
            ml_detector = get_ml_detector(file_type)
            if not ml_detector:
                print(f"Error: ML detector not available for file type: {file_type}")
                sys.exit(1)
                
            # Set ML parameters
            ml_detector.set_confidence_threshold(args.ml_confidence)
            if args.advanced_model:
                ml_detector.set_model("image_advanced")
                
            print(f"Performing ML-based steganalysis on {args.file}...")
            is_stego_likely, confidence, ml_method, viz_path = ml_detector.detect(
                args.file, visualize=args.visualize
            )
            
            # Output ML results
            confidence_pct = confidence * 100
            print(f"\n=== ML Steganalysis Results ===")
            print(f"Steganography detected: {'Yes' if is_stego_likely else 'No'}")
            print(f"Confidence: {confidence_pct:.1f}%")
            
            if is_stego_likely:
                print(f"Suspected method: {ml_method}")
                
                # Get more detailed analysis if available and confidence is high
                if hasattr(ml_detector, 'analyze_embedding_method') and confidence > args.ml_confidence:
                    method_analysis = ml_detector.analyze_embedding_method(args.file)
                    print("\nProbable embedding methods:")
                    for method, prob in sorted(method_analysis.items(), key=lambda x: x[1], reverse=True):
                        if prob > 0.05:  # Only show methods with >5% probability
                            print(f"  - {method}: {prob*100:.1f}%")
            
            if args.visualize and viz_path:
                print(f"\nVisualization saved to: {viz_path}")
                
            # If ML detection is confident enough, exit
            if confidence > 0.9:
                if args.json:
                    result = {
                        "file": args.file,
                        "is_stego_likely": is_stego_likely,
                        "confidence": confidence,
                        "method": ml_method,
                        "visualization": viz_path
                    }
                    print(json.dumps(result, indent=2))
                sys.exit(0)
                
            # If not confident, fall back to traditional analysis
            if args.verbose:
                print("\nML detection not confident enough, falling back to traditional methods...")
                
        except Exception as e:
            print(f"Error in ML analysis: {str(e)}")
            print("Falling back to traditional analysis...")
    
    # Traditional detection
    detector = get_detector(file_type)
    if not detector:
        print(f"Error: No detector available for file type: {file_type}")
        sys.exit(1)
    
    # Handle detailed analysis request
    if args.detailed_analysis or args.rs_analysis:
        if hasattr(detector, 'perform_detailed_analysis'):
            print(f"Performing detailed steganalysis on {args.file}...")
            results = detector.perform_detailed_analysis(args.file)
            
            if args.json:
                # Output as JSON
                print(json.dumps(results, indent=2))
            else:
                # Pretty print the results
                print("\n=== Detailed Steganalysis Results ===")
                for channel, data in results.items():
                    if channel == 'combined' or channel == 'error':
                        continue
                        
                    print(f"\n{channel} Channel:")
                    if 'overall' in data:
                        confidence = data['overall']['confidence'] * 100
                        print(f"  Steganography likelihood: {'Yes' if data['overall']['is_stego_likely'] else 'No'} (confidence: {confidence:.1f}%)")
                    
                    if args.rs_analysis and 'rs_analysis' in data:
                        rs = data['rs_analysis']
                        print(f"  RS Analysis:")
                        print(f"    Estimated embedding rate: {rs['embedding_rate']:.2%}")
                        print(f"    RM/SM ratios: {rs['rm_ratio']:.3f}/{rs['sm_ratio']:.3f}")
                        print(f"    R-M/S-M ratios: {rs['r_m_ratio']:.3f}/{rs['s_m_ratio']:.3f}")
                
                if 'combined' in results:
                    combined = results['combined']
                    print("\nOverall Assessment:")
                    confidence = combined['confidence'] * 100
                    print(f"  Steganography detected: {'Yes' if combined['is_stego_likely'] else 'No'}")
                    print(f"  Confidence: {confidence:.1f}%")
                    
                if 'error' in results:
                    print(f"\nError: {results['error']}")
                
            # Exit after detailed analysis
            sys.exit(0)
        else:
            print(f"Detailed analysis not available for file type: {file_type}")
            if not args.verbose:  # Continue with regular detection if verbose
                sys.exit(1)
    
    # Regular detection and extraction
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
    
    # Try to display the data in a readable format if it's text
    if isinstance(extracted_data, bytes):
        try:
            decoded_text = extracted_data.decode('utf-8')
            print(f"Hidden Message: {decoded_text}")
        except UnicodeDecodeError:
            print(f"Hidden Message: [Binary data of {len(extracted_data)} bytes]")
    else:
        print(f"Hidden Message: {extracted_data}")
    
    # Save to output file if specified
    if args.output:
        mode = 'wb' if isinstance(extracted_data, bytes) else 'w'
        with open(args.output, mode) as f:
            if isinstance(extracted_data, bytes):
                f.write(extracted_data)
            else:
                f.write(extracted_data)
        print(f"Extracted data saved to {args.output}")

if __name__ == "__main__":
    main()
