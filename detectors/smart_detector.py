from detectors.base_detector import BaseSteganographyDetector
from detectors.detector_factory import get_detector
import os
import logging

class SmartDetector(BaseSteganographyDetector):
    """
    A smart detector that combines traditional and ML-based detection methods.
    It intelligently decides when to use ML to avoid unnecessary performance impacts.
    """
    
    def __init__(self, file_type, ml_detector=None, options=None):
        """
        Initialize the smart detector with traditional and optional ML detector
        
        Args:
            file_type: Type of file to analyze ('image', 'audio', 'text')
            ml_detector: Optional ML detector instance
            options: Dictionary of options including:
                - ml_confidence: Confidence threshold for ML detection (default: 0.7)
                - always_use_ml: Always use ML detection if available (default: False)
                - visualize: Generate visualization of detection results (default: False)
        """
        self.file_type = file_type
        self.traditional_detector = get_detector(file_type)
        self.ml_detector = ml_detector
        
        # Default options
        self.options = {
            'ml_confidence': 0.7,
            'always_use_ml': False,
            'visualize': False,
            'ml_first': True,  # Try ML detection first by default
            'file_size_threshold': 10 * 1024 * 1024,  # 10MB threshold for large files
            'cache_ml_results': True  # Cache ML results to avoid redundant processing
        }
        
        # Update with provided options
        if options:
            self.options.update(options)
        
        # Initialize results cache
        self._ml_results_cache = {}
        
        # Check if we have valid detectors
        if not self.traditional_detector:
            raise ValueError(f"No traditional detector available for file type: {file_type}")
    
    def detect(self, file_path):
        """
        Intelligently detect steganography using a combination of methods
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Tuple of (method, confidence, is_ml_used)
            - method: The detected steganography method, or None if not detected
            - confidence: Confidence score (0-1) or None if not available
            - is_ml_used: Whether ML was used in the detection
        """
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        is_large_file = file_size > self.options['file_size_threshold']
        
        # Fast path for large files - skip ML unless specifically requested
        if is_large_file and not self.options['always_use_ml']:
            logging.info(f"Large file detected ({file_size / 1024 / 1024:.1f} MB). Skipping ML analysis.")
            method = self.traditional_detector.detect(file_path)
            return method, None, False
        
        # Check cache for ML results if enabled
        if self.options['cache_ml_results'] and file_path in self._ml_results_cache:
            ml_result = self._ml_results_cache[file_path]
            return ml_result['method'], ml_result['confidence'], True
        
        # ML detection if available and requested
        ml_result = None
        if self.ml_detector and (self.options['always_use_ml'] or self.options['ml_first']):
            try:
                is_stego_likely, confidence, ml_method, viz_path = self.ml_detector.detect(
                    file_path, visualize=self.options['visualize']
                )
                
                # Store in cache
                if self.options['cache_ml_results']:
                    self._ml_results_cache[file_path] = {
                        'is_stego_likely': is_stego_likely,
                        'confidence': confidence,
                        'method': ml_method if is_stego_likely else None,
                        'visualization_path': viz_path
                    }
                
                # If ML is confident enough, return its result
                if is_stego_likely and confidence >= self.options['ml_confidence']:
                    return ml_method, confidence, True
                    
                # Store ML result for potential later use
                ml_result = (ml_method if is_stego_likely else None, confidence)
            except Exception as e:
                logging.warning(f"ML detection error: {str(e)}")
                # Fall back to traditional detection
        
        # Always run traditional detection if ML is not confident enough or unavailable
        method = self.traditional_detector.detect(file_path)
        
        # If both methods detected something, use the one with higher confidence
        if method and ml_result and ml_result[0]:
            # Traditional method detected with higher implied confidence than ML
            if ml_result[1] < self.options['ml_confidence']:
                return method, None, False
            # ML method with good confidence
            else:
                return ml_result[0], ml_result[1], True
        
        # Otherwise return the successful detection
        return method, None, False
    
    def extract(self, file_path, method=None):
        """
        Extract hidden data using the appropriate detector
        
        Args:
            file_path: Path to the file to extract from
            method: The steganography method to use for extraction
            
        Returns:
            The extracted data if successful, otherwise None
        """
        # If method contains 'ml_' prefix, use ML detector for extraction
        if method and method.startswith('ml_') and self.ml_detector:
            try:
                return self.ml_detector.extract(file_path, method)
            except Exception as e:
                logging.warning(f"ML extraction error: {str(e)}. Falling back to traditional extraction.")
                # Strip ml_ prefix and try traditional extraction
                traditional_method = method[3:] if method.startswith('ml_') else method
                return self.traditional_detector.extract(file_path, traditional_method)
        
        # Otherwise use traditional detector
        return self.traditional_detector.extract(file_path, method)