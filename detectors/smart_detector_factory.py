from detectors.detector_factory import get_detector
from ml_detectors.ml_detector_factory import get_ml_detector
from detectors.smart_detector import SmartDetector

def get_smart_detector(file_type, ml_enabled=True, options=None):
    """
    Factory function to create a SmartDetector that intelligently combines
    traditional and ML-based detection methods.
    
    Args:
        file_type: Type of file to analyze ('image', 'audio', 'text')
        ml_enabled: Whether to enable ML detection if available
        options: Dictionary of options for the SmartDetector
        
    Returns:
        A SmartDetector instance, or None if no detector is available for the file type
    """
    # Get traditional detector first to check if file type is supported
    traditional_detector = get_detector(file_type)
    if not traditional_detector:
        return None
    
    # Get ML detector only if enabled
    ml_detector = None
    if ml_enabled:
        try:
            ml_detector = get_ml_detector(file_type)
        except (ImportError, ModuleNotFoundError):
            # ML dependencies missing, continue without ML
            pass
    
    # Create smart detector
    return SmartDetector(file_type, ml_detector, options)