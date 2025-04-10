from ml_detectors.image_ml_detector import ImageMLDetector
from ml_detectors.audio_ml_detector import AudioMLDetector
from ml_detectors.text_ml_detector import TextMLDetector

def get_ml_detector(file_type):
    """Factory function to get the appropriate ML detector"""
    if file_type == 'image':
        return ImageMLDetector()
    elif file_type == 'audio':
        return AudioMLDetector()
    elif file_type == 'text':
        return TextMLDetector()
    else:
        return None
