from detectors.image_detector import ImageSteganographyDetector
from detectors.audio_detector import AudioSteganographyDetector
from detectors.text_detector import TextSteganographyDetector

def get_detector(file_type):
    """
    Factory function to get the appropriate detector based on file type
    """
    if file_type == 'image':
        return ImageSteganographyDetector()
    elif file_type == 'audio':
        return AudioSteganographyDetector()
    elif file_type == 'text':
        return TextSteganographyDetector()
    else:
        return None
