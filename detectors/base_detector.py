from abc import ABC, abstractmethod

class BaseSteganographyDetector(ABC):
    """
    Base abstract class for steganography detectors
    """
    
    @abstractmethod
    def detect(self, file_path):
        """
        Detect if steganography is present in the file
        Returns the method used if detected, otherwise None
        """
        pass
    
    @abstractmethod
    def extract(self, file_path, method=None):
        """
        Extract hidden data from the file
        Returns the extracted data if successful, otherwise None
        """
        pass
