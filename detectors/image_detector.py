import cv2
import numpy as np
from PIL import Image
from detectors.base_detector import BaseSteganographyDetector
from utils.file_utils import read_file_as_bytes

class ImageSteganographyDetector(BaseSteganographyDetector):
    """
    Detector for steganography in image files
    """
    
    def detect(self, file_path):
        """
        Detect if steganography is present in the image
        Returns the method used if detected, otherwise None
        """
        # Try different detection methods and return the first successful one
        if self._detect_lsb(file_path):
            return "LSB"
        if self._detect_dct(file_path):
            return "DCT"
        return None
    
    def extract(self, file_path, method=None):
        """
        Extract hidden data from the image
        Returns the extracted data if successful, otherwise None
        """
        if method == "LSB" or not method:
            data = self._extract_lsb(file_path)
            if data:
                return data
                
        if method == "DCT" or not method:
            data = self._extract_dct(file_path)
            if data:
                return data
                
        return None
    
    def _detect_lsb(self, file_path):
        """
        Detect if LSB steganography is present in the image
        """
        try:
            # Load the image
            img = cv2.imread(file_path)
            if img is None:
                return False
            
            # Convert to grayscale for simplicity
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Extract LSBs
            lsb = gray & 1
            
            # Check if the LSB distribution is suspicious
            # In normal images, LSBs should be fairly random (around 50% zeros and 50% ones)
            zero_count = np.sum(lsb == 0)
            one_count = np.sum(lsb == 1)
            total_pixels = gray.size
            
            zero_ratio = zero_count / total_pixels
            one_ratio = one_count / total_pixels
            
            # If the distribution is too skewed, it might indicate steganography
            if abs(zero_ratio - 0.5) > 0.05 or abs(one_ratio - 0.5) > 0.05:
                return True
                
            # Additional check: look for patterns in the LSB plane
            # Calculate horizontal and vertical differences
            h_diff = np.abs(np.diff(lsb, axis=1))
            v_diff = np.abs(np.diff(lsb, axis=0))
            
            h_changes = np.sum(h_diff)
            v_changes = np.sum(v_diff)
            
            # Calculate expected number of changes for random data
            expected_changes = total_pixels * 0.5
            
            # If there are significantly fewer changes than expected, it might be steganography
            if h_changes < expected_changes * 0.4 or v_changes < expected_changes * 0.4:
                return True
                
            return False
            
        except Exception as e:
            print(f"Error in LSB detection: {e}")
            return False
    
    def _extract_lsb(self, file_path):
        """
        Extract data hidden using LSB steganography
        """
        try:
            # Open the image
            img = Image.open(file_path)
            width, height = img.size
            
            # Convert image to numpy array
            array = np.array(list(img.getdata()))
            
            # Determine the number of channels
            if len(array.shape) == 1:
                # Grayscale image
                n_channels = 1
            else:
                # Color image
                n_channels = array.shape[1]
            
            # Get all LSBs
            bits = []
            for pixel in array:
                for channel in range(n_channels):
                    if n_channels == 1:
                        bits.append(pixel & 1)
                    else:
                        bits.append(pixel[channel] & 1)
            
            # Convert bits to bytes
            bytes_data = bytearray()
            for i in range(0, len(bits), 8):
                if i + 8 <= len(bits):
                    byte = 0
                    for j in range(8):
                        byte |= bits[i + j] << (7 - j)
                    bytes_data.append(byte)
            
            # Try to decode as ASCII
            try:
                # Find terminator (null byte)
                null_pos = bytes_data.find(0)
                if null_pos != -1:
                    bytes_data = bytes_data[:null_pos]
                
                message = bytes_data.decode('ascii')
                
                # Check if the message contains printable characters
                if any(32 <= ord(c) <= 126 for c in message):
                    return message
                return None
            except:
                # If we can't decode as ASCII, return the raw bytes
                return bytes(bytes_data)
                
        except Exception as e:
            print(f"Error in LSB extraction: {e}")
            return None
    
    def _detect_dct(self, file_path):
        """
        Detect if DCT-based steganography (like that used in JPEG) is present
        """
        try:
            # Read the image
            img = cv2.imread(file_path)
            if img is None:
                return False
            
            # Convert to YCrCb color space
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            
            # Take the Y channel
            y = ycrcb[:,:,0]
            
            # Apply DCT transform
            height, width = y.shape
            height = height - height % 8  # Make sure dimensions are multiples of 8
            width = width - width % 8
            y = y[:height, :width]
            
            # Process the image in 8x8 blocks
            dct_coeffs = []
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    block = y[i:i+8, j:j+8]
                    dct_block = cv2.dct(np.float32(block))
                    dct_coeffs.extend(dct_block.flatten().tolist())
            
            # Analyze the coefficients
            dct_coeffs = np.array(dct_coeffs)
            
            # In JPEG steganography, coefficients are often modified in a specific way
            # Check for anomalies in coefficient distribution
            hist, _ = np.histogram(dct_coeffs, bins=100)
            
            # Look for unusual peaks or patterns
            diff = np.diff(hist)
            unusual_changes = np.sum(np.abs(diff) > np.mean(np.abs(diff)) * 2)
            
            if unusual_changes > len(diff) * 0.1:
                return True
                
            return False
            
        except Exception as e:
            print(f"Error in DCT detection: {e}")
            return False
    
    def _extract_dct(self, file_path):
        """
        Extract data hidden using DCT-based steganography
        """
        # DCT extraction is complex and algorithm-specific
        # This is a simplified implementation focusing on common patterns
        try:
            # Read the image
            img = cv2.imread(file_path)
            if img is None:
                return None
            
            # Convert to YCrCb color space
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            
            # Take the Y channel
            y = ycrcb[:,:,0]
            
            # Apply DCT transform
            height, width = y.shape
            height = height - height % 8
            width = width - width % 8
            y = y[:height, :width]
            
            # Process the image in 8x8 blocks and extract potential data
            bits = []
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    block = y[i:i+8, j:j+8]
                    dct_block = cv2.dct(np.float32(block))
                    
                    # Extract LSB of the mid-frequency coefficients
                    # (commonly used in F5, OutGuess, etc.)
                    for k in range(1, 8):
                        for l in range(1, 8):
                            if k + l >= 5 and k + l <= 7:  # Mid-frequency coefficients
                                # Get LSB of the coefficient
                                bit = int(abs(dct_block[k, l]) % 2)
                                bits.append(bit)
            
            # Convert bits to bytes
            bytes_data = bytearray()
            for i in range(0, len(bits), 8):
                if i + 8 <= len(bits):
                    byte = 0
                    for j in range(8):
                        byte |= bits[i + j] << (7 - j)
                    bytes_data.append(byte)
            
            # Try to decode as ASCII
            try:
                # Find terminator
                null_pos = bytes_data.find(0)
                if null_pos != -1:
                    bytes_data = bytes_data[:null_pos]
                
                message = bytes_data.decode('ascii')
                
                # Check if the message contains printable characters
                if any(32 <= ord(c) <= 126 for c in message):
                    return message
                return None
            except:
                # Return raw bytes if ASCII decoding fails
                return bytes(bytes_data)
                
        except Exception as e:
            print(f"Error in DCT extraction: {e}")
            return None
