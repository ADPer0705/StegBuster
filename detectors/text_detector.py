import re
import numpy as np
from detectors.base_detector import BaseSteganographyDetector

class TextSteganographyDetector(BaseSteganographyDetector):
    """
    Detector for steganography in text files
    """
    
    def detect(self, file_path):
        """
        Detect if steganography is present in the text file
        Returns the method used if detected, otherwise None
        """
        if self._detect_whitespace_steg(file_path):
            return "Whitespace"
        if self._detect_zero_width_steg(file_path):
            return "Zero-Width Characters"
        if self._detect_lexical_steg(file_path):
            return "Lexical"
        return None
    
    def extract(self, file_path, method=None):
        """
        Extract hidden data from the text file
        Returns the extracted data if successful, otherwise None
        """
        if method == "Whitespace" or not method:
            data = self._extract_whitespace_steg(file_path)
            if data:
                return data
                
        if method == "Zero-Width Characters" or not method:
            data = self._extract_zero_width_steg(file_path)
            if data:
                return data
                
        if method == "Lexical" or not method:
            data = self._extract_lexical_steg(file_path)
            if data:
                return data
                
        return None
    
    def _detect_whitespace_steg(self, file_path):
        """
        Detect if whitespace steganography is present in the text file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for unusual patterns of spaces and tabs
            lines = content.split('\n')
            
            space_count = 0
            tab_count = 0
            trailing_space_count = 0
            
            for line in lines:
                space_count += line.count(' ')
                tab_count += line.count('\t')
                if line.endswith(' ') or line.endswith('\t'):
                    trailing_space_count += 1
            
            # Check for suspicious indicators
            
            # High number of trailing spaces (unusual in normal text)
            if trailing_space_count > len(lines) * 0.2:
                return True
                
            # Unusual ratio of tabs to spaces
            if tab_count > 0 and space_count > 0:
                ratio = tab_count / space_count
                if ratio > 0.5:  # Normal text files usually have more spaces than tabs
                    return True
                    
            # Look for patterns of spaces/tabs at line endings
            pattern_found = False
            for i in range(len(lines) - 1):
                if (lines[i].endswith(' ') and lines[i+1].endswith('\t')) or \
                   (lines[i].endswith('\t') and lines[i+1].endswith(' ')):
                    pattern_found = True
                    break
                    
            if pattern_found:
                return True
                
            return False
            
        except Exception as e:
            print(f"Error in whitespace steganography detection: {e}")
            return False
    
    def _extract_whitespace_steg(self, file_path):
        """
        Extract data hidden using whitespace steganography
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract bits based on whitespace
            # Common method: space = 0, tab = 1
            
            lines = content.split('\n')
            bits = []
            
            # Check for trailing whitespace in each line
            for line in lines:
                if line.endswith(' '):
                    bits.append(0)
                elif line.endswith('\t'):
                    bits.append(1)
            
            # If no trailing whitespace found, try another approach
            if not bits:
                # Check for spaces between words
                for line in lines:
                    words = line.split()
                    for i in range(len(words) - 1):
                        gap = re.search(r'(\s+)', line)
                        if gap:
                            if gap.group(1) == ' ':
                                bits.append(0)
                            elif gap.group(1) == '\t':
                                bits.append(1)
                            elif gap.group(1) == '  ':  # Double space
                                bits.append(0)
                                bits.append(0)
                            elif gap.group(1) == ' \t':  # Space tab
                                bits.append(0)
                                bits.append(1)
            
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
                message = bytes_data.decode('ascii')
                if any(32 <= ord(c) <= 126 for c in message):
                    return message
                return None
            except:
                return bytes(bytes_data)
                
        except Exception as e:
            print(f"Error in whitespace steganography extraction: {e}")
            return None
    
    def _detect_zero_width_steg(self, file_path):
        """
        Detect if zero-width character steganography is present
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for presence of zero-width characters
            zero_width_chars = [
                '\u200B',  # Zero-width space
                '\u200C',  # Zero-width non-joiner
                '\u200D',  # Zero-width joiner
                '\u2060',  # Word joiner
                '\u2061',  # Function application
                '\u2062',  # Invisible times
                '\u2063',  # Invisible separator
                '\u2064',  # Invisible plus
                '\uFEFF'   # Zero-width no-break space
            ]
            
            for char in zero_width_chars:
                if char in content:
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error in zero-width character detection: {e}")
            return False
    
    def _extract_zero_width_steg(self, file_path):
        """
        Extract data hidden using zero-width character steganography
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Map zero-width characters to bits
            # Common encoding:
            # Zero-width space (\u200B) = 0
            # Zero-width non-joiner (\u200C) = 1
            
            bits = []
            for char in content:
                if char == '\u200B':
                    bits.append(0)
                elif char == '\u200C':
                    bits.append(1)
            
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
                message = bytes_data.decode('ascii')
                if any(32 <= ord(c) <= 126 for c in message):
                    return message
                return None
            except:
                return bytes(bytes_data)
                
        except Exception as e:
            print(f"Error in zero-width character extraction: {e}")
            return None
    
    def _detect_lexical_steg(self, file_path):
        """
        Detect if lexical steganography is present (e.g., first letter of each line/word)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Extract first character of each line
            first_chars = [line[0] if line else '' for line in lines]
            first_chars = [c for c in first_chars if c]
            
            # Check if the first characters form readable text
            if self._is_possibly_text(''.join(first_chars)):
                return True
                
            # Extract first character of each word
            words = re.findall(r'\b\w+\b', content)
            first_chars = [word[0] if word else '' for word in words]
            first_chars = [c for c in first_chars if c]
            
            # Check if the first characters form readable text
            if self._is_possibly_text(''.join(first_chars)):
                return True
                
            return False
            
        except Exception as e:
            print(f"Error in lexical steganography detection: {e}")
            return False
    
    def _extract_lexical_steg(self, file_path):
        """
        Extract data hidden using lexical steganography
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Try different methods of extraction
            
            # Method 1: First character of each line
            first_chars = [line[0] if line and line.strip() else '' for line in lines]
            first_chars = [c for c in first_chars if c]
            message1 = ''.join(first_chars)
            
            # Method 2: First character of each word
            words = re.findall(r'\b\w+\b', content)
            first_chars = [word[0] if word else '' for word in words]
            first_chars = [c for c in first_chars if c]
            message2 = ''.join(first_chars)
            
            # Return the message that looks more like readable text
            if self._is_possibly_text(message1) and len(message1) > 3:
                return message1
            if self._is_possibly_text(message2) and len(message2) > 3:
                return message2
                
            return None
            
        except Exception as e:
            print(f"Error in lexical steganography extraction: {e}")
            return None
    
    def _is_possibly_text(self, text):
        """
        Check if a string might be readable text
        """
        if not text:
            return False
            
        # Check if text contains mostly printable ASCII characters
        printable_count = sum(32 <= ord(c) <= 126 for c in text)
        if printable_count / len(text) < 0.7:
            return False
            
        # Check for reasonable character distribution
        # English text typically has more vowels
        vowel_count = sum(c.lower() in 'aeiou' for c in text)
        consonant_count = sum(c.lower() in 'bcdfghjklmnpqrstvwxyz' for c in text)
        
        if len(text) > 5:
            # In English, vowels typically make up 40% of text
            vowel_ratio = vowel_count / (vowel_count + consonant_count) if (vowel_count + consonant_count) > 0 else 0
            if 0.2 <= vowel_ratio <= 0.6:
                return True
                
        return False
