import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from detectors.base_detector import BaseSteganographyDetector
from utils.statistical_analysis import chi_square_attack_lsb, sample_pair_analysis, rs_analysis, perform_detailed_analysis

class AudioSteganographyDetector(BaseSteganographyDetector):
    """
    Detector for steganography in audio files
    """
    
    def detect(self, file_path):
        """
        Detect if steganography is present in the audio file
        Returns the method used if detected, otherwise None
        """
        if self._detect_lsb_audio(file_path):
            return "LSB"
        if self._detect_echo_hiding(file_path):
            return "Echo Hiding"
        if self._detect_phase_coding(file_path):
            return "Phase Coding"
        return None
    
    def extract(self, file_path, method=None):
        """
        Extract hidden data from the audio file
        Returns the extracted data if successful, otherwise None
        """
        if method == "LSB" or not method:
            data = self._extract_lsb_audio(file_path)
            if data:
                return data
                
        if method == "Echo Hiding" or not method:
            data = self._extract_echo_hiding(file_path)
            if data:
                return data
                
        if method == "Phase Coding" or not method:
            data = self._extract_phase_coding(file_path)
            if data:
                return data
                
        return None
    
    def _detect_lsb_audio(self, file_path):
        """
        Detect if LSB steganography is present in the audio file
        """
        try:
            # Read the audio file
            sample_rate, data = wavfile.read(file_path)
            
            # Handle multi-channel audio
            if len(data.shape) > 1:
                # Use the first channel for simplicity
                data = data[:, 0]
            
            # Use advanced chi-square attack
            chi2_stat, p_value, is_stego_likely = chi_square_attack_lsb(data, sample_size=10000)
            
            # Use Sample Pair Analysis as a secondary method
            spa_rate = sample_pair_analysis(data, sample_size=10000)
            
            # Use RS Analysis as a third method (adapted for audio)
            rs_rate, _, _, _, _, _ = rs_analysis(data, sample_size=1000)
            
            # Combine results for more accurate detection
            # Consider it positive if any two methods indicate steganography
            detection_count = sum([
                1 if is_stego_likely else 0,
                1 if spa_rate > 0.3 else 0,  # 30% embedding rate threshold for SPA
                1 if rs_rate > 0.2 else 0    # 20% embedding rate threshold for RS
            ])
            
            if detection_count >= 2:  # At least 2 methods detect steganography
                return True
                
            # Additional check for LSB distribution as a fallback
            lsb = data & 1
            zero_count = np.sum(lsb == 0)
            one_count = np.sum(lsb == 1)
            total_samples = len(data)
            
            zero_ratio = zero_count / total_samples
            one_ratio = one_count / total_samples
            
            if abs(zero_ratio - 0.5) > 0.05 or abs(one_ratio - 0.5) > 0.05:
                return True
                
            return False
            
        except Exception as e:
            print(f"Error in audio LSB detection: {e}")
            return False

    def perform_detailed_analysis(self, file_path):
        """
        Perform detailed steganalysis on the audio file
        Returns a dictionary with analysis results
        """
        try:
            # Read the audio file
            sample_rate, data = wavfile.read(file_path)
            
            # Handle multi-channel audio
            results = {}
            
            if len(data.shape) > 1:
                # Process each channel
                num_channels = data.shape[1]
                for i in range(num_channels):
                    channel_data = data[:, i]
                    results[f"Channel_{i+1}"] = perform_detailed_analysis(channel_data)
            else:
                # Single channel
                results["Channel_1"] = perform_detailed_analysis(data)
            
            # Combine channel results for an overall assessment
            combined_votes = sum(channel_data['overall']['detection_votes'] for channel_data in results.values())
            max_votes = 3 * len(results)  # 3 methods × number of channels
            
            results['combined'] = {
                'is_stego_likely': combined_votes >= max_votes / 2,
                'confidence': combined_votes / max_votes
            }
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def _chi_square_attack(self, data):
        """
        Perform Chi-square attack to detect LSB steganography
        """
        try:
            # Split data into even and odd indices
            even_bins = data[::2] & 1
            odd_bins = data[1::2] & 1
            
            # Count occurrences of 0s and 1s
            even_counts = np.bincount(even_bins, minlength=2)
            odd_counts = np.bincount(odd_bins, minlength=2)
            
            # Calculate Chi-square statistic
            chi_square = 0
            for i in range(2):
                expected = (even_counts[i] + odd_counts[i]) / 2
                if expected > 0:
                    chi_square += ((even_counts[i] - expected) ** 2) / expected
                    chi_square += ((odd_counts[i] - expected) ** 2) / expected
            
            return chi_square
        except Exception as e:
            print(f"Error in Chi-square attack: {e}")
            return 0
    
    def _extract_lsb_audio(self, file_path):
        """
        Extract data hidden using LSB steganography in audio
        """
        try:
            # Read the audio file
            sample_rate, data = wavfile.read(file_path)
            
            # Handle multi-channel audio
            if len(data.shape) > 1:
                # Use the first channel for simplicity
                data = data[:, 0]
            
            # Extract LSBs
            bits = data & 1
            
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
            print(f"Error in audio LSB extraction: {e}")
            return None
    
    def _detect_echo_hiding(self, file_path):
        """
        Detect if echo hiding steganography is used in the audio file
        """
        try:
            # Read the audio file
            sample_rate, data = wavfile.read(file_path)
            
            # Handle multi-channel audio
            if len(data.shape) > 1:
                # Use the first channel for simplicity
                data = data[:, 0]
            
            # Calculate autocorrelation to detect echoes
            autocorr = np.correlate(data, data, mode='full')
            
            # Normalize
            autocorr = autocorr / np.max(autocorr)
            
            # Skip the central peak (self-correlation)
            half_len = len(autocorr) // 2
            autocorr = autocorr[half_len+1:]
            
            # Find peaks that might indicate echo
            peaks = []
            threshold = 0.1
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > threshold and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append((i, autocorr[i]))
            
            # Check if there are suspicious patterns of peaks
            if len(peaks) > 0 and len(peaks) < 10:  # Not too many, not too few
                # Check if the peaks have fairly regular spacing
                if len(peaks) >= 2:
                    distances = [peaks[i+1][0] - peaks[i][0] for i in range(len(peaks)-1)]
                    mean_dist = np.mean(distances)
                    std_dist = np.std(distances)
                    
                    # If spacing is regular, it might be echo hiding
                    if std_dist / mean_dist < 0.2:
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error in echo hiding detection: {e}")
            return False
    
    def _extract_echo_hiding(self, file_path):
        """
        Extract data hidden using echo hiding in audio
        """
        try:
            # Read the audio file
            sample_rate, data = wavfile.read(file_path)
            
            # Handle multi-channel audio
            if len(data.shape) > 1:
                # Use the first channel for simplicity
                data = data[:, 0]
            
            # Calculate autocorrelation
            autocorr = np.correlate(data, data, mode='full')
            
            # Normalize
            autocorr = autocorr / np.max(autocorr)
            
            # Skip the central peak
            half_len = len(autocorr) // 2
            autocorr = autocorr[half_len+1:]
            
            # Find peaks
            peaks = []
            threshold = 0.1
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > threshold and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append((i, autocorr[i]))
            
            # Analyze peaks to decode message
            if len(peaks) == 0:
                return None
                
            # Sort peaks by position
            peaks.sort(key=lambda x: x[0])
            
            # Define delay values for 0 and 1 bits
            # These values depend on the specific implementation
            delay_zero = sample_rate * 0.001  # 1ms
            delay_one = sample_rate * 0.002   # 2ms
            
            # Decode bits based on peak positions
            bits = []
            for pos, _ in peaks:
                if abs(pos - delay_zero) < abs(pos - delay_one):
                    bits.append(0)
                else:
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
            print(f"Error in echo hiding extraction: {e}")
            return None
    
    def _detect_phase_coding(self, file_path):
        """
        Detect if phase coding steganography is used in the audio file
        """
        try:
            # Read the audio file
            sample_rate, data = wavfile.read(file_path)
            
            # Handle multi-channel audio
            if len(data.shape) > 1:
                # Use the first channel for simplicity
                data = data[:, 0]
            
            # Perform FFT to get magnitude and phase
            fft_data = fft(data)
            magnitude = np.abs(fft_data)
            phase = np.angle(fft_data)
            
            # In phase coding, phase values are modified to encode data
            # Check for unusual patterns in phase values
            
            # Calculate phase differences
            phase_diff = np.diff(phase)
            
            # Analyze phase differences
            # In normal audio, phase differences should be somewhat smooth
            # In stego audio, there might be abrupt changes
            
            # Calculate mean and standard deviation of differences
            mean_diff = np.mean(np.abs(phase_diff))
            std_diff = np.std(np.abs(phase_diff))
            
            # Check for unusual ratio of std/mean
            if std_diff / mean_diff > 2.0:
                return True
                
            # Check for suspicious clustering of phase values
            hist, _ = np.histogram(phase, bins=100)
            
            # Calculate entropy of histogram
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Low entropy indicates clustering, which might be suspicious
            if entropy < 4.0:
                return True
                
            return False
            
        except Exception as e:
            print(f"Error in phase coding detection: {e}")
            return False
    
    def _extract_phase_coding(self, file_path):
        """
        Extract data hidden using phase coding in audio
        """
        try:
            # Read the audio file
            sample_rate, data = wavfile.read(file_path)
            
            # Handle multi-channel audio
            if len(data.shape) > 1:
                # Use the first channel for simplicity
                data = data[:, 0]
            
            # Perform FFT
            fft_data = fft(data)
            phase = np.angle(fft_data)
            
            # In phase coding, usually the first few frequency components are used
            # Extract binary data from phase values
            
            # Simplified implementation - extract from first 100 phase values
            bits = []
            for i in range(100):
                # Phase values near 0 might represent 0, and near π might represent 1
                if abs(phase[i]) < np.pi/2:
                    bits.append(0)
                else:
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
            print(f"Error in phase coding extraction: {e}")
            return None
