#!/usr/bin/env python3
"""
Audio Steganography Demo Generator for StegBuster
================================================

This script generates demonstration audio files for StegBuster with embedded messages
using different steganography techniques:
1. LSB (Least Significant Bit) steganography in WAV files
2. Echo hiding steganography in WAV files

The generated files can be used to test the detection capabilities of StegBuster.
"""

import os
import numpy as np
import wave
import struct
import math
import random
from scipy.io import wavfile

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
LSB_DIR = os.path.join(AUDIO_DIR, "lsb")
ECHO_DIR = os.path.join(AUDIO_DIR, "echo")

# Ensure directories exist
os.makedirs(LSB_DIR, exist_ok=True)
os.makedirs(ECHO_DIR, exist_ok=True)

# Sample secret messages
SHORT_MESSAGE = "StegBuster can find this hidden message! Crypto Finals 2025"
LONG_MESSAGE = """
This is a longer hidden message that demonstrates the capacity of steganography techniques.
StegBuster is designed to detect various forms of steganography across multiple file formats.
This project showcases both traditional statistical methods and machine learning approaches.
Good luck with your Crypto Finals presentation!
"""

def generate_audio_tone(duration=5, sample_rate=44100, frequencies=[440, 880]):
    """Generate a simple audio tone with multiple frequencies."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a complex tone with multiple frequencies
    signal = np.zeros_like(t)
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)
    
    # Add some harmonics
    for i in range(1, 4):
        harmonic_freq = frequencies[0] * (i + 1)
        signal += (1.0 / (i + 1)) * np.sin(2 * np.pi * harmonic_freq * t)
    
    # Normalize to 16-bit range
    signal = signal / np.max(np.abs(signal))
    signal = signal * 32767
    signal = signal.astype(np.int16)
    
    return signal, sample_rate

def generate_audio_noise(duration=5, sample_rate=44100, amplitude=0.1):
    """Generate white noise audio."""
    samples = int(duration * sample_rate)
    noise = np.random.normal(0, amplitude, samples)
    
    # Convert to int16
    noise = (noise * 32767).astype(np.int16)
    return noise, sample_rate

def text_to_binary(text):
    """Convert text to a binary string."""
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

def embed_lsb_audio(audio_data, message):
    """Embed a message into audio using LSB steganography."""
    # Convert message to binary
    binary_message = text_to_binary(message)
    binary_message += '00000000'  # Add null terminator
    
    # Make a copy of the audio data
    stego_audio = audio_data.copy()
    
    # Check if we can fit the message
    max_bytes = len(audio_data) // 8
    message_bytes = len(binary_message) // 8
    
    if message_bytes > max_bytes:
        raise ValueError(f"Message too large! Max size: {max_bytes} bytes, Message size: {message_bytes} bytes")
    
    # Embed each bit of the message
    for i in range(len(binary_message)):
        if i >= len(stego_audio):
            break
        
        # Clear the LSB
        stego_audio[i] = stego_audio[i] & ~1
        
        # Set the LSB according to the message bit
        if binary_message[i] == '1':
            stego_audio[i] = stego_audio[i] | 1
    
    return stego_audio

def embed_echo_hiding(audio_data, message, sample_rate=44100):
    """Embed a message using echo hiding steganography."""
    # Convert message to binary
    binary_message = text_to_binary(message)
    
    # Parameters for the echo
    delay_one = int(sample_rate * 0.001)  # 1ms delay for bit '1'
    delay_zero = int(sample_rate * 0.002)  # 2ms delay for bit '0'
    alpha = 0.3  # Echo intensity (0 to 1)
    
    # Create a copy of the audio
    stego_audio = audio_data.copy().astype(np.float64)
    
    # Calculate the frame size (how much audio per bit)
    frame_size = len(stego_audio) // len(binary_message)
    
    if frame_size < max(delay_one, delay_zero) * 3:
        raise ValueError("Audio too short for the provided message with these echo parameters")
    
    # Process each bit
    for i, bit in enumerate(binary_message):
        start = i * frame_size
        end = start + frame_size if i < len(binary_message) - 1 else len(stego_audio)
        
        if start >= len(stego_audio):
            break
            
        # Get the current frame
        frame = stego_audio[start:end]
        
        # Create delayed version based on bit value
        delay = delay_one if bit == '1' else delay_zero
        
        # Create the echo
        echo = np.zeros_like(frame)
        echo[delay:] = frame[:-delay] * alpha
        
        # Add the echo to the frame
        stego_audio[start:end] = frame + echo
    
    # Normalize to avoid clipping
    max_val = np.max(np.abs(stego_audio))
    if max_val > 0:
        stego_audio = stego_audio * (32767 / max_val)
    
    # Convert back to int16
    stego_audio = stego_audio.astype(np.int16)
    
    return stego_audio

def save_wav(filename, audio_data, sample_rate):
    """Save audio data to a WAV file."""
    wavfile.write(filename, sample_rate, audio_data)

def main():
    print("Generating steganography demo audio files...")
    
    # === LSB Steganography in WAV ===
    print("Creating LSB steganography examples...")
    
    # Example 1: Simple tone with short message
    audio1, sr1 = generate_audio_tone(duration=5, frequencies=[440, 880])
    audio1_clean_path = os.path.join(LSB_DIR, "tone_original.wav")
    save_wav(audio1_clean_path, audio1, sr1)
    
    audio1_stego = embed_lsb_audio(audio1, SHORT_MESSAGE)
    save_wav(os.path.join(LSB_DIR, "tone_lsb_short.wav"), audio1_stego, sr1)
    
    # Example 2: Noise with longer message
    audio2, sr2 = generate_audio_noise(duration=10)
    audio2_clean_path = os.path.join(LSB_DIR, "noise_original.wav")
    save_wav(audio2_clean_path, audio2, sr2)
    
    audio2_stego = embed_lsb_audio(audio2, LONG_MESSAGE[:400])  # Use part of the long message
    save_wav(os.path.join(LSB_DIR, "noise_lsb_long.wav"), audio2_stego, sr2)
    
    # Example 3: Combined tone+noise with medium message
    audio3, sr3 = generate_audio_tone(duration=7, frequencies=[523, 659])  # Different frequencies
    noise = generate_audio_noise(duration=7, sample_rate=sr3, amplitude=0.05)[0]
    combined_audio = audio3 + noise
    
    audio3_clean_path = os.path.join(LSB_DIR, "combined_original.wav")
    save_wav(audio3_clean_path, combined_audio, sr3)
    
    audio3_stego = embed_lsb_audio(combined_audio, SHORT_MESSAGE + SHORT_MESSAGE)  # Medium length
    save_wav(os.path.join(LSB_DIR, "combined_lsb_medium.wav"), audio3_stego, sr3)
    
    # === Echo Hiding Steganography in WAV ===
    print("Creating echo hiding steganography examples...")
    
    # Example 1: Simple tone with short message using echo hiding
    audio4, sr4 = generate_audio_tone(duration=8, frequencies=[392, 784])
    audio4_clean_path = os.path.join(ECHO_DIR, "tone_original.wav")
    save_wav(audio4_clean_path, audio4, sr4)
    
    audio4_stego = embed_echo_hiding(audio4, SHORT_MESSAGE, sr4)
    save_wav(os.path.join(ECHO_DIR, "tone_echo_short.wav"), audio4_stego, sr4)
    
    # Example 2: Longer audio with medium message
    audio5, sr5 = generate_audio_tone(duration=15, frequencies=[349, 698])
    audio5_clean_path = os.path.join(ECHO_DIR, "long_tone_original.wav")
    save_wav(audio5_clean_path, audio5, sr5)
    
    # Use a medium length message
    audio5_stego = embed_echo_hiding(audio5, SHORT_MESSAGE + SHORT_MESSAGE[:20], sr5)
    save_wav(os.path.join(ECHO_DIR, "long_tone_echo_medium.wav"), audio5_stego, sr5)
    
    print("Audio steganography demo files generated successfully!")
    print(f"LSB examples saved to: {LSB_DIR}")
    print(f"Echo hiding examples saved to: {ECHO_DIR}")

if __name__ == "__main__":
    main()