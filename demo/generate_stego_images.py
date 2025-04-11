#!/usr/bin/env python3
"""
Image Steganography Demo Generator for StegBuster
================================================

This script generates demonstration image files for StegBuster with embedded messages
using different steganography techniques:
1. LSB (Least Significant Bit) steganography in PNG images
2. DCT (Discrete Cosine Transform) steganography in JPEG images

The generated files can be used to test the detection capabilities of StegBuster.
"""

import os
import numpy as np
from PIL import Image
import random
import string
import math
import cv2
from scipy.fftpack import dct, idct

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LSB_DIR = os.path.join(BASE_DIR, "images", "lsb")
DCT_DIR = os.path.join(BASE_DIR, "images", "dct")

# Ensure directories exist
os.makedirs(LSB_DIR, exist_ok=True)
os.makedirs(DCT_DIR, exist_ok=True)

# Sample secret messages
SHORT_MESSAGE = "StegBuster can find this hidden message! Crypto Finals 2025"
LONG_MESSAGE = """
This is a longer hidden message that demonstrates the capacity of steganography techniques.
StegBuster is designed to detect various forms of steganography across multiple file formats.
This project showcases both traditional statistical methods and machine learning approaches.
Good luck with your Crypto Finals presentation!
"""

def generate_random_image(width=300, height=200):
    """Generate a random colorful image."""
    # Create gradients and patterns for more realistic looking images
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    
    # Create RGB channels with different patterns
    r = np.sin(xv * 10) * np.cos(yv * 10) * 255
    g = np.sin(xv * 5) * np.cos(yv * 15) * 255
    b = np.sin(xv * 15) * np.cos(yv * 5) * 255
    
    # Add some noise for texture
    r += np.random.normal(0, 20, (height, width))
    g += np.random.normal(0, 20, (height, width))
    b += np.random.normal(0, 20, (height, width))
    
    # Clip values to valid range
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    
    # Combine channels
    img = np.dstack((r, g, b))
    return Image.fromarray(img)

def text_to_binary(text):
    """Convert text to a binary string."""
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

def embed_lsb(image, message):
    """Embed a message into an image using LSB steganography."""
    # Convert message to binary
    binary_message = text_to_binary(message)
    binary_message += '00000000'  # Add null terminator
    
    # Get image data as numpy array
    img_array = np.array(image)
    height, width, channels = img_array.shape
    
    # Calculate maximum message length that can be embedded
    max_bytes = height * width * channels // 8
    message_bytes = len(binary_message) // 8
    
    if message_bytes > max_bytes:
        raise ValueError(f"Message too large! Max size: {max_bytes} bytes, Message size: {message_bytes} bytes")
    
    # Make a copy to avoid modifying original array
    stego_array = img_array.copy()
    
    # Flatten the array for easier processing
    flattened = stego_array.flatten()
    
    # Embed each bit of the message
    for i in range(len(binary_message)):
        if i >= len(flattened):
            break
        
        # Replace the LSB of the pixel value with our message bit
        if binary_message[i] == '1':
            flattened[i] = (flattened[i] & 0xFE) | 1  # Set LSB to 1
        else:
            flattened[i] = (flattened[i] & 0xFE)  # Set LSB to 0
    
    # Reshape back to original dimensions
    stego_array = flattened.reshape(height, width, channels)
    return Image.fromarray(stego_array.astype(np.uint8))

def embed_dct(image_path, output_path, message, block_size=8):
    """Embed a message into a JPEG image using DCT steganography."""
    # Convert message to binary
    binary_message = text_to_binary(message)
    binary_message += '00000000'  # Add null terminator
    
    # Read the image with OpenCV (needed for DCT operations)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Get dimensions
    height, width = img_yuv.shape[0], img_yuv.shape[1]
    
    # Process only the Y channel (luminance) for simplicity
    y_channel = img_yuv[:,:,0]
    
    # Number of blocks we can use
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    total_blocks = num_blocks_h * num_blocks_w
    
    # Maximum message length
    max_bits = total_blocks  # One bit per 8x8 block
    
    if len(binary_message) > max_bits:
        raise ValueError(f"Message too long! Max bits: {max_bits}, Message bits: {len(binary_message)}")
    
    # Coefficients to modify (middle frequency coefficients are good choices)
    target_coeff = (4, 3)  # Middle frequency coefficient position
    
    # Strength of embedding (higher values more resistant to compression but more visible)
    alpha = 25
    
    # Process each block
    bit_index = 0
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            if bit_index >= len(binary_message):
                break
                
            # Extract block
            block = y_channel[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size].astype(float)
            
            # Apply DCT to the block
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Embed data - modify the selected coefficient based on message bit
            if binary_message[bit_index] == '1':
                # Make coefficient magnitude greater than alpha
                if dct_block[target_coeff] > 0:
                    dct_block[target_coeff] = max(dct_block[target_coeff], alpha)
                else:
                    dct_block[target_coeff] = min(dct_block[target_coeff], -alpha)
            else:
                # Make coefficient magnitude less than alpha
                if abs(dct_block[target_coeff]) > alpha:
                    dct_block[target_coeff] = alpha * np.sign(dct_block[target_coeff])
            
            # Apply inverse DCT
            block = idct(idct(dct_block, norm='ortho').T, norm='ortho').T
            
            # Put the block back
            y_channel[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block
            
            bit_index += 1
    
    # Clip values to valid range
    y_channel = np.clip(y_channel, 0, 255).astype(np.uint8)
    img_yuv[:,:,0] = y_channel
    
    # Convert back to BGR and save
    img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(output_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return True

def main():
    print("Generating steganography demo images...")
    
    # === LSB Steganography in PNG ===
    # Generate a few different images with different content
    print("Creating LSB steganography examples...")
    
    # Example 1: Simple gradient with short message
    img1 = generate_random_image(400, 300)
    img1_clean_path = os.path.join(LSB_DIR, "gradient_original.png")
    img1.save(img1_clean_path)
    img1_stego = embed_lsb(img1, SHORT_MESSAGE)
    img1_stego.save(os.path.join(LSB_DIR, "gradient_lsb_short.png"))
    
    # Example 2: Pattern image with longer message
    img2 = generate_random_image(600, 400)
    img2_clean_path = os.path.join(LSB_DIR, "pattern_original.png")
    img2.save(img2_clean_path)
    img2_stego = embed_lsb(img2, LONG_MESSAGE)
    img2_stego.save(os.path.join(LSB_DIR, "pattern_lsb_long.png"))
    
    # Example 3: Colorful image with medium message
    img3 = generate_random_image(500, 500)
    img3_clean_path = os.path.join(LSB_DIR, "colorful_original.png")
    img3.save(img3_clean_path)
    img3_stego = embed_lsb(img3, SHORT_MESSAGE + SHORT_MESSAGE)  # Medium length
    img3_stego.save(os.path.join(LSB_DIR, "colorful_lsb_medium.png"))
    
    # === DCT Steganography in JPEG ===
    print("Creating DCT steganography examples...")
    
    # First save as PNG for DCT embedding
    img4 = generate_random_image(512, 384)
    img4_clean_path = os.path.join(DCT_DIR, "dct_original.png")
    img4.save(img4_clean_path)
    
    # Convert to JPEG (clean version)
    img4_jpeg_clean = os.path.join(DCT_DIR, "dct_original.jpg")
    img4_jpeg_stego = os.path.join(DCT_DIR, "dct_hidden_short.jpg")
    Image.open(img4_clean_path).convert('RGB').save(img4_jpeg_clean, quality=95)
    
    # Embed using DCT
    embed_dct(img4_clean_path, img4_jpeg_stego, SHORT_MESSAGE)
    
    # Another DCT example with longer message
    img5 = generate_random_image(640, 480)
    img5_clean_path = os.path.join(DCT_DIR, "dct_original2.png")
    img5.save(img5_clean_path)
    
    img5_jpeg_clean = os.path.join(DCT_DIR, "dct_original2.jpg")
    img5_jpeg_stego = os.path.join(DCT_DIR, "dct_hidden_long.jpg")
    Image.open(img5_clean_path).convert('RGB').save(img5_jpeg_clean, quality=95)
    
    # Embed using DCT (with longer message)
    embed_dct(img5_clean_path, img5_jpeg_stego, LONG_MESSAGE[:200])  # Using part of long message
    
    print("Image steganography demo files generated successfully!")
    print(f"LSB examples saved to: {LSB_DIR}")
    print(f"DCT examples saved to: {DCT_DIR}")
    
if __name__ == "__main__":
    main()