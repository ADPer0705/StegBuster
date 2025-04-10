import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import logging

def visualize_image_stego(image_path, heatmap, confidence):
    """Generate visualization of stego detection results
    
    Args:
        image_path: Path to the original image
        heatmap: 2D array representing detection heatmap
        confidence: Confidence score (0-1)
        
    Returns:
        str: Path to the saved visualization
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax[0].imshow(img)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        # Heatmap overlay
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        ax[1].imshow(img)
        im = ax[1].imshow(heatmap_resized, alpha=0.6, cmap='jet')
        ax[1].set_title('Steganography Heatmap')
        ax[1].axis('off')
        fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        
        # LSB visualization
        lsb_planes = np.zeros_like(img)
        for i in range(3):  # RGB channels
            lsb_planes[:,:,i] = img[:,:,i] % 2 * 255
        ax[2].imshow(lsb_planes)
        ax[2].set_title('LSB Planes')
        ax[2].axis('off')
        
        # Add confidence score as text
        fig.suptitle(f'Steganography Detection Confidence: {confidence:.2%}', fontsize=16)
        
        # Save visualization
        output_path = image_path + '_stego_analysis.png'
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)  # Close the figure to free memory
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")
        return None

def visualize_comparison(original_path, stego_path, diff_highlight=True):
    """Generate visual comparison between original and stego images
    
    Args:
        original_path: Path to the original image
        stego_path: Path to the stego image
        diff_highlight: Whether to highlight differences
        
    Returns:
        str: Path to the saved comparison
    """
    try:
        # Load images
        original = cv2.imread(original_path)
        stego = cv2.imread(stego_path)
        
        if original is None or stego is None:
            raise ValueError("Could not load images")
            
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        stego = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)
        
        # Resize if dimensions don't match
        if original.shape != stego.shape:
            stego = cv2.resize(stego, (original.shape[1], original.shape[0]))
        
        # Create difference image
        diff = cv2.absdiff(original, stego)
        
        # Enhance difference for visibility
        if diff_highlight:
            diff = cv2.convertScaleAbs(diff * 10)
        
        # Create figure with subplots
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].imshow(original)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(stego)
        ax[1].set_title('Stego Image')
        ax[1].axis('off')
        
        ax[2].imshow(diff)
        ax[2].set_title('Difference (Enhanced)')
        ax[2].axis('off')
        
        # Save comparison
        output_path = original_path + '_comparison.png'
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error in comparison visualization: {str(e)}")
        return None
