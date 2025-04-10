import numpy as np
import logging
import os
from utils.model_manager import load_model, preprocess_image

class ImageMLDetector:
    """Machine Learning based detector for image steganography"""
    
    def __init__(self):
        self.model = None
        self.model_name = "image_basic"  # Default model
        self.confidence_threshold = 0.7  # Default confidence threshold
        
    def set_model(self, model_name):
        """Set the model to use"""
        self.model_name = model_name
        
    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold"""
        self.confidence_threshold = threshold
        
    def load_model(self, force_download=False):
        """Load the ML model"""
        try:
            self.model = load_model(self.model_name, force_download)
            return True
        except Exception as e:
            logging.error(f"Error loading ML model: {str(e)}")
            return False
            
    def detect(self, file_path, visualize=False):
        """
        Detect if steganography is present in the image using ML
        
        Args:
            file_path: Path to the image file
            visualize: Whether to generate visualization
            
        Returns:
            tuple: (is_stego_likely, confidence, method, visualization_path)
        """
        # Make sure model is loaded
        if self.model is None:
            if not self.load_model():
                return False, 0.0, None, None
                
        try:
            # Preprocess the image
            input_data = preprocess_image(file_path)
            
            # Get input and output details
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            
            # Set input tensor
            self.model.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            self.model.invoke()
            
            # Get output tensor
            output_data = self.model.get_tensor(output_details[0]['index'])
            
            # Process results
            # Assuming output is [probability_clean, probability_stego]
            confidence = float(output_data[0][1])
            is_stego_likely = confidence >= self.confidence_threshold
            
            # Generate heatmap for visualization
            heatmap = None
            visualization_path = None
            
            if visualize:
                try:
                    from visualizers.image_visualizer import visualize_image_stego
                    import cv2
                    
                    # Simple heatmap for demonstration
                    # In a real implementation, you would use techniques like Grad-CAM
                    # to generate a proper attention heatmap
                    img = cv2.imread(file_path)
                    height, width = img.shape[:2]
                    heatmap = np.zeros((height//16, width//16), dtype=np.float32)
                    
                    # Example: random heatmap for demonstration
                    # In practice, this would come from model activations
                    for i in range(heatmap.shape[0]):
                        for j in range(heatmap.shape[1]):
                            heatmap[i, j] = np.random.random() * confidence
                    
                    visualization_path = visualize_image_stego(file_path, heatmap, confidence)
                except Exception as e:
                    logging.error(f"Error generating visualization: {str(e)}")
            
            # Determine possible method
            method = "Deep Learning"
            if confidence > 0.9:
                method = "Advanced ML (high confidence)"
            
            return is_stego_likely, confidence, method, visualization_path
            
        except Exception as e:
            logging.error(f"Error in ML steganalysis: {str(e)}")
            return False, 0.0, None, None
            
    def analyze_embedding_method(self, file_path):
        """
        Analyze the image to determine the likely embedding method
        
        Args:
            file_path: Path to the image file
            
        Returns:
            dict: Analysis results
        """
        # This would be implemented with a more advanced model
        # that can classify different steganography methods
        
        methods = {
            "LSB": 0.0,
            "DCT": 0.0,
            "F5": 0.0,
            "OutGuess": 0.0,
            "Other": 0.0
        }
        
        if self.model is None:
            if not self.load_model():
                return methods
                
        try:
            # For demonstration, return LSB as the most likely method
            # In a real implementation, this would be determined by the model
            methods["LSB"] = 0.85
            methods["DCT"] = 0.10
            methods["Other"] = 0.05
            
            return methods
        except Exception as e:
            logging.error(f"Error analyzing embedding method: {str(e)}")
            return methods
