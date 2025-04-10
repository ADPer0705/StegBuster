import os
import requests
import hashlib
import logging
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
MODEL_REGISTRY = {
    'image_basic': {
        'url': 'https://github.com/username/stegbuster-models/releases/download/v1.0/image_basic.tflite',
        'hash': '0123456789abcdef0123456789abcdef',
        'size': '5MB'
    },
    'image_advanced': {
        'url': 'https://github.com/username/stegbuster-models/releases/download/v1.0/image_advanced.tflite',
        'hash': '1123456789abcdef0123456789abcdef',
        'size': '25MB'
    }
    # Additional models can be added here
}

def load_model(model_name, force_download=False):
    """Load a model, downloading if necessary"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODEL_REGISTRY[model_name]
    model_path = os.path.join(MODEL_DIR, f"{model_name}.tflite")
    
    # Download if needed
    if not os.path.exists(model_path) or force_download:
        os.makedirs(MODEL_DIR, exist_ok=True)
        logging.info(f"Downloading {model_name} model ({model_info['size']})...")
        
        try:
            response = requests.get(model_info['url'])
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            with open(model_path, 'wb') as f:
                f.write(response.content)
            
            # Verify hash
            file_hash = hashlib.md5(response.content).hexdigest()
            if file_hash != model_info['hash']:
                os.remove(model_path)
                raise ValueError(f"Downloaded model has incorrect hash: {file_hash}")
                
            logging.info(f"Model downloaded successfully")
        except Exception as e:
            logging.error(f"Error downloading model: {str(e)}")
            if os.path.exists(model_path):
                os.remove(model_path)
            raise
    
    # Load and return the model
    try:
        import tensorflow as tf
        # Check if TensorFlow is available
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except ImportError:
        logging.error("TensorFlow not installed. Install with 'pip install tensorflow'")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess an image for ML model input"""
    try:
        import cv2
        
        # Load and resize image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img = cv2.resize(img, target_size)
        
        # Convert to RGB if needed (models typically expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")
        raise
