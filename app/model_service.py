"""
Model service for loading and running inference.
"""
import joblib
import os
from typing import Tuple, Dict
import numpy as np

class ModelService:
    def __init__(self, model_path: str = "model/model_v1.joblib"):
        self.model_path = model_path
        self.model = None
        self.metrics = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and its metrics."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        
        # Try to load metrics
        metrics_path = self.model_path.replace('.joblib', '').replace('model_', 'metrics_') + '.joblib'
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
        else:
            self.metrics = {'version': 'v1', 'accuracy': 0.0}
        
        print(f"Model loaded: {self.model_path}")
        print(f"Version: {self.metrics.get('version', 'unknown')}")
        print(f"Accuracy: {self.metrics.get('accuracy', 0.0):.4f}")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for a single text.
        
        Returns:
            (label, confidence) tuple
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Get prediction and probabilities
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        
        # Confidence is the max probability
        confidence = float(np.max(probabilities))
        
        return prediction, confidence
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts.
        
        Returns:
            List of (label, confidence) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        predictions = self.model.predict(texts)
        probabilities = self.model.predict_proba(texts)
        
        results = []
        for pred, probs in zip(predictions, probabilities):
            confidence = float(np.max(probs))
            results.append((pred, confidence))
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            'version': self.metrics.get('version', 'unknown'),
            'accuracy': self.metrics.get('accuracy', 0.0),
            'model_path': self.model_path
        }

# Global model instance
model_service: ModelService = None

def get_model_service() -> ModelService:
    """Get or create the global model service instance."""
    global model_service
    if model_service is None:
        model_service = ModelService()
    return model_service
