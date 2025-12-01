"""
Model service for loading and running inference.
"""
import joblib
import os
from typing import Tuple, Dict, Optional
import numpy as np
from app.config import LOAD_MODEL_FROM_HUB, HF_MODEL_REPO, HF_TOKEN, MODEL_VERSION

class ModelService:
    def __init__(self, model_path: str = "model/model_v1.joblib", load_from_hub: bool = None):
        self.model_path = model_path
        self.model = None
        self.metrics = None
        self.load_from_hub = load_from_hub if load_from_hub is not None else LOAD_MODEL_FROM_HUB
        self.load_model()
    
    def load_model(self):
        """Load the trained model and its metrics from local disk or HuggingFace Hub."""
        
        if self.load_from_hub and HF_MODEL_REPO:
            self._load_from_huggingface()
        else:
            self._load_from_local()
        
        print(f"Model loaded: {self.model_path}")
        print(f"Version: {self.metrics.get('version', 'unknown')}")
        print(f"Accuracy: {self.metrics.get('accuracy', 0.0):.4f}")
    
    def _load_from_local(self):
        """Load model from local disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        
        # Try to load metrics
        metrics_path = self.model_path.replace('.joblib', '').replace('model_', 'metrics_') + '.joblib'
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
        else:
            self.metrics = {'version': MODEL_VERSION, 'accuracy': 0.0}
        
        print(f"Loaded from local disk: {self.model_path}")
    
    def _load_from_huggingface(self):
        """Load model from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
            
            model_filename = os.path.basename(self.model_path)
            metrics_filename = model_filename.replace('model_', 'metrics_')
            
            print(f"Downloading model from HuggingFace Hub: {HF_MODEL_REPO}")
            
            # Download model
            model_local_path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=model_filename,
                token=HF_TOKEN,
                cache_dir=".hf_cache"
            )
            
            self.model = joblib.load(model_local_path)
            print(f"Model downloaded and cached: {model_local_path}")
            
            # Try to download metrics
            try:
                metrics_local_path = hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    filename=metrics_filename,
                    token=HF_TOKEN,
                    cache_dir=".hf_cache"
                )
                self.metrics = joblib.load(metrics_local_path)
            except:
                self.metrics = {'version': MODEL_VERSION, 'accuracy': 0.0}
                print(f"Metrics file not found, using defaults")
            
        except Exception as e:
            print(f"Failed to load from HuggingFace Hub: {e}")
            print(f"Falling back to local loading...")
            self._load_from_local()
    
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
