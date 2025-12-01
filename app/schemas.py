"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Literal

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I absolutely loved this movie! Best film I've seen all year."
            }
        }

class PredictResponse(BaseModel):
    text: str
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I absolutely loved this movie!",
                "label": "positive",
                "confidence": 0.94,
                "model_version": "v1"
            }
        }

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "I love this product!",
                    "Terrible experience, very disappointed.",
                    "It's okay, nothing special."
                ]
            }
        }

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    total: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

class ModelInfoResponse(BaseModel):
    version: str
    accuracy: float
    total_predictions: int
    total_feedback: int

class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    predicted_label: Literal["positive", "negative", "neutral"]
    correct_label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was okay, nothing special.",
                "predicted_label": "positive",
                "correct_label": "neutral",
                "confidence": 0.62
            }
        }

class FeedbackResponse(BaseModel):
    message: str
    feedback_id: int
    stored: bool

class LowConfidenceSample(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    timestamp: str

