"""
FastAPI application for sentiment analysis.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, ModelInfoResponse,
    FeedbackRequest, FeedbackResponse
)
from app.model_service import get_model_service
from app.feedback_service import get_feedback_service
from app.config import API_TITLE, API_VERSION, API_DESCRIPTION, MODEL_VERSION
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global counters (will be replaced with proper persistence in Week 5)
prediction_count = 0
feedback_count = 0

@app.on_event("startup")
async def startup_event():
    """Load model and feedback service on startup."""
    try:
        get_model_service()
        get_feedback_service()
        print("API startup complete")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Sentiment Analysis Self-Learning API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        model_service = get_model_service()
        model_info = model_service.get_model_info()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version=model_info['version']
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version="none"
        )

@app.post("/v1/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict sentiment for a single text.
    
    Returns the predicted label (positive/negative) and confidence score.
    If confidence is low, automatically logs the sample for later review.
    """
    global prediction_count
    
    try:
        model_service = get_model_service()
        feedback_service = get_feedback_service()
        
        label, confidence = model_service.predict(request.text)
        
        # Auto-log low-confidence predictions
        if feedback_service.should_log_low_confidence(confidence):
            feedback_service.add_low_confidence_sample(
                text=request.text,
                predicted_label=label,
                confidence=confidence
            )
        
        prediction_count += 1
        
        return PredictResponse(
            text=request.text,
            label=label,
            confidence=confidence,
            model_version=MODEL_VERSION
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/v1/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Predict sentiment for multiple texts in a single request.
    
    More efficient than calling /v1/predict multiple times.
    """
    global prediction_count
    
    try:
        model_service = get_model_service()
        results = model_service.predict_batch(request.texts)
        
        predictions = [
            PredictResponse(
                text=text,
                label=label,
                confidence=confidence,
                model_version=MODEL_VERSION
            )
            for text, (label, confidence) in zip(request.texts, results)
        ]
        
        prediction_count += len(predictions)
        
        return BatchPredictResponse(
            predictions=predictions,
            total=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/v1/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback when the model's prediction was incorrect.
    
    This helps the model learn from its mistakes during weekly retraining.
    """
    global feedback_count
    
    try:
        feedback_service = get_feedback_service()
        
        feedback_id = feedback_service.add_user_feedback(
            text=request.text,
            predicted_label=request.predicted_label,
            correct_label=request.correct_label,
            confidence=request.confidence
        )
        
        feedback_count += 1
        
        return FeedbackResponse(
            message="Thank you for your feedback! This will help improve the model.",
            feedback_id=feedback_id,
            stored=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")

@app.get("/v1/model_info", response_model=ModelInfoResponse, tags=["Info"])
async def get_model_info():
    """Get information about the current model."""
    try:
        model_service = get_model_service()
        feedback_service = get_feedback_service()
        
        model_info = model_service.get_model_info()
        feedback_stats = feedback_service.get_feedback_stats()
        
        return ModelInfoResponse(
            version=model_info['version'],
            accuracy=model_info['accuracy'],
            total_predictions=prediction_count,
            total_feedback=feedback_stats['total']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
