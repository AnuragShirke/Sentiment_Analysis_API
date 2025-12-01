"""
Configuration settings for the sentiment analysis API.
"""
import os
from typing import Optional

# Model settings
MODEL_VERSION = "v1"
CONFIDENCE_THRESHOLD = 0.55  # Below this, we log as "low confidence"

# Paths
MODEL_DIR = "model"
DATA_DIR = "data"
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_buffer.csv")

# HuggingFace settings
HF_USERNAME: Optional[str] = os.getenv("HF_USERNAME")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

# Repository names (will be set once user provides HF_USERNAME)
HF_MODEL_REPO: Optional[str] = f"{HF_USERNAME}/sentiment-analysis-model" if HF_USERNAME else None
HF_DATASET_REPO: Optional[str] = f"{HF_USERNAME}/sentiment-analysis-data" if HF_USERNAME else None

# Toggle for loading from HF Hub vs local disk
# Set to True in production (HF Spaces), False for local development
LOAD_MODEL_FROM_HUB: bool = os.getenv("LOAD_MODEL_FROM_HUB", "false").lower() == "true"

# API settings
API_TITLE = "Sentiment Analysis Self-Learning API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
A production-grade sentiment analysis API that continuously learns from user feedback.

**Features:**
- Real-time sentiment prediction (positive/negative/neutral)
- Confidence scoring
- Low-confidence sample logging
- User feedback collection
- Weekly automated retraining
"""
