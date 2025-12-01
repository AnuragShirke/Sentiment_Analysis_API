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

# HuggingFace settings (to be filled later)
HF_USERNAME: Optional[str] = os.getenv("HF_USERNAME")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
HF_MODEL_REPO: Optional[str] = None  # Will be set in Week 3
HF_DATASET_REPO: Optional[str] = None  # Will be set in Week 3

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
