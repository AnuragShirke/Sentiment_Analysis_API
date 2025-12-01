"""
Feedback service for collecting user corrections and low-confidence samples.
"""
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Optional
from app.config import CONFIDENCE_THRESHOLD, DATA_DIR, FEEDBACK_FILE

class FeedbackService:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Initialize feedback buffer if it doesn't exist
        if not os.path.exists(FEEDBACK_FILE):
            self._create_feedback_file()
    
    def _create_feedback_file(self):
        """Create empty feedback CSV with proper columns."""
        df = pd.DataFrame(columns=[
            'text', 
            'predicted_label', 
            'correct_label',
            'confidence',
            'timestamp',
            'feedback_type'  # 'user_correction' or 'low_confidence'
        ])
        df.to_csv(FEEDBACK_FILE, index=False)
        print(f"Created feedback file: {FEEDBACK_FILE}")
    
    def add_user_feedback(
        self, 
        text: str, 
        predicted_label: str, 
        correct_label: str,
        confidence: float
    ) -> int:
        """
        Store user-corrected label.
        
        Returns:
            feedback_id: Row number of the stored feedback
        """
        feedback_data = {
            'text': text,
            'predicted_label': predicted_label,
            'correct_label': correct_label,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'user_correction'
        }
        
        # Append to CSV
        df = pd.DataFrame([feedback_data])
        df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        
        # Get row count as feedback ID
        feedback_id = len(pd.read_csv(FEEDBACK_FILE))
        
        return feedback_id
    
    def add_low_confidence_sample(
        self,
        text: str,
        predicted_label: str,
        confidence: float
    ) -> bool:
        """
        Store low-confidence prediction for later review/labeling.
        
        Returns:
            True if stored, False if already exists
        """
        # Check if this text already exists
        if os.path.exists(FEEDBACK_FILE):
            existing_df = pd.read_csv(FEEDBACK_FILE)
            if text in existing_df['text'].values:
                return False  # Don't store duplicates
        
        feedback_data = {
            'text': text,
            'predicted_label': predicted_label,
            'correct_label': None,  # Unknown, needs labeling later
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'low_confidence'
        }
        
        df = pd.DataFrame([feedback_data])
        df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        
        return True
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about collected feedback."""
        if not os.path.exists(FEEDBACK_FILE):
            return {
                'total': 0,
                'user_corrections': 0,
                'low_confidence': 0
            }
        
        df = pd.read_csv(FEEDBACK_FILE)
        
        return {
            'total': len(df),
            'user_corrections': len(df[df['feedback_type'] == 'user_correction']),
            'low_confidence': len(df[df['feedback_type'] == 'low_confidence'])
        }
    
    def sync_to_hub(self) -> bool:
        """
        Sync feedback data to HuggingFace Datasets Hub.
        
        Returns:
            True if successful, False otherwise
        """
        from app.config import HF_DATASET_REPO, HF_TOKEN
        
        if not HF_DATASET_REPO or not HF_TOKEN:
            print("HuggingFace credentials not set, skipping sync")
            return False
        
        if not os.path.exists(FEEDBACK_FILE):
            print("No feedback file to sync")
            return False
        
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        if len(feedback_df) == 0:
            print("Feedback file is empty, skipping sync")
            return False
        
        try:
            from datasets import Dataset, load_dataset
            
            print(f"Syncing {len(feedback_df)} feedback samples to HuggingFace...")
            
            # Try to load existing dataset
            try:
                dataset_dict = load_dataset(HF_DATASET_REPO, token=HF_TOKEN)
            except:
                # Dataset doesn't exist yet, will be created by push_dataset_hf.py
                print("Dataset not found on HF Hub. Run scripts/push_dataset_hf.py first.")
                return False
            
            # Update feedback split
            dataset_dict['feedback'] = Dataset.from_pandas(feedback_df)
            
            # Push to hub
            dataset_dict.push_to_hub(
                repo_id=HF_DATASET_REPO,
                token=HF_TOKEN,
                commit_message=f"Update feedback ({len(feedback_df)} samples)"
            )
            
            print(f"Feedback synced successfully to {HF_DATASET_REPO}")
            return True
            
        except Exception as e:
            print(f"Failed to sync feedback to HuggingFace: {e}")
            return False
    
    def should_log_low_confidence(self, confidence: float) -> bool:
        """Check if prediction confidence is below threshold."""
        return confidence < CONFIDENCE_THRESHOLD

# Global feedback service instance
feedback_service: Optional[FeedbackService] = None

def get_feedback_service() -> FeedbackService:
    """Get or create the global feedback service instance."""
    global feedback_service
    if feedback_service is None:
        feedback_service = FeedbackService()
    return feedback_service
