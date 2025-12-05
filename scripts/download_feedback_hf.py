"""
Script to download latest feedback from HuggingFace Datasets.
"""
import os
import pandas as pd
from huggingface_hub import hf_hub_download

def download_feedback(
    repo_name: str,
    hf_token: str,
    output_path: str = "data/feedback_latest.csv"
):
    """
    Download feedback CSV from HuggingFace Datasets repository.
    
    Args:
        repo_name: HF dataset repo (username/repo-name)
        hf_token: HuggingFace access token
        output_path: Where to save the downloaded file
    
    Returns:
        Path to downloaded file, or None if failed
    """
    try:
        print(f"Downloading feedback from {repo_name}...")
        
        # Download the feedback CSV file
        local_path = hf_hub_download(
            repo_id=repo_name,
            filename="data/feedback_buffer.csv",
            repo_type="dataset",
            token=hf_token,
            cache_dir=".hf_cache"
        )
        
        # Read and save to output location
        df = pd.read_csv(local_path)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to output path
        df.to_csv(output_path, index=False)
        
        print(f"✓ Downloaded {len(df)} feedback samples to {output_path}")
        print(f"  - User corrections: {len(df[df['feedback_type'] == 'user_correction'])}")
        print(f"  - Low confidence: {len(df[df['feedback_type'] == 'low_confidence'])}")
        
        return output_path
        
    except Exception as e:
        print(f"Failed to download feedback: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USERNAME = os.getenv("HF_USERNAME")
    
    if not HF_TOKEN or not HF_USERNAME:
        print("Error: HF_TOKEN and HF_USERNAME must be set")
        sys.exit(1)
    
    repo_name = f"{HF_USERNAME}/sentiment-analysis-data"
    
    download_feedback(repo_name, HF_TOKEN)
