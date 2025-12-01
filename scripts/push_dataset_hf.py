"""
Script to push dataset to HuggingFace Datasets Hub.
"""
import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, create_repo

def push_dataset_to_hub(
    train_path: str,
    test_path: str,
    repo_name: str,
    hf_token: str,
    feedback_path: str = None
):
    """
    Push datasets to HuggingFace Datasets Hub.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        repo_name: HF repo name (format: username/repo-name)
        hf_token: HuggingFace access token
        feedback_path: Optional path to feedback CSV
    """
    print(f"Preparing datasets for upload...")
    
    # Load CSVs
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Convert to HF Dataset format (train and test only)
    # Note: Feedback has different schema, so we upload it separately
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'test': Dataset.from_pandas(test_df)
    })
    
    # Create repo
    try:
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            repo_type="dataset",
            exist_ok=True
        )
        print(f"Repository ready: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Push train/test to hub
    print("Uploading train/test splits to HuggingFace...")
    dataset_dict.push_to_hub(
        repo_id=repo_name,
        token=hf_token,
        commit_message="Upload sentiment analysis dataset (train/test)"
    )
    
    print(f"Dataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_name}")
    
    # Now upload feedback separately if it exists
    if feedback_path and os.path.exists(feedback_path):
        feedback_df = pd.read_csv(feedback_path)
        if len(feedback_df) > 0:
            print(f"\nUploading feedback split ({len(feedback_df)} samples)...")
            update_feedback_only(feedback_path, repo_name, hf_token)
    
    return f"https://huggingface.co/datasets/{repo_name}"

def update_feedback_only(
    feedback_path: str,
    repo_name: str,
    hf_token: str
):
    """
    Upload feedback CSV directly to the dataset repository.
    
    This uploads the raw CSV file instead of using DatasetDict to avoid
    schema mismatch errors (feedback has different columns than train/test).
    
    Args:
        feedback_path: Path to feedback CSV
        repo_name: HF repo name
        hf_token: HuggingFace access token
    """
    if not os.path.exists(feedback_path):
        print(f"No feedback file found at {feedback_path}")
        return
    
    feedback_df = pd.read_csv(feedback_path)
    if len(feedback_df) == 0:
        print("Feedback file is empty, skipping upload")
        return
    
    print(f"Uploading feedback CSV ({len(feedback_df)} samples)...")
    
    try:
        from huggingface_hub import upload_file
        
        # Upload the raw CSV file to the data/ directory in the repo
        upload_file(
            path_or_fileobj=feedback_path,
            path_in_repo="data/feedback_buffer.csv",
            repo_id=repo_name,
            repo_type="dataset",
            token=hf_token,
            commit_message=f"Update feedback data ({len(feedback_df)} samples)"
        )
        
        print("Feedback CSV uploaded successfully!")
        print(f"View at: https://huggingface.co/datasets/{repo_name}/tree/main/data")
        
    except Exception as e:
        print(f"Failed to upload feedback: {e}")

if __name__ == "__main__":
    import sys
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USERNAME = os.getenv("HF_USERNAME")
    
    if not HF_TOKEN or not HF_USERNAME:
        print("Error: HF_TOKEN and HF_USERNAME must be set")
        sys.exit(1)
    
    # Default paths
    train_path = "data/base_train.csv"
    test_path = "data/base_test.csv"
    feedback_path = "data/feedback_buffer.csv"
    repo_name = f"{HF_USERNAME}/sentiment-analysis-data"
    
    # Check if we're just updating feedback
    if len(sys.argv) > 1 and sys.argv[1] == "--feedback-only":
        update_feedback_only(feedback_path, repo_name, HF_TOKEN)
    else:
        push_dataset_to_hub(
            train_path=train_path,
            test_path=test_path,
            repo_name=repo_name,
            hf_token=HF_TOKEN,
            feedback_path=feedback_path
        )
