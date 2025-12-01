"""
Script to push the trained model to HuggingFace Hub.
"""
import os
import joblib
from huggingface_hub import HfApi, create_repo, upload_file
from pathlib import Path

def push_model_to_hub(
    model_path: str,
    repo_name: str,
    hf_token: str,
    commit_message: str = "Upload model"
):
    """
    Push a trained model to HuggingFace Model Hub.
    
    Args:
        model_path: Path to the .joblib model file
        repo_name: HF repo name (format: username/repo-name)
        hf_token: HuggingFace access token
        commit_message: Commit message for this upload
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Pushing model to HuggingFace Hub...")
    print(f"  Model: {model_path}")
    print(f"  Repo: {repo_name}")
    
    # Initialize API
    api = HfApi(token=hf_token)
    
    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            repo_type="model",
            exist_ok=True
        )
        print(f"Repository ready: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload the model file
    model_filename = Path(model_path).name
    
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_filename,
        repo_id=repo_name,
        repo_type="model",
        commit_message=commit_message,
        token=hf_token
    )
    
    print(f"Model uploaded successfully!")
    print(f"View at: https://huggingface.co/{repo_name}")
    
    # Also upload metrics if they exist
    metrics_path = model_path.replace('model_', 'metrics_')
    if os.path.exists(metrics_path):
        metrics_filename = Path(metrics_path).name
        api.upload_file(
            path_or_fileobj=metrics_path,
            path_in_repo=metrics_filename,
            repo_id=repo_name,
            repo_type="model",
            commit_message=f"{commit_message} (with metrics)",
            token=hf_token
        )
        print(f"Metrics uploaded: {metrics_filename}")
    
    return f"https://huggingface.co/{repo_name}"

if __name__ == "__main__":
    import sys
    
    # Get credentials from environment or arguments
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USERNAME = os.getenv("HF_USERNAME")
    
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set")
        print("Set it with: export HF_TOKEN='your_token_here'")
        sys.exit(1)
    
    if not HF_USERNAME:
        print("Error: HF_USERNAME environment variable not set")
        print("Set it with: export HF_USERNAME='your_username'")
        sys.exit(1)
    
    # Default values
    model_path = "model/model_v1.joblib"
    repo_name = f"{HF_USERNAME}/sentiment-analysis-model"
    
    # Allow overrides from command line
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        repo_name = sys.argv[2]
    
    push_model_to_hub(
        model_path=model_path,
        repo_name=repo_name,
        hf_token=HF_TOKEN,
        commit_message=f"Upload {Path(model_path).stem}"
    )
