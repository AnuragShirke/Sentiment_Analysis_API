"""
Automated retraining script for weekly model updates.

This script:
1. Downloads latest feedback from HuggingFace Datasets
2. Merges feedback with base training data
3. Trains a new model version
4. Evaluates against held-out test set
5. Compares to previous model
6. Uploads to HF Hub if improved
"""
import os
import sys
import pandas as pd
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.evaluate import evaluate_model, compare_models, print_evaluation_summary, load_metrics
from scripts.download_feedback_hf import download_feedback
from scripts.push_model_hf import push_model_to_hub

# Configuration
MIN_FEEDBACK_SAMPLES = 10
IMPROVEMENT_THRESHOLD = 0.001  # 0.1% improvement required

def get_next_version(hf_repo: str, hf_token: str) -> str:
    """
    Determine the next model version number.
   
    Args:
        hf_repo: HuggingFace model repository
        hf_token: HF access token
    
    Returns:
        Next version string (e.g., "v2", "v3")
    """
    try:
        from huggingface_hub import list_repo_files
        
        files = list_repo_files(hf_repo, token=hf_token)
        model_files = [f for f in files if f.startswith('model_v') and f.endswith('.joblib')]
        
        if not model_files:
            return "v1"
        
        # Extract version numbers
        versions = []
        for f in model_files:
            try:
                v_num = int(f.replace('model_v', '').replace('.joblib', ''))
                versions.append(v_num)
            except:
                continue
        
        if versions:
            next_v = max(versions) + 1
            return f"v{next_v}"
        else:
            return "v1"
            
    except Exception as e:
        print(f"Failed to get next version from HF Hub: {e}")
        # Fallback: check local directory
        try:
            local_models = [f for f in os.listdir('model') if f.startswith('model_v')]
            if local_models:
                versions = [int(f.replace('model_v', '').replace('.joblib', '')) for f in local_models]
                next_v = max(versions) + 1
                return f"v{next_v}"
        except:
            pass
        
        return "v2"  # Default if all else fails

def download_base_dataset(hf_dataset_repo: str, hf_token: str) -> tuple:
    """
    Download base training and test data from HuggingFace Datasets.
    
    Args:
        hf_dataset_repo: HuggingFace dataset repository
        hf_token: HF access token
    
    Returns:
        Tuple of (train_path, test_path)
    """
    try:
        from datasets import load_dataset
        
        print("Downloading base dataset from HuggingFace...")
        dataset = load_dataset(hf_dataset_repo, token=hf_token)
        
        # Convert to pandas and save locally
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV
        train_path = 'data/base_train.csv'
        test_path = 'data/base_test.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"✓ Downloaded {len(train_df)} training samples")
        print(f"✓ Downloaded {len(test_df)} test samples")
        
        return train_path, test_path
        
    except Exception as e:
        print(f"❌ Failed to download base dataset: {e}")
        return None, None

def merge_datasets(base_train_path: str, feedback_path: str) -> pd.DataFrame:
    """
    Merge base training data with feedback corrections.
    
    Args:
        base_train_path: Path to base training CSV
        feedback_path: Path to feedback CSV
    
    Returns:
        Merged DataFrame
    """
    # Load base training data
    base_df = pd.read_csv(base_train_path)
    print(f"Base training data: {len(base_df)} samples")
    
    # Load feedback
    if not os.path.exists(feedback_path):
        print(f"No feedback file found at {feedback_path}")
        return base_df
    
    feedback_df = pd.read_csv(feedback_path)
    print(f"Feedback data: {len(feedback_df)} samples")
    
    # Check minimum feedback threshold
    user_corrections = feedback_df[feedback_df['feedback_type'] == 'user_correction']
    if len(user_corrections) < MIN_FEEDBACK_SAMPLES:
        print(f"⚠ Warning: Only {len(user_corrections)} user corrections (minimum: {MIN_FEEDBACK_SAMPLES})")
        print("Not enough feedback for meaningful retraining. Keeping current model.")
        return None
    
    # Prepare feedback for merging
    # For user corrections: use correct_label
    # For low confidence: we don't have correct label, skip those for now
    user_corrections = feedback_df[feedback_df['feedback_type'] == 'user_correction'].copy()
    user_corrections['label'] = user_corrections['correct_label']
    
    # Select only needed columns
    feedback_processed = user_corrections[['text', 'label']].copy()
    
    print(f"User corrections to incorporate: {len(feedback_processed)}")
    
    # Combine datasets
    combined_df = pd.concat([base_df[['text', 'label']], feedback_processed], ignore_index=True)
    
    # Remove duplicates (keep last occurrence - newest labels)
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='last')
    
    print(f"Combined dataset: {len(combined_df)} samples (after deduplication)")
    
    return combined_df

def train_new_model(train_df: pd.DataFrame) -> Pipeline:
    """
    Train a new model on the merged dataset.
    
    Args:
        train_df: Training DataFrame with 'text' and 'label' columns
    
    Returns:
        Trained pipeline
    """
    print("\nTraining new model...")
    
    # Create pipeline (same architecture as baseline)
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True
        )),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ))
    ])
    
    X_train = train_df['text']
    y_train = train_df['label']
    
    model.fit(X_train, y_train)
    
    print("✓ Model training complete")
    
    return model

def main():
    """Main retraining workflow."""
    
    print("="*70)
    print("AUTOMATED MODEL RETRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Get credentials from environment
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USERNAME = os.getenv("HF_USERNAME")
    
    if not HF_TOKEN or not HF_USERNAME:
        print("❌ Error: HF_TOKEN and HF_USERNAME environment variables must be set")
        sys.exit(1)
    
    HF_MODEL_REPO = f"{HF_USERNAME}/sentiment-analysis-model"
    HF_DATASET_REPO = f"{HF_USERNAME}/sentiment-analysis-data"
    
    # Step 1: Download base dataset from HuggingFace
    print("\n[1/7] Downloading base dataset from HuggingFace...")
    train_path, test_path = download_base_dataset(HF_DATASET_REPO, HF_TOKEN)
    
    if not train_path or not test_path:
        print("❌ Failed to download base dataset. Exiting.")
        sys.exit(1)
    
    # Step 2: Download latest feedback
    print("\n[2/7] Downloading feedback from HuggingFace...")
    feedback_path = download_feedback(HF_DATASET_REPO, HF_TOKEN)
    
    if not feedback_path:
        print("❌ Failed to download feedback. Exiting.")
        sys.exit(1)
    
    # Step 3: Merge datasets
    print("\n[3/7] Merging feedback with base training data...")
    merged_df = merge_datasets(train_path, feedback_path)
    
    if merged_df is None:
        print("⚠ Insufficient feedback for retraining. Exiting.")
        sys.exit(0)
    
    # Step 4: Train new model
    print("\n[4/7] Training new model...")
    new_model = train_new_model(merged_df)
    
    # Step 5: Evaluate new model
    print("\n[5/7] Evaluating new model...")
    test_df = pd.read_csv(test_path)
    X_test = test_df['text']
    y_test = test_df['label']
    
    new_metrics = evaluate_model(new_model, X_test, y_test)
    print_evaluation_summary(new_metrics, "New Model")
    
    # Step 6: Compare with current model
    print("\n[6/7] Comparing with current model...")
    
    # Download current model metrics from HF Hub
    try:
        from huggingface_hub import hf_hub_download
        metrics_file = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="metrics_v1.joblib",
            token=HF_TOKEN,
            cache_dir=".hf_cache"
        )
        old_metrics = joblib.load(metrics_file)
    except Exception as e:
        print(f"Failed to load current metrics from HF Hub: {e}")
        old_metrics = {'accuracy': 0.818}  # Default to v1 accuracy
    
    should_deploy, reason = compare_models(new_metrics, old_metrics, IMPROVEMENT_THRESHOLD)
    
    print(f"\nDecision: {reason}")
    
    # Step 7: Deploy if improved
    if should_deploy:
        print("\n[7/7] Deploying new model to HuggingFace Hub...")
        
        # Determine next version
        next_version = get_next_version(HF_MODEL_REPO, HF_TOKEN)
        print(f"Next version: {next_version}")
        
        # Save model locally
        new_model_path = f"model/model_{next_version}.joblib"
        new_metrics_path = f"model/metrics_{next_version}.joblib"
        
        joblib.dump(new_model, new_model_path)
        joblib.dump({
            **new_metrics,
            'version': next_version,
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(merged_df),
            'improvement_over_previous': new_metrics['accuracy'] - old_metrics.get('accuracy', 0)
        }, new_metrics_path)
        
        print(f"✓ Saved locally: {new_model_path}")
        
        # Upload to HF Hub
        try:
            push_model_to_hub(
                model_path=new_model_path,
                repo_name=HF_MODEL_REPO,
                hf_token=HF_TOKEN,
                commit_message=f"Automated retraining: {next_version} (accuracy: {new_metrics['accuracy']:.4f})"
            )
            print(f"\n✅ SUCCESS: Model {next_version} deployed!")
            print(f"   Accuracy: {new_metrics['accuracy']:.4f}")
            print(f"   Improvement: +{(new_metrics['accuracy'] - old_metrics.get('accuracy', 0))*100:.2f}%")
        except Exception as e:
            print(f"❌ Failed to upload to HF Hub: {e}")
            sys.exit(1)
    else:
        print("\n[7/7] Skipping deployment (no improvement)")
        print("✓ Current model remains in production")
    
    print("\n" + "="*70)
    print(f"Retraining complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()
