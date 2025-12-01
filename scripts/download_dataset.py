"""
Script to download and prepare the SST-2 sentiment dataset.
"""
import pandas as pd
from datasets import load_dataset
import os

def download_sst2():
    """Download SST-2 dataset from HuggingFace."""
    print("Downloading SST-2 dataset from HuggingFace...")
    dataset = load_dataset("glue", "sst2")
    
    # Convert to pandas DataFrames
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['validation'])  # SST-2 calls it 'validation'
    
    # Rename columns for consistency
    train_df = train_df.rename(columns={'sentence': 'text', 'label': 'label'})
    test_df = test_df.rename(columns={'sentence': 'text', 'label': 'label'})
    
    # Map labels: 0 = negative, 1 = positive
    label_map = {0: 'negative', 1: 'positive'}
    train_df['label'] = train_df['label'].map(label_map)
    test_df['label'] = test_df['label'].map(label_map)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    train_df.to_csv('data/base_train.csv', index=False)
    test_df.to_csv('data/base_test.csv', index=False)
    
    print(f"Training set: {len(train_df)} samples saved to data/base_train.csv")
    print(f"Test set: {len(test_df)} samples saved to data/base_test.csv")
    print(f"\nLabel distribution (training):")
    print(train_df['label'].value_counts())
    
    return train_df, test_df

if __name__ == "__main__":
    download_sst2()
