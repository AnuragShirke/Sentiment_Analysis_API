"""
Initial model training script.
Trains TF-IDF + Logistic Regression on SST-2 dataset.
"""
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

def train_model():
    """Train the baseline sentiment analysis model."""
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv('data/base_train.csv')
    test_df = pd.read_csv('data/base_test.csv')
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create TF-IDF + LogisticRegression pipeline
    print("\nTraining TF-IDF + Logistic Regression model...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # unigrams and bigrams
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
    
    # Fit model
    X_train = train_df['text']
    y_train = train_df['label']
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    X_test = test_df['text']
    y_test = test_df['label']
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"{'='*50}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    os.makedirs('model', exist_ok=True)
    model_path = 'model/model_v1.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save metrics
    metrics = {
        'version': 'v1',
        'accuracy': accuracy,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }
    
    metrics_path = 'model/metrics_v1.joblib'
    joblib.dump(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")
    
    return model, accuracy

if __name__ == "__main__":
    train_model()
