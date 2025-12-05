"""
Model evaluation and comparison utilities.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import Dict, Tuple
import joblib

def evaluate_model(model, X_test, y_test) -> Dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained sklearn pipeline
        X_test: Test features (texts)
        y_test: Test labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class metrics
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'n_samples': len(y_test)
    }

def compare_models(new_metrics: Dict, old_metrics: Dict, threshold: float = 0.001) -> Tuple[bool, str]:
    """
    Compare two models and decide if new model should be promoted.
    
    Args:
        new_metrics: Metrics from newly trained model
        old_metrics: Metrics from current production model
        threshold: Minimum improvement required (default 0.1%)
    
    Returns:
        (should_promote, reason) tuple
    """
    new_acc = new_metrics['accuracy']
    old_acc = old_metrics['accuracy']
    
    improvement = new_acc - old_acc
    improvement_pct = improvement * 100
    
    if improvement > threshold:
        reason = f"New model is better: {new_acc:.4f} vs {old_acc:.4f} (+{improvement_pct:.2f}%)"
        return True, reason
    elif improvement < -threshold:
        reason = f"New model is worse: {new_acc:.4f} vs {old_acc:.4f} ({improvement_pct:.2f}%)"
        return False, reason
    else:
        reason = f"No significant change: {new_acc:.4f} vs {old_acc:.4f} ({improvement_pct:.2f}%, threshold: {threshold*100:.2f}%)"
        return False, reason

def should_promote(new_accuracy: float, old_accuracy: float, threshold: float = 0.001) -> bool:
    """
    Simplified promotion check.
    
    Args:
        new_accuracy: New model's accuracy
        old_accuracy: Current model's accuracy
        threshold: Minimum improvement (default 0.1%)
    
    Returns:
        True if new model should be promoted
    """
    return (new_accuracy - old_accuracy) > threshold

def print_evaluation_summary(metrics: Dict, model_name: str = "Model"):
    """
    Print a formatted evaluation summary.
    
    Args:
        metrics: Evaluation metrics dictionary
        model_name: Name of the model being evaluated
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Samples:   {metrics['n_samples']}")
    print(f"{'='*60}\n")

def load_metrics(metrics_path: str) -> Dict:
    """
    Load saved metrics from disk.
    
    Args:
        metrics_path: Path to metrics .joblib file
    
    Returns:
        Metrics dictionary
    """
    try:
        return joblib.load(metrics_path)
    except Exception as e:
        print(f"Failed to load metrics from {metrics_path}: {e}")
        return {'accuracy': 0.0, 'version': 'unknown'}

if __name__ == "__main__":
    # Example usage
    print("Model evaluation utilities loaded successfully")
