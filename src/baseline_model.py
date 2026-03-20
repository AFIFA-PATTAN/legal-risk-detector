"""
Baseline Model Module
This module implements a baseline model using TF-IDF + Logistic Regression
for legal risk detection in contracts.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import os


# Global variables to store the trained model and vectorizer
vectorizer = None
classifier = None


def train_baseline_model(X_train, y_train, max_features=5000):
    """
    Train a baseline model using TF-IDF vectorizer and Logistic Regression.
    
    Args:
        X_train (array): Training texts
        y_train (array): Training labels (0 or 1)
        max_features (int): Maximum number of features for TF-IDF
        
    Returns:
        tuple: (vectorizer, classifier)
    """
    global vectorizer, classifier
    
    print("\n[BASELINE MODEL] Training baseline model...")
    print("[BASELINE MODEL] Vectorizing text with TF-IDF...")
    
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    
    print(f"[BASELINE MODEL] Created {X_train_vec.shape[1]} features")
    
    # Train logistic regression classifier
    print("[BASELINE MODEL] Training Logistic Regression classifier...")
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train_vec, y_train)
    
    print("[BASELINE MODEL] Training completed!")
    
    return vectorizer, classifier


def predict_baseline(X_test):
    """
    Make predictions on test data.
    
    Args:
        X_test (array): Test texts
        
    Returns:
        array: Predicted labels (0 or 1)
    """
    if vectorizer is None or classifier is None:
        print("[BASELINE MODEL] ERROR: Model not trained yet!")
        return None
    
    X_test_vec = vectorizer.transform(X_test)
    predictions = classifier.predict(X_test_vec)
    return predictions


def get_prediction_probabilities(X_test):
    """
    Get probability scores for predictions.
    
    Args:
        X_test (array): Test texts
        
    Returns:
        array: Probability scores for class 1
    """
    if vectorizer is None or classifier is None:
        print("[BASELINE MODEL] ERROR: Model not trained yet!")
        return None
    
    X_test_vec = vectorizer.transform(X_test)
    probabilities = classifier.predict_proba(X_test_vec)
    return probabilities


def evaluate_baseline_model(X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Args:
        X_test (array): Test texts
        y_test (array): True labels
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    predictions = predict_baseline(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    print("\n" + "="*60)
    print("BASELINE MODEL - PERFORMANCE METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60)
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Non-Risky', 'Risky']))
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


def save_baseline_model(filepath):
    """
    Save the trained model to disk.
    
    Args:
        filepath (str): Path to save the model
    """
    if vectorizer is None or classifier is None:
        print("[BASELINE MODEL] ERROR: Model not trained yet!")
        return
    
    model_data = {
        'vectorizer': vectorizer,
        'classifier': classifier
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"[BASELINE MODEL] Model saved to {filepath}")


def load_baseline_model(filepath):
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
    """
    global vectorizer, classifier
    
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    vectorizer = model_data['vectorizer']
    classifier = model_data['classifier']
    
    print(f"[BASELINE MODEL] Model loaded from {filepath}")


if __name__ == "__main__":
    print("Baseline model module ready to use")
