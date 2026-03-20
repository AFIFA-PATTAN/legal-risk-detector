"""
BERT Model Module
This module implements a BERT-based model for text classification
in legal risk detection.
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np


class ContractDataset(Dataset):
    """
    Custom Dataset class for contract texts and labels.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of contract texts
            labels (list): List of labels (0 or 1)
            tokenizer: BERT tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }


def train_bert_model(X_train, y_train, X_test, y_test, epochs=3, batch_size=16):
    """
    Train a BERT model for binary classification.
    
    Args:
        X_train (list): Training texts
        y_train (list): Training labels
        X_test (list): Test texts
        y_test (list): Test labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        model, tokenizer: Trained model and tokenizer
    """
    print("\n[BERT MODEL] Setting up BERT model...")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[BERT MODEL] Using device: {device}")
    
    # Load tokenizer and model
    model_name = "bert-base-uncased"
    print(f"[BERT MODEL] Loading {model_name}...")
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Create datasets
    print("[BERT MODEL] Creating datasets...")
    train_dataset = ContractDataset(X_train, y_train, tokenizer)
    test_dataset = ContractDataset(X_test, y_test, tokenizer)
    
    # Define training arguments (compatible with older/newer transformers versions)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        # remove evaluation_strategy and save_strategy for max compatibility
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train
    print(f"\n[BERT MODEL] Training for {epochs} epochs...")
    try:
        trainer.train()
    except Exception as e:
        print(f"[BERT MODEL] WARNING: Training failed: {e}")
        print("[BERT MODEL] Skipping BERT model training and returning None.")
        return None, None

    print("[BERT MODEL] Training completed!")
    return model, tokenizer


def evaluate_bert_model(model, tokenizer, X_test, y_test):
    """
    Evaluate BERT model performance.
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        X_test (list): Test texts
        y_test (list): Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    if model is None or tokenizer is None:
        print("[BERT MODEL] No trained model/tokenizer available for evaluation.")
        return None

    print("\n[BERT MODEL] Evaluating model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for text in X_test:
            # Encode
            encoding = tokenizer(
                str(text),
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Predict
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    print("\n" + "="*60)
    print("BERT MODEL - PERFORMANCE METRICS")
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


def predict_bert(model, tokenizer, X_test):
    """
    Make predictions using trained BERT model.
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        X_test (list): Test texts
        
    Returns:
        array: Predicted labels
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for text in X_test:
            encoding = tokenizer(
                str(text),
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            predictions.append(pred)
    
    return np.array(predictions)


def save_bert_model(model, tokenizer, filepath):
    """
    Save trained BERT model and tokenizer.
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        filepath (str): Path to save model
    """
    model.save_pretrained(filepath)
    tokenizer.save_pretrained(filepath)
    print(f"[BERT MODEL] Model saved to {filepath}")


if __name__ == "__main__":
    print("BERT model module ready to use")
