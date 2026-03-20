"""
Data Preprocessing Module
This module handles loading and cleaning legal contract data.
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load raw contract data from a CSV file.
    Expected columns: 'text' and 'label'
    
    Args:
        filepath (str): Path to the CSV file containing contracts
        
    Returns:
        pd.DataFrame: DataFrame with contract data
    """
    print(f"\n[PREPROCESSING] Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[PREPROCESSING] Loaded {len(df)} samples")
    return df


def clean_text(text):
    """
    Clean and preprocess text data:
    - Convert to lowercase
    - Remove URLs
    - Remove special characters and punctuation
    - Remove extra whitespace
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def preprocess_data(df):
    """
    Preprocess the entire dataset.
    
    Args:
        df (pd.DataFrame): Raw data with 'text' and 'label' columns
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    print("\n[PREPROCESSING] Cleaning text...")
    
    # Clone dataframe to avoid modifying original
    df_clean = df.copy()
    
    # Clean text column
    df_clean['text'] = df_clean['text'].apply(clean_text)
    
    # Remove rows with missing values
    df_clean = df_clean.dropna(subset=['text', 'label'])
    
    # Remove empty texts
    df_clean = df_clean[df_clean['text'].str.len() > 0]
    
    print(f"[PREPROCESSING] After cleaning: {len(df_clean)} samples")
    return df_clean


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Args:
        df (pd.DataFrame): Preprocessed data
        test_size (float): Proportion of data for testing (default 0.2)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df['text'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n[PREPROCESSING] Train set: {len(X_train)} samples")
    print(f"[PREPROCESSING] Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Preprocessing module ready to use")
