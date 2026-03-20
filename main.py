"""
Legal Risk Detection in Contracts - Main Script
This script orchestrates the entire pipeline for detecting legal risks in contracts.

Usage:
    python main.py

Expected data format:
    CSV file at data/contracts.csv with columns: 'text' and 'label'
    - text: Contract text
    - label: 0 (non-risky) or 1 (risky)
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import modules
from preprocessing import load_data, preprocess_data, split_data
import baseline_model
import bert_model
from explainability import explain_prediction


def create_sample_data():
    """
    Create sample data for demonstration.
    This creates: data/contracts.csv
    """
    import pandas as pd
    
    print("\n[MAIN] Creating sample data for demonstration...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Sample legal contract texts with risk labels
    # 0 = non-risky, 1 = risky
    samples = {
        'text': [
            'This agreement outlines the terms and conditions between the two parties.',
            'All services provided are without any liability or warranty.',
            'The company shall indemnify and hold harmless the client from all damages.',
            'Payment shall be made monthly within thirty days of invoice.',
            'Either party may terminate this agreement with thirty days written notice.',
            'All information must be kept confidential and not disclosed to third parties.',
            'The client assumes all risk and liability for any use of the services.',
            'No warranties are provided express or implied including fitness for purpose.',
            'The company is not responsible for any indirect damages or lost profits.',
            'Limitations of liability shall not exceed the total amount paid by client.',
            'All disputes shall be resolved by binding arbitration.',
            'Force majeure events exclude liability from performance obligations.',
            'This contract grants unlimited rights to modify terms at any time.',
            'The company reserves the right to terminate without cause or notice.',
            'All intellectual property and data collected belongs to the company.',
            'Client waives all rights to pursue claims or legal action.',
            'Standard terms and conditions for routine service provision.',
            'Professional services delivered according to industry standards.',
            'Mutual obligations clearly defined with reasonable dispute resolution.',
            'Transparent pricing with clear cancellation and refund policies.',
        ],
        'label': [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    }
    
    df = pd.DataFrame(samples)
    df.to_csv('data/contracts.csv', index=False)
    print(f"[MAIN] Created data/contracts.csv with {len(df)} samples")
    
    return df


def main():
    """
    Main pipeline for legal risk detection.
    """
    print("\n" + "="*80)
    print("LEGAL RISK DETECTION IN CONTRACTS - COMPLETE PIPELINE")
    print("="*80)
    
    # ========== STEP 1: Load Data ==========
    print("\n[MAIN] >>> STEP 1: Loading Data")
    
    data_path = 'data/contracts.csv'
    
    # Create sample data if it doesn't exist
    if not os.path.exists(data_path):
        create_sample_data()
    
    # Load data
    df = load_data(data_path)
    
    # ========== STEP 2: Preprocess Data ==========
    print("\n[MAIN] >>> STEP 2: Preprocessing Data")
    
    df_clean = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2)
    
    # ========== STEP 3: Train Baseline Model ==========
    print("\n[MAIN] >>> STEP 3: Training Baseline Model (TF-IDF + Logistic Regression)")
    
    vectorizer, classifier = baseline_model.train_baseline_model(X_train, y_train)
    baseline_metrics = baseline_model.evaluate_baseline_model(X_test, y_test)
    
    # ========== STEP 4: Train BERT Model ==========
    print("\n[MAIN] >>> STEP 4: Training BERT Model")
    print("[MAIN] Note: BERT model training is simplified and uses 1 epoch for demo purposes")
    
    bert_trained = False
    try:
        bert_model_obj, bert_tokenizer = bert_model.train_bert_model(
            X_train, y_train, X_test, y_test,
            epochs=1,
            batch_size=8
        )

        bert_metrics = bert_model.evaluate_bert_model(
            bert_model_obj, bert_tokenizer, X_test, y_test
        )

        bert_trained = True
    except Exception as e:
        print(f"\n[MAIN] WARNING: BERT training failed: {e}")
        print("[MAIN] Continuing without BERT model due to failure or environment constraints.")
        bert_trained = False
    
    # ========== STEP 5: Model Explainability (SHAP) ==========
    print("\n[MAIN] >>> STEP 5: Explainability Analysis Using SHAP")
    
    # Use one test sample for SHAP explanation
    sample_idx = 0
    sample_text = X_test[sample_idx]
    print(f"[MAIN] Explaining test sample index {sample_idx}: {sample_text[:150]}...")

    explanation = explain_prediction(
        classifier,
        vectorizer,
        sample_text,
        background_texts=X_train[:min(50, len(X_train))],
        top_n=10,
    )

    print("\n[MAIN] SHAP explanation complete")
    print(f"[MAIN] Predicted label: {explanation['prediction']}, probability: {explanation['probability']:.4f}")

    # ========== STEP 6: Save Results ==========
    print("\n[MAIN] >>> STEP 6: Saving Results")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save models
    baseline_model.save_baseline_model('results/baseline_model.pkl')
    
    if bert_trained:
        bert_model.save_bert_model(bert_model_obj, bert_tokenizer, 'results/bert_model')
    
    # Save explainability report
    report_path = 'results/explainability_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('SHAP Explainability for one test example\n')
        f.write(f"Text: {explanation['text']}\n")
        f.write(f"Prediction: {explanation['prediction']}\n")
        f.write(f"Probability: {explanation['probability']:.4f}\n")
        f.write('Top words (word, shap importance):\n')
        for word, val in explanation['top_words']:
            f.write(f"  - {word}: {val:.4f}\n")

    print(f"[MAIN] Explainability report saved to {report_path}")

    # ========== STEP 7: Summary Report ==========
    print("\n[MAIN] >>> STEP 7: Summary Report")
    
    summary = f"""
{'='*80}
LEGAL RISK DETECTION - RESULTS SUMMARY
{'='*80}

DATASET INFORMATION:
  - Total samples: {len(df)}
  - Training samples: {len(X_train)}
  - Test samples: {len(X_test)}

BASELINE MODEL PERFORMANCE (TF-IDF + Logistic Regression):
  - Accuracy:  {baseline_metrics['accuracy']:.4f}
  - Precision: {baseline_metrics['precision']:.4f}
  - Recall:    {baseline_metrics['recall']:.4f}
  - F1-Score:  {baseline_metrics['f1']:.4f}
"""
    
    if bert_trained:
        summary += f"""
BERT MODEL PERFORMANCE:
  - Accuracy:  {bert_metrics['accuracy']:.4f}
  - Precision: {bert_metrics['precision']:.4f}
  - Recall:    {bert_metrics['recall']:.4f}
  - F1-Score:  {bert_metrics['f1']:.4f}
"""
    
    summary += f"""
OUTPUT FILES:
  - Baseline model: results/baseline_model.pkl
  - Explainability report: results/explainability_report.txt
"""
    
    if bert_trained:
        summary += "  - BERT model: results/bert_model/\n"
    
    summary += f"{'='*80}\n"
    
    print(summary)
    
    # Save summary to file
    with open('results/summary.txt', 'w') as f:
        f.write(summary)
    
    print("\n[MAIN] ✓ Pipeline completed successfully!")
    print("[MAIN] Check the 'results' folder for output files.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[MAIN] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
