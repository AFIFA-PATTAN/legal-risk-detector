# Legal Risk Detection in Contracts - Usage Guide

## Project Overview

This project implements a machine learning pipeline to detect legal risks in contracts using:
- **Baseline Model**: TF-IDF + Logistic Regression (fast, interpretable)
- **BERT Model**: Fine-tuned transformer for advanced NLP (more accurate)
- **SHAP/LIME**: Explainability to understand model predictions

## Project Structure

```
legal-risk-detector/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── preprocessing.py            # Data loading and cleaning
│   ├── baseline_model.py          # TF-IDF + Logistic Regression
│   ├── bert_model.py              # BERT fine-tuning
│   └── explainability.py          # SHAP/LIME explanations
├── data/
│   └── contracts.csv              # Your dataset (create automatically)
├── notebooks/                     # Jupyter notebooks for experiments
├── results/                       # Output models and reports (created after running)
├── main.py                        # Main orchestration script
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Installation & Setup

### 1. Virtual Environment Setup

The project includes a virtual environment (.venv). To activate it:

**On PowerShell (Windows):**
```powershell
.venv\Scripts\Activate.ps1
```

**If you get execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**On macOS/Linux (Bash):**
```bash
source .venv/bin/activate
```

### 2. Install Dependencies

With the virtual environment activated:
```bash
pip install -r requirements.txt
```

This installs:
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning models
- `torch`: Deep learning framework
- `transformers`: Hugging Face BERT models
- `lime`: Model-agnostic explanations
- `shap`: SHAP values for interpretability
- `jupyter`: Interactive notebooks

## Quick Start

### Run the Complete Pipeline

```bash
python main.py
```

This will:
1. Create sample data (if it doesn't exist)
2. Load and preprocess contracts
3. Train baseline model (TF-IDF + Logistic Regression)
4. Train BERT model (optional, requires GPU for speed)
5. Generate explanations using LIME
6. Save results to `results/` folder

### Expected Output

```
================================================================================
LEGAL RISK DETECTION IN CONTRACTS - COMPLETE PIPELINE
================================================================================

[MAIN] >>> STEP 1: Loading Data
[PREPROCESSING] Loading data from data/contracts.csv...
[PREPROCESSING] Loaded 20 samples

[MAIN] >>> STEP 2: Preprocessing Data
[PREPROCESSING] Cleaning text...
[PREPROCESSING] After cleaning: 20 samples
[PREPROCESSING] Train set: 16 samples
[PREPROCESSING] Test set: 4 samples

[MAIN] >>> STEP 3: Training Baseline Model (TF-IDF + Logistic Regression)
[BASELINE MODEL] Training baseline model...
[BASELINE MODEL] Vectorizing text with TF-IDF...
[BASELINE MODEL] Created 1234 features
[BASELINE MODEL] Training Logistic Regression classifier...

============================================================
BASELINE MODEL - PERFORMANCE METRICS
============================================================
Accuracy:  0.7500
Precision: 0.6667
Recall:    1.0000
F1-Score:  0.8000
```

## Using Your Own Data

### Data Format

Create a CSV file at `data/contracts.csv` with two columns:

```csv
text,label
"This is a contract text...",0
"Another contract with risky clauses...",1
```

**Column Requirements:**
- `text`: Contract text (required)
- `label`: 0 (non-risky) or 1 (risky)

### Example CSV

```csv
text,label
This agreement outlines normal terms and conditions,0
This contract has unlimited liability and no warranties,1
Standard payment terms with 30 days notice for termination,0
Unlimited company rights with no liability caps,1
```

## Module Guide

### preprocessing.py

Handles data loading and cleaning:

```python
from preprocessing import load_data, preprocess_data, split_data

# Load data
df = load_data('data/contracts.csv')

# Clean and preprocess
df_clean = preprocess_data(df)

# Split into train/test
X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2)
```

**What it does:**
- Converts text to lowercase
- Removes URLs and emails
- Keeps only alphanumeric characters
- Removes extra whitespace
- Removes missing values

### baseline_model.py

Train and evaluate TF-IDF + Logistic Regression:

```python
import baseline_model

# Train
vectorizer, classifier = baseline_model.train_baseline_model(X_train, y_train)

# Predict
predictions = baseline_model.predict_baseline(X_test)

# Evaluate
metrics = baseline_model.evaluate_baseline_model(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Save/Load
baseline_model.save_baseline_model('model.pkl')
baseline_model.load_baseline_model('model.pkl')
```

### bert_model.py

Fine-tune BERT for contract classification:

```python
import bert_model

# Train (uses GPU if available)
model, tokenizer = bert_model.train_bert_model(
    X_train, y_train, X_test, y_test,
    epochs=3,
    batch_size=16
)

# Predict
predictions = bert_model.predict_bert(model, tokenizer, X_test)

# Evaluate
metrics = bert_model.evaluate_bert_model(model, tokenizer, X_test, y_test)

# Save
bert_model.save_bert_model(model, tokenizer, 'bert_model_folder')
```

### explainability.py

Understand model predictions using LIME:

```python
from explainability import explain_with_lime, print_explanations

# Define prediction function
def predict_func(texts):
    return baseline_model.predict_baseline(texts)

# Explain predictions
explanations = explain_with_lime(predict_func, X_test[:5], num_features=10)

# Display explanations
print_explanations(explanations)
```

**Output Example:**
```
--- Sample 1 ---
Text: This agreement has unlimited liability...
Prediction: RISKY

Top Contributing Words:
  • 'unlimited': 0.3245 (Increases risk)
  • 'liability': 0.2891 (Increases risk)
  • 'no': 0.1567 (Increases risk)
```

## Output Files

After running `main.py`, check the `results/` folder:

- **baseline_model.pkl**: Saved baseline model for future predictions
- **bert_model/**: Saved BERT model directory
- **explainability_report.txt**: Detailed explanations of predictions
- **summary.txt**: Performance metrics summary

## Common Issues & Solutions

### Issue: BERT training is too slow

**Solution:**
- Reduce epochs: Change `epochs=3` to `epochs=1` in `main.py`
- Use smaller batch size: Change `batch_size=16` to `batch_size=8`
- Reduce training samples: Use a subset of your data

### Issue: Out of memory (GPU or CPU)

**Solution:**
- Reduce batch size (smaller numbers = less memory)
- Skip BERT training (baseline model is sufficient)
- Use a smaller pre-trained model: `distilbert-base-uncased`

### Issue: CSV file not found

**Solution:**
- Run `python main.py` to automatically create sample data, OR
- Manually create `data/contracts.csv` with your data

### Issue: Python/pip commands not found

**Solution:**
- Make sure virtual environment is activated: `.venv\Scripts\Activate.ps1` (Windows)
- Check if Python is in PATH: `python --version`

## Next Steps

1. **Replace sample data**: Add your own contracts to `data/contracts.csv`
2. **Experiment**: Use Jupyter notebooks in `notebooks/` folder
3. **Fine-tune**: Adjust model parameters in `main.py`
4. **Deploy**: Save your trained models and use them for predictions

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Transformers Library](https://huggingface.co/transformers/)
- [LIME Paper](https://arxiv.org/abs/1602.04938)
- [SHAP Documentation](https://shap.readthedocs.io/)

## License

This project is licensed under the MIT License - see LICENSE file for details.
