# Legal Risk Detection in Contracts using BERT

## 1. Project Overview

This project is a simple proof-of-concept system to detect risky clauses in contracts. It uses a baseline text classifier and a fine-tuned BERT model to determine whether a clause is risky (`1`) or safe (`0`).

Why it matters:
- Contracts often contain terms that can cause legal exposure.
- Automated screening helps reviewers focus on important clauses faster.
- Using explainability makes decisions easier to understand.

## 2. Features

- Baseline model: TF-IDF + Logistic Regression
- BERT fine-tuning for better text understanding
- SHAP explainability for interpreting predictions
- Evaluation metrics: Precision, Recall, F1 (plus accuracy)

## 3. Dataset

A small synthetic dataset is included for demonstration:
- `data/contracts.csv` contains 100 rows
- 50 risky clauses (e.g., unlimited liability, no termination, vague terms)
- 50 non-risky clauses (e.g., clear responsibilities, fair termination, liability limits)

## 4. Project Structure

- `main.py` : End-to-end pipeline (load data, preprocess, train baseline, train BERT, explain)
- `src/preprocessing.py` : Data loading and text cleaning
- `src/baseline_model.py` : TF-IDF + Logistic Regression training and evaluation
- `src/bert_model.py` : BERT model training and evaluation
- `src/explainability.py` : SHAP explainability logic
- `generate_contract_dataset.py` : Script to generate synthetic dataset
- `data/contracts.csv` : Example data

## 5. How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline:
   ```bash
   python main.py
   ```

## 6. Sample Output

Example metrics from baseline run:
- Accuracy: 0.50
- Precision: 0.50
- Recall: 1.00
- F1-Score: 0.67

SHAP explanation example:
- clause: "all information must be kept confidential and not disclosed to third parties"
- model output: risky
- top word impact: `parties` decreases risk

## 7. Future Improvements

- Use larger and real contract datasets
- Better BERT fine-tuning (more epochs and data)
- Add cross-validation and model selection
- Enhance explainability with full word-level summaries

