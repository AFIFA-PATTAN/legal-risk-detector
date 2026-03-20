"""
Explainability Module
This module provides model interpretability using SHAP (SHapley Additive exPlanations).
Helps understand which words/features contribute most to risk predictions.
"""

import shap
import numpy as np


def explain_prediction(model, vectorizer, sample_text, background_texts=None, top_n=10):
    """
    Explain a single sample prediction using SHAP values.

    Args:
        model: Trained baseline model (LogisticRegression)
        vectorizer: Trained TF-IDF vectorizer
        sample_text (str): Text sample to explain
        background_texts (list of str): Optional texts to build background (same format as sample_text)
        top_n (int): Number of top words to show

    Returns:
        dict: Contains prediction, probability, and top contributing words
    """
    # Vectorize the sample text
    sample_vec = vectorizer.transform([sample_text])

    # Prepare background data for SHAP
    if background_texts is None:
        background_vec = sample_vec
    else:
        background_vec = vectorizer.transform(background_texts)

    # Convert to dense arrays for SHAP (KernelExplainer accepts dense data)
    background_arr = background_vec.toarray()
    sample_arr = sample_vec.toarray()

    # Create SHAP explainer for class 1 probability
    def model_proba(x):
        # x is dense matrix for SHAP, return probability of risky class
        return model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(model_proba, background_arr)

    shap_values = explainer.shap_values(sample_arr)

    # shap_values can be a list (for each class) or array
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)[1]

    shap_values = np.array(shap_values).reshape(-1)

    feature_names = vectorizer.get_feature_names_out()

    # Get non-zero features in sample
    sample_nonzero = sample_arr[0] != 0

    contributions = []
    for idx in np.where(sample_nonzero)[0]:
        contributions.append((feature_names[idx], float(shap_values[idx])))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    top_contributions = contributions[:top_n]

    prob = float(model.predict_proba(sample_vec)[0, 1])
    pred = int(model.predict(sample_vec)[0])

    print("\n[EXPLAINABILITY] SHAP explanation for sample")
    print(f"Text: {sample_text[:120]}...")
    print(f"Prediction: {'RISKY' if pred == 1 else 'NON-RISKY'} (probability={prob:.3f})")
    print("Top important words:")
    for word, value in top_contributions:
        direction = 'increases' if value > 0 else 'decreases'
        print(f"  - {word}: {value:.4f} ({direction} risk)")

    return {
        'text': sample_text,
        'prediction': pred,
        'probability': prob,
        'top_words': top_contributions
    }


def explain_with_shap(model, vectorizer, texts, background_texts=None, top_n=10, max_samples=3):
    """
    Explain a list of samples using SHAP.

    Args:
        model: Trained baseline model
        vectorizer: Trained TF-IDF vectorizer
        texts (list of str): Text samples to explain
        background_texts (list of str): Optional background examples
        top_n (int): Number of top words
        max_samples (int): Maximum number of samples to explain

    Returns:
        list: Explanation dicts for each sample
    """
    reports = []
    for i, text in enumerate(texts[:max_samples]):
        print(f"\n[EXPLAINABILITY] Explaining sample {i+1}/{min(max_samples, len(texts))}")
        report = explain_prediction(model, vectorizer, text, background_texts, top_n)
        reports.append(report)
    return reports


def explain_with_lime(*args, **kwargs):
    """
    Legacy wrapper for backward compatibility; now uses SHAP.
    """
    print("[EXPLAINABILITY] LIME usage removed; using SHAP instead.")
    if len(args) >= 2:
        model, vectorizer = args[0], args[1]
        texts = args[2] if len(args) > 2 else []
        return explain_with_shap(model, vectorizer, texts, **kwargs)
    raise ValueError("explain_with_lime expects (model, vectorizer, texts, ...) now")


if __name__ == "__main__":
    print("Explainability module ready to use")
