# Speech-classification-with-Pca-Smote-and-Expectation-Maximization-
Speech Recognition 


# PCA + SMOTE + EM (GMM) Membership Features — Reproducible Example

This repository contains a compact, reproducible Python example that implements the pipeline described in the paper:
**PCA → SMOTE (or fallback) → Expectation Maximization (GaussianMixture)** to obtain soft cluster membership features which are appended to PCA features for classification of (synthetic) pathological speech-like signals.

This is a toy / demonstration implementation intended to show how to:
- generate a synthetic imbalanced dataset of speech-like signals,
- extract simple time/frequency features,
- run PCA,
- oversample minority classes with SMOTE (optional; falls back to a random oversampler),
- train a Gaussian Mixture Model (EM) to obtain membership probabilities,
- append membership probabilities as new features and train classifiers (SVM / GaussianNB),
- evaluate baseline vs augmented pipelines.

---

## Repo structure

