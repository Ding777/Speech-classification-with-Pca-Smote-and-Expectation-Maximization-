
---

# pca_smote_em_pipeline.py (full file)
Copy this file into the repo as `pca_smote_em_pipeline.py` (same code as you provided, with the `prepare_dataset()` function included):

```python
"""
pca_smote_em_pipeline.py

Complete reproducible example:
  PCA -> SMOTE (or fallback) -> EM (GMM) membership features -> classification

Requirements:
  numpy, pandas, scikit-learn
  (optional) imbalanced-learn for SMOTE

Usage:
  python pca_smote_em_pipeline.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

RND = 42
np.random.seed(RND)


# -------------------------
# Dataset preparation
# -------------------------
def make_signal(freq, length=1024, sr=8000, noise_level=0.2, phase=0.0):
    """
    Generate a simple synthetic "speech-like" signal:
      sinusoid + Gaussian noise + slow amplitude modulation
    """
    t = np.arange(length) / sr
    sig = np.sin(2 * np.pi * freq * t + phase)
    sig += noise_level * np.random.randn(length)
    sig *= (1.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    return sig


def extract_simple_features(sig, sr=8000, n_bins=16):
    """
    Extract a small set of features from a single signal:
      - time: mean, std, rough zero-crossing rate
      - spectral: log-energy in geometric frequency bands
    Returns: 1D numpy array of features (3 + n_bins)
    """
    mean = sig.mean()
    std = sig.std()
    zcr = np.mean(np.abs(np.diff(np.sign(sig))))
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
    bands = np.geomspace(50, sr / 2, n_bins + 1)
    band_energies = []
    for i in range(n_bins):
        mask = (freqs >= bands[i]) & (freqs < bands[i + 1])
        band_energies.append(np.log1p(fft[mask].sum()) if mask.any() else 0.0)
    feat = np.array([mean, std, zcr] + band_energies)
    return feat


def prepare_dataset(counts=(210, 70, 20), length=2048, sr=8000, n_bins=16, seed=RND):
    """
    Prepare an imbalanced synthetic dataset.
    Args:
      counts: tuple/list of ints, number of samples per class (class labels = 0..len(counts)-1)
      length: signal length in samples
      sr: sample rate for signal generation
      n_bins: number of spectral bands for features
      seed: RNG seed
    Returns:
      X: (N, F) numpy array of features
      y: (N,) numpy array of integer labels
      df: pandas DataFrame with features + 'label' column
    """
    np.random.seed(seed)
    features = []
    labels = []
    for cls, cnt in enumerate(counts):
        for i in range(cnt):
            if cls == 0:
                f = np.random.uniform(80, 250)
                noise = np.random.uniform(0.08, 0.2)
            elif cls == 1:
                f = np.random.uniform(60, 200)
                noise = np.random.uniform(0.12, 0.28)
            else:
                f = np.random.uniform(40, 180)
                noise = np.random.uniform(0.18, 0.4)
            sig = make_signal(f, length=length, sr=sr, noise_level=noise, phase=np.random.rand() * 2 * np.pi)
            feat = extract_simple_features(sig, sr=sr, n_bins=n_bins)
            features.append(feat)
            labels.append(cls)
    X = np.vstack(features)
    y = np.array(labels)
    colnames = ["mean", "std", "zcr"] + [f"band_{i}" for i in range(X.shape[1] - 3)]
    df = pd.DataFrame(X, columns=colnames)
    df["label"] = y
    return X, y, df


# -------------------------
# Helper oversampler fallback
# -------------------------
def simple_random_oversample(X, y, seed=RND):
    """
    Random oversampling fallback: replicate samples from minority classes to match the majority class size.
    Returns oversampled X and y (shuffled).
    """
    np.random.seed(seed)
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_new = []
    y_new = []
    for cls in unique:
        Xc = X[y == cls]
        nc = len(Xc)
        repeats = max_count // nc
        rem = max_count % nc
        parts = [Xc] * repeats
        if rem:
            parts.append(Xc[np.random.choice(nc, rem, replace=True)])
        X_rep = np.vstack(parts)
        X_new.append(X_rep)
        y_new.append(np.full(len(X_rep), cls))
    Xo = np.vstack(X_new)
    yo = np.concatenate(y_new)
    perm = np.random.permutation(len(yo))
    return Xo[perm], yo[perm]


# -------------------------
# Pipeline functions
# -------------------------
def eval_model(clf, Xs, ys):
    ypred = clf.predict(Xs)
    return {
        "accuracy": accuracy_score(ys, ypred),
        "macro_f1": f1_score(ys, ypred, average="macro"),
        "macro_recall": recall_score(ys, ypred, average="macro"),
    }


def run_pipeline(X, y, test_size=0.3, random_state=RND, pca_n_components=10, use_smote=True):
    """
    Run the baseline and paper-style pipeline on provided features X and labels y.
    Returns a dict with baseline and augmented results and demo DataFrame.
    """
    # train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # standard scaling
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # baseline classifiers trained on imbalanced data
    clf_svm = SVC(kernel="rbf", probability=True, random_state=random_state)
    clf_gnb = GaussianNB()
    clf_svm.fit(X_train_s, y_train)
    clf_gnb.fit(X_train_s, y_train)
    baseline_svm = eval_model(clf_svm, X_test_s, y_test)
    baseline_gnb = eval_model(clf_gnb, X_test_s, y_test)

    # PCA
    pca = PCA(n_components=min(pca_n_components, X_train.shape[1]), random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    # Oversample (SMOTE if available, otherwise simple_random_oversample)
    used_smote = False
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=random_state)
            X_res, y_res = sm.fit_resample(X_train_pca, y_train)
            used_smote = True
        except Exception:
            X_res, y_res = simple_random_oversample(X_train_pca, y_train, seed=random_state)
            used_smote = False
    else:
        X_res, y_res = simple_random_oversample(X_train_pca, y_train, seed=random_state)

    # EM clustering (GaussianMixture) to get soft membership probabilities
    n_components_gmm = len(np.unique(y_res))
    gmm = GaussianMixture(n_components=n_components_gmm, covariance_type="full", random_state=random_state, max_iter=300)
    gmm.fit(X_res)
    members_train = gmm.predict_proba(X_res)
    members_test = gmm.predict_proba(X_test_pca)

    # create augmented feature sets (PCA features + membership degrees)
    X_res_aug = np.hstack([X_res, members_train])
    X_test_aug = np.hstack([X_test_pca, members_test])

    # scale augmented features
    scaler_aug = StandardScaler().fit(X_res_aug)
    X_res_aug_s = scaler_aug.transform(X_res_aug)
    X_test_aug_s = scaler_aug.transform(X_test_aug)

    # train classifiers on augmented features
    clf_svm_aug = SVC(kernel="rbf", probability=True, random_state=random_state)
    clf_gnb_aug = GaussianNB()
    clf_svm_aug.fit(X_res_aug_s, y_res)
    clf_gnb_aug.fit(X_res_aug_s, y_res)
    aug_svm = eval_model(clf_svm_aug, X_test_aug_s, y_test)
    aug_gnb = eval_model(clf_gnb_aug, X_test_aug_s, y_test)

    # demo table (first 6 test samples): PCA components + memberships + label
    demo_df = pd.DataFrame(X_test_pca[:6], columns=[f"pc{i}" for i in range(X_test_pca.shape[1])])
    members_df = pd.DataFrame(members_test[:6], columns=[f"mem_k{i}" for i in range(members_test.shape[1])])
    demo_out = pd.concat([demo_df, members_df], axis=1)
    demo_out["label"] = y_test[:6]

    results = {
        "baseline_svm": baseline_svm,
        "baseline_gnb": baseline_gnb,
        "aug_svm": aug_svm,
        "aug_gnb": aug_gnb,
        "used_smote": used_smote,
        "demo_table": demo_out,
        "classifiers": {
            "clf_svm": clf_svm,
            "clf_gnb": clf_gnb,
            "clf_svm_aug": clf_svm_aug,
            "clf_gnb_aug": clf_gnb_aug,
        },
        "scalers": {
            "scaler": scaler,
            "scaler_aug": scaler_aug,
            "pca": pca,
            "gmm": gmm
        }
    }
    return results


# -------------------------
# Main: build dataset and run
# -------------------------
def main():
    # Prepare dataset (you can change counts to change class imbalance)
    counts = (210, 70, 20)
    length = 2048
    sr = 8000
    n_bins = 16
    X, y, df = prepare_dataset(counts=counts, length=length, sr=sr, n_bins=n_bins, seed=RND)

    print("Initial class distribution:", pd.Series(y).value_counts().sort_index().to_dict())
    results = run_pipeline(X, y, test_size=0.3, random_state=RND, pca_n_components=10, use_smote=True)

    print("\nBaseline SVM:", results["baseline_svm"])
    print("Baseline GNB:", results["baseline_gnb"])
    print("\nAugmented SVM:", results["aug_svm"])
    print("Augmented GNB:", results["aug_gnb"])
    print("\nSMOTE used?:", results["used_smote"])

    print("\nDemo (first 6 test samples: PCA comps + EM memberships + label):")
    print(results["demo_table"].round(4).to_string(index=False))

    # Optional: show a classification report for the augmented SVM on the test set
    clf = results["classifiers"]["clf_svm_aug"]
    # to produce the classification report we need the test set used internally; regenerate it quickly:
    # NOTE: this duplicates some steps to compute y_test and X_test_aug_s again for printing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RND)
    scaler = results["scalers"]["scaler"]
    X_train_s = scaler.transform(X_train); X_test_s = scaler.transform(X_test)
    pca = results["scalers"]["pca"]
    X_test_pca = pca.transform(X_test_s)
    members_test = results["scalers"]["gmm"].predict_proba(X_test_pca)
    X_test_aug = np.hstack([X_test_pca, members_test])
    X_test_aug_s = results["scalers"]["scaler_aug"].transform(X_test_aug)
    y_pred = clf.predict(X_test_aug_s)
    print("\nClassification report (SVM, augmented pipeline) on test set:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save a small CSV summary
    summary = pd.DataFrame({
        "model": ["SVM_baseline", "GNB_baseline", "SVM_augmented", "GNB_augmented"],
        "accuracy": [results["baseline_svm"]["accuracy"], results["baseline_gnb"]["accuracy"],
                     results["aug_svm"]["accuracy"], results["aug_gnb"]["accuracy"]],
        "macro_f1": [results["baseline_svm"]["macro_f1"], results["baseline_gnb"]["macro_f1"],
                     results["aug_svm"]["macro_f1"], results["aug_gnb"]["macro_f1"]],
        "macro_recall": [results["baseline_svm"]["macro_recall"], results["baseline_gnb"]["macro_recall"],
                         results["aug_svm"]["macro_recall"], results["aug_gnb"]["macro_recall"]]
    })
    out_csv = "example_pipeline_results_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary CSV to: {out_csv}")


if __name__ == "__main__":
    main()
