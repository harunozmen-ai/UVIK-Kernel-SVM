"""
Title: LOGARITHMIC-CUBIC TRANSFORMED UNIVERSAL VOLUMETRIC INTEGRAL KERNEL (UVIK)
Author: Harun ÖZMEN (harun.ozmen@ahievran.edu.tr)
Official Implementation for SVM Classification

How to Cite:
Ozmen, H. (2026). Logarithmic-Cubic Transformed Universal Volumetric Integral Kernel (UVIK): 
A Dimension-Independent Hybrid SVM Approach. [Journal Name/GitHub Link]

Description:
This script implements the UVIK kernel which solves the "distance concentration" 
problem in high-dimensional spaces using volumetric expansion and logarithmic damping.
Formula: K(x, y) = exp(-gamma * ln(1 + ||x-y||^3))

"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from scipy.spatial.distance import cdist
from sklearn.metrics import make_scorer, f1_score, cohen_kappa_score

# ==========================================
# 1. UNIVERSAL UVIK CORE FUNCTION
# ==========================================
def uvik_kernel_matrix(X1, X2, gamma=0.01):
    """
    Universal Log-Cubic UVIK Kernel Matrix Calculation.
    Logic: Combines cubic volumetric expansion with logarithmic damping.
    """
    r = cdist(X1, X2, 'euclidean')
    psi_r = np.log1p(np.power(r, 3)) # ln(1 + r^3)
    return np.exp(-gamma * psi_r)

# ==========================================
# 2. EXPERIMENT ENGINE (ALL METRICS)
# ==========================================
def run_benchmark():
    # List of datasets to be tested
    datasets = [
        ("Iris", load_iris),
        ("Wine", load_wine),
        ("Breast Cancer", load_breast_cancer),
        ("Ionosphere", lambda: fetch_openml(name='ionosphere', version=1, as_frame=False, parser='liac-arff')),
        ("Sonar", lambda: fetch_openml(name='sonar', version=1, as_frame=False, parser='liac-arff'))
    ]

    # Define the metrics.
    scoring = {
        'accuracy': 'accuracy',
        'f1_weighted': 'f1_weighted',
        'kappa': make_scorer(cohen_kappa_score)
    }

    print("\n UVIK Universal Benchmark Launched (Full Metrics Analysis)...")
    print("-" * 110)
    header = f"{'Data Set':<15} | {'(d)':<5} | {'Acc (%) ± STD':<18} | {'F1-Score ± STD':<18} | {'Kappa ± STD':<15}"
    print(header)
    print("-" * 110)

    for name, loader in datasets:
        try:
            # 1. Upload Data
            data = loader()
            X, y = data.data, data.target
            
            # Digitize labels (Label Encoding)
            if isinstance(y[0], str) or y.dtype.kind in 'UO': 
                y = LabelEncoder().fit_transform(y)

            # 2. Pre-processing
            X_scaled = StandardScaler().fit_transform(np.nan_to_num(X))
            d = X_scaled.shape[1]

            # 3. Calculate UVIK Matrix
            K = uvik_kernel_matrix(X_scaled, X_scaled, gamma=0.01)

            # 4. 5- Train and Test the Model with Fold CV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            model = SVC(kernel='precomputed', C=10)
            
            # Multiple metric calculations
            scores = cross_validate(model, K, y, cv=cv, scoring=scoring)

            # 5. Calculate Statistics
            acc_str = f"{scores['test_accuracy'].mean()*100:.2f} ±{scores['test_accuracy'].std()*100:.2f}"
            f1_str  = f"{scores['test_f1_weighted'].mean():.4f} ±{scores['test_f1_weighted'].std():.4f}"
            kap_str = f"{scores['test_kappa'].mean():.4f} ±{scores['test_kappa'].std():.4f}"

            # 6. Print to screen
            print(f"{name:<15} | {d:<5} | {acc_str:<18} | {f1_str:<18} | {kap_str:<15}")

        except Exception as e:
            print(f"❌ Error in {name}: {e}")

    print("-" * 110)
    print("\n All experiments are complete.")

if __name__ == "__main__":
    run_benchmark()
    