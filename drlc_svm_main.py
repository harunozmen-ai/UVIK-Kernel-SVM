"""
Title: A DIMENSION-ROBUST LOGARITHMIC–CUBIC (DRLC) KERNEL FOR SUPPORT VECTOR MACHINES
Author: Harun ÖZMEN (harun.ozmen@ahievran.edu.tr)
Official Implementation for SVM Classification

How to Cite:
Ozmen, H. (2026). A DIMENSION-ROBUST LOGARITHMIC–CUBIC (DRLC) KERNEL FOR SUPPORT VECTOR MACHINES: 


Description:
This script implements the DRLC kernel which solves the "distance concentration" 
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
from sklearn.metrics import make_scorer, cohen_kappa_score
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt

# ==========================================
# 1. DRLC CORE FUNCTION
# ==========================================
def drlc_kernel_matrix(X1, X2, gamma=0.01):
    """A DIMENSION-ROBUST LOGARITHMIC–CUBIC (DRLC) KERNEL  Matrix Calculation"""
    r = cdist(X1, X2, 'euclidean')
    psi_r = np.log1p(np.power(r, 3))  # ln(1 + r^3)
    return np.exp(-gamma * psi_r)

# ==========================================
# 2. EXPERIMENT ENGINE (FULL METRICS + DASHBOARD)
# ==========================================
def run_benchmark():
    datasets_info = [
        ("Iris", load_iris),
        ("Wine", load_wine),
        ("Breast Cancer", load_breast_cancer),
        ("Ionosphere", lambda: fetch_openml(name='ionosphere', version=1, as_frame=False, parser='liac-arff')),
        ("Sonar", lambda: fetch_openml(name='sonar', version=1, as_frame=False, parser='liac-arff')),
        ("Gas Sensor", lambda: fetch_openml(data_id=1476, as_frame=False, parser='liac-arff')),
        ("Musk", lambda: fetch_openml(name='musk', version=1, as_frame=False, parser='liac-arff')),
        ("LSVT", lambda: fetch_openml(data_id=1484, as_frame=False, parser='liac-arff')),
        ("CNAE-9", lambda: fetch_openml(name='cnae-9', version=1, as_frame=False, parser='liac-arff')),
        ("Internet Ads", lambda: fetch_openml(data_id=40978, as_frame=False, parser='liac-arff')),
        ("Semeion", lambda: fetch_openml(name='semeion', version=1, as_frame=False, parser='liac-arff'))
    ]

    scoring = {
        'accuracy': 'accuracy',
        'f1_weighted': 'f1_weighted',
        'kappa': make_scorer(cohen_kappa_score)
    }

    drlc_acc_list, rbf_acc_list, poly_acc_list = [], [], []

    # # DASHBOARD PREPARATION (3 rows, 4 columns)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    print("\n DRLC Benchmark (RBF vs POLY)...")
    print("-" * 120)
    header = f"{'Data Set':<15} | {'Acc (%) ± STD':<18} | {'F1-Score':<10} | {'Kappa':<10} | {'λ_min':<10} | {'λ_max':<10} | {'RBF Acc%':<10} | {'Poly Acc%':<10}"
    print(header)
    print("-" * 120)

    for i, (name, loader) in enumerate(datasets_info):
        try:
            # 1. Load & preprocess
            data = loader()
            X, y = data.data, data.target

            if isinstance(y[0], str) or y.dtype.kind in 'UO':
                y = LabelEncoder().fit_transform(y)

            X_scaled = StandardScaler().fit_transform(np.nan_to_num(X))
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # 2. DRLC Kernel
            K_drlc= drlc_kernel_matrix(X_scaled, X_scaled, gamma=0.01)
            eigenvalues = np.linalg.eigvalsh(K_drlc)
            eigenvalues_sorted = sorted(np.abs(eigenvalues), reverse=True)

            lam_min = np.min(eigenvalues)
            lam_max = np.max(eigenvalues)

            # Dashboard plot
            ax = axes[i]
            ax.semilogy(range(1, len(eigenvalues_sorted)+1), eigenvalues_sorted,
                        color='royalblue', linewidth=1.5)
            ax.axhline(y=1e-10, color='red', linestyle='--', alpha=0.6)
            ax.set_title(f"{name}", fontsize=14, fontweight='bold')
            ax.grid(True, which="both", ls="-", alpha=0.2)

            if i % 4 == 0:
                ax.set_ylabel("Eigenvalue Magnitude")
            if i >= 7:
                ax.set_xlabel("Index")

            # 3. Performance

            # DRLC
            u_results = cross_validate(
                SVC(kernel='precomputed', C=10),
                K_drlc, y,
                cv=cv,
                scoring=scoring
            )

            # RBF
            rbf_results = cross_validate(
                SVC(kernel='rbf', C=10, gamma='scale'),
                X_scaled, y,
                cv=cv,
                scoring='accuracy'
            )

            # POLY
            poly_results = cross_validate(
                SVC(kernel='poly', degree=3, C=10, gamma='scale'),
                X_scaled, y,
                cv=cv,
                scoring='accuracy'
            )

            u_acc = u_results['test_accuracy'].mean()
            r_acc = rbf_results['test_score'].mean()
            p_acc = poly_results['test_score'].mean()

            drlc_acc_list.append(u_acc)
            rbf_acc_list.append(r_acc)
            poly_acc_list.append(p_acc)

            print(f"{name:<15} | "
                  f"{u_acc*100:5.2f} ±{u_results['test_accuracy'].std()*100:4.2f} | "
                  f"{u_results['test_f1_weighted'].mean():.4f} | "
                  f"{u_results['test_kappa'].mean():.4f} | "
                  f"{lam_min:<10.1e} | "
                  f"{lam_max:<10.1f} | "
                  f"{r_acc*100:<10.2f} | "
                  f"{p_acc*100:<10.2f}")

        except Exception as e:
            print(f" Error in {name}: {e}")

    #  subplot
    if len(datasets_info) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig("Figure4_DRLC_Dashboard.png", dpi=300)
    print("\n Figure 4 Dashboard saved as 'Figure4_UVIK_Dashboard.png'")

    # 4. Statistical Tests
    print("-" * 120)
    print("\n STATISTICAL SIGNIFICANCE RESULTS")

    _, p_wil = wilcoxon(drlc_acc_list, rbf_acc_list)
    _, p_fri = friedmanchisquare(drlc_acc_list, rbf_acc_list, poly_acc_list)

    print(f"Wilcoxon p-value (DRLC vs RBF): {p_wil:.5f}")
    print(f"Friedman p-value (Overall)   : {p_fri:.5f}")

    if p_wil < 0.05:
        print(" CONCLUSION: DRLC's success is statistically SIGNIFICANT.")

    print("\n The experiments are complete.")

if __name__ == "__main__":
    run_benchmark()

    

