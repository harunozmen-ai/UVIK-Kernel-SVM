#  UVIK: Universal Volumetric Integral Kernel for SVM

Official implementation of the **Universal Volumetric Integral Kernel (UVIK)**, specifically engineered to overcome the "distance concentration" problem in high-dimensional classification tasks.

##  Scientific Foundation
The UVIK kernel applies a hybrid transformation—combining cubic volumetric expansion with logarithmic damping—to Euclidean distances ($r$):

$$K(x, y) = \exp(-\gamma \cdot \ln(1 + r^3))$$

### Key Benefits:
- **Dimension Independence:** High performance on both low-dimensional (d=4) and high-dimensional (d=60) data.
- **Volumetric Margin:** The cubic component $r^3$ provides sharper class separation.
- **Numerical Stability:** The logarithmic term $\ln(1+r^3)$ prevents gradient issues during optimization.

##  Experimental Performance
Results obtained using **5-Fold Stratified Cross-Validation**:

| Dataset | Dimensions (d) | Accuracy (%) ± STD | F1-Score ± STD | Cohen's Kappa ± STD |
|---------|----------------|-------------------|----------------|---------------------|
| **Iris** | 4 | 96.67 ± 3.65 | 0.9666 ± 0.0366 | 0.9500 ± 0.0548 |
| **Wine** | 13 | 98.33 ± 2.22 | 0.9833 ± 0.0222 | 0.9748 ± 0.0336 |
| **Breast Cancer** | 30 | 97.01 ± 1.19 | 0.9699 ± 0.0121 | 0.9354 ± 0.0262 |
| **Ionosphere** | 34 | 94.59 ± 1.40 | 0.9452 ± 0.0142 | 0.8801 ± 0.0313 |
| **Sonar** | 60 | 84.16 ± 3.51 | 0.8406 ± 0.0349 | 0.6805 ± 0.0697 |

##  Quick Start
1. Ensure you have `scikit-learn`, `numpy`, and `scipy` installed.
2. Run the benchmark:
```bash
python uvik_svm_main.py
