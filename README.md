#  UVIK: Universal Volumetric Integral Kernel for SVM

Official implementation of the **Universal Volumetric Integral Kernel (UVIK)**, specifically engineered to overcome the "distance concentration" problem in high-dimensional classification tasks.

##  Scientific Foundation
The UVIK kernel applies a hybrid transformation—combining cubic volumetric expansion with logarithmic damping—to Euclidean distances ($r$):

$$K(x, y) = \exp(-\gamma \cdot \ln(1 + r^3))$$

### Key Benefits:
- **Dimension Independence:** High performance on both low-dimensional (d=4) and high-dimensional (d=60) data.
- **Volumetric Margin:** The cubic component $r^3$ provides sharper class separation.
- **Numerical Stability:** The logarithmic term $\ln(1+r^3)$ prevents gradient issues during optimization.

## 📊 Experimental Performance
Results obtained using **5-Fold Stratified Cross-Validation**:

| Dataset | Dimensions (d) | Accuracy (%) | F1-Score | Cohen's Kappa |
|---------|----------------|--------------|----------|---------------|
| **Iris** | 4 | 96.67 ±3.65 | 0.9666 | 0.9500 |
| **Wine** | 13 | 98.33 ±2.11 | 0.9832 | 0.9748 |
| **Breast Cancer** | 30 | 95.78 ±1.52 | 0.9576 | 0.9102 |
| **Sonar** | 60 | 84.16 ±3.51 | 0.8406 | 0.6805 |

##  Quick Start
1. Ensure you have `scikit-learn`, `numpy`, and `scipy` installed.
2. Run the benchmark:
```bash
python uvik_svm_main.py
