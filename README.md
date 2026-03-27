#  A DIMENSION-ROBUST LOGARITHMIC–CUBIC (DRLC) KERNEL FOR SUPPORT VECTOR MACHINES

Official implementation of the **A DIMENSION-ROBUST LOGARITHMIC–CUBIC (DRLC) KERNEL**, specifically engineered to overcome the "distance concentration" problem in high-dimensional classification tasks.

##  Scientific Foundation
The UVIK kernel applies a hybrid transformation—combining cubic volumetric expansion with logarithmic damping—to Euclidean distances ($r$):

$$K(x, y) = \exp(-\gamma \cdot \ln(1 + r^3))$$

### Key Benefits:
- **Dimension Independence:** High performance on both low-dimensional (d=4) and high-dimensional (d=1558) data.
- **Volumetric Margin:** The cubic component $r^3$ provides sharper class separation.
- **Numerical Stability:** The logarithmic term $\ln(1+r^3)$ prevents gradient issues during optimization.

##  Experimental Performance
Results obtained using **5-Fold Stratified Cross-Validation**:

### Performance Benchmarks (DRLC vs. RBF & Poly)

| Data Set | Feature (d) | DRLC (Proposed) | SVM-RBF | SVM-Poly | λ_min | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** | 4 | **96.67%** | 95.33% | 94.00% | -6.2e-03 | 0.9666 |
| **Wine** | 13 | **98.33%** | 98.33% | 96.08% | 4.9e-03 | 0.9833 |
| **Breast Cancer** | 30 | 97.01% | **97.89%** | 95.96% | 1.9e-03 | 0.9699 |
| **Ionosphere** | 34 | 94.59% | **95.16%** | 82.34% | -2.3e-03 | 0.9452 |
| **Sonar** | 60 | 84.16% | **88.47%** | 82.69% | 1.8e-02 | 0.8406 |
| **Gas Sensor** | 128 | 99.24% | **99.30%** | 96.55% | -2.1e-02 | 0.9924 |
| **Musk** | 166 | 99.97% | **100.00%** | 100.00% | -1.5e-03 | 0.9997 |
| **LSVT** | 309 | 84.09% | **89.66%** | 77.75% | 5.2e-02 | 0.8240 |
| **CNAE-9** | 856 | 82.59% | **82.78%** | 32.69% | -1.9e-15 | 0.8408 |
| **Internet Ads** | 1558 | 95.85% | **97.10%** | 95.67% | -1.2e-02 | 0.9557 |
| **Semeion** | 256 | 93.72% | **96.04%** | 94.92% | 3.8e-02 | 0.9374 |

##  Quick Start
1. Ensure you have `scikit-learn`, `numpy`, and `scipy` installed.
2. Run the benchmark:
```bash
python drlc_svm_main.py
