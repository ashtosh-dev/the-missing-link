# 🔗 The Missing Link

**Connecting every formula in my AIML syllabus to its real-world machine learning equivalent — one module at a time.**

Most AIML students live in two worlds simultaneously:

- **In class** — computing Σx, Σy, correlation coefficients, and regression tables by hand for 20 minutes
- **On YouTube / Kaggle** — calling `.fit()` and watching models train in 3 lines of Python

Nobody connects them. This series is that connection.

Each article takes a topic directly from the college syllabus and maps every formula, every table column, and every normal equation to its real-world ML equivalent — with the math, the code, and the visualization all in one place.

---

## 📖 Article Series

| # | Article | Topics Covered | Status |
|---|---|---|---|
| 1 | [Your Stats Professor Is Teaching You Machine Learning. They Just Forgot to Mention It.](https://medium.com/@ashtosh.shenoy) | Correlation, Lines of Regression, r, sklearn bridge | ✅ Published |
| 2 | When a Straight Line Lies: The Curve Fitting Chapter Nobody Explains | Curve Fitting, Normal Equations, Cost Function, Gradient Descent, Underfitting, Overfitting, R² | 🔜 Coming Soon |
| 3 | The Missing Link — Linear Mappings and Matrices | Linear Transformations, Matrix Representation, Change of Basis, PCA | 🔜 Coming Soon |
| 4 | The Missing Link — Inner Product Spaces | Inner Product, Orthogonality, Gram-Schmidt, QR Factorization | 🔜 Coming Soon |
| 5 | The Missing Link — SVD | Singular Value Decomposition, Dimensionality Reduction | 🔜 Coming Soon |
| 6 | The Missing Link — Probability Distributions | Binomial, Normal, Confidence Scores in ML | 🔜 Coming Soon |
| 7 | The Missing Link — Sampling Theory | Central Limit Theorem, Training on Sample Data | 🔜 Coming Soon |

---

## 📁 Repository Structure

```
the-missing-link/
│
├── article-01-regression/
│   ├── Visuals/
│   │   ├── fig00_regression_lines.png     # Two regression lines + r gauge
│   │   ├── fig01_angle_proof.png          # Angle between lines = visual proof of r
│   │   ├── fig02_sklearn_vs_manual.png    # sklearn vs manual: identical result
│   │   └── fig04_straight_line_fail.png   # Straight line predicting above 100
│   └── 00regression_visualization.py      # Generates all Article 1 figures
│
├── article-02-curve-fitting/
│   ├── Visuals/
│   │   ├── fig0_straight_line_fail.png    # Straight line breaks on curved data
│   │   ├── fig1_three_types.png           # All 3 curve types on same data
│   │   ├── fig2_overfit_underfit.png      # Underfitting vs just right vs overfitting
│   │   ├── fig3_bias_variance.png         # Train R² vs Test R² across degrees
│   │   └── fig4_manual_vs_sklearn.png     # Manual table vs sklearn: identical
│   ├── curve_fitting_examples.py          # All teaching code from article, runnable
│   ├── curve_fitting_visualization.py     # Generates fig1–fig4
│   └── fig0_straight_line_fail.py         # Generates fig0
│
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ashtosh-dev/the-missing-link
cd the-missing-link
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Code

### Article 1 — Regression

```bash
cd article-01-regression
python 00regression_visualization.py
```

**Expected output:**
```
Manual →  r = 0.9286,  y = 0.9286x + 7.2857
sklearn → y = 0.9286x + 7.2857

Figures saved to Visuals/
```

---

### Article 2 — Curve Fitting

**Teaching code** — follow along with the article, see all manual vs sklearn comparisons:

```bash
cd article-02-curve-fitting
python curve_fitting_examples.py
```

**Expected output (excerpt):**
```
Type 1 (manual):  y = 6.5758x + 38.3333
Type 1 (sklearn): y = 6.5758x + 38.3333
→ Identical: True

Type 2 (manual):  y = 36.0142 · x^0.4519
Type 2 (sklearn): y = 36.0142 · x^0.4519
→ Identical: True

Type 3 (manual):  y = -0.8068x² + 15.4508x + 20.5833
Type 3 (sklearn): y = -0.8068x² + 15.4508x + 20.5833
→ Identical: True

Degree 1: Train R²=0.8971  Test R²=0.9456  underfit
Degree 2: Train R²=0.9996  Test R²=0.9996  just right ✓
Degree 9: Train R²=1.0000  Test R²=0.6658  completely overfit

→ Same destination. Different journey.
```

**Visualizations** — generate all article figures:

```bash
python curve_fitting_visualization.py  # fig1 to fig4
python fig0_straight_line_fail.py      # fig0
```

---

## 🧰 Tech Stack

- Python 3.x
- NumPy
- Scikit-learn
- Matplotlib
- Pandas
- SciPy

---

## 📚 Further Reading

- [sklearn LinearRegression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [sklearn PolynomialFeatures Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [Pearson Correlation Coefficient — Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

---

## 🤝 Connect

- 📝 Medium: [medium.com/@ashtosh.shenoy](https://medium.com/@ashtosh.shenoy)
- 💼 LinkedIn: [linkedin.com/in/ashutosh-shenoy](https://www.linkedin.com/in/ashutosh-shenoy/)

---

*Written in real time as I move through my AIML syllabus. New articles and code drop as new modules are covered in class.*
