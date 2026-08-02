# 🔗 The Missing Link

**Connecting every formula in my AIML syllabus to its real-world machine learning equivalent — one module at a time.**

Most AIML students live in two worlds simultaneously:

- **In class** — computing Σx, Σy, correlation coefficients, and regression tables by hand
- **In projects** — calling `.fit()` and watching models train in 3 lines of Python

Nobody connects them. This series is that connection.

Each article takes a topic directly from the college syllabus and maps every formula to its real-world ML equivalent — with the math, the code, and the visualization all in one place.

---

## 📖 Article Series

| #  | Article | Topics Covered | Status |
|----|---------|----------------|--------|
| 1 | [Your Stats Professor Is Teaching You Machine Learning. They Just Forgot to Mention It.](https://medium.com/@ashtosh.shenoy) | Correlation, Lines of Regression, r, sklearn bridge | ✅ Published |
| 2A | [When a Straight Line Lies: The Curve Fitting Chapter Nobody Explains](https://medium.com/@ashtosh.shenoy/when-a-straight-line-lies-the-curve-fitting-chapter-nobody-explains-93abcb329c6a) | Curve Fitting (3 types), Normal Equations, Cost Function, Log Trick, Manual vs sklearn | ✅ Published |
| 2B | [Which Curve Do You Trust? The Part of Curve Fitting Nobody Teaches](https://medium.com/@ashtosh.shenoy/which-curve-do-you-trust-the-part-of-curve-fitting-nobody-teaches-d0c211703707) | Underfitting, Overfitting, R², Train/Test Split, Gradient Descent | ✅ Published |
| 3A | [Your AIML Class Is Teaching You Neural Networks. Nobody Mentioned It.](https://medium.com/@ashtosh.shenoy/bf0a07e1861d) | Linear Transformations, Matrix Representation [T]\_S, Dense layer connection | ✅ Published |
| 3B | [Same Data. Different Coordinates. This Is What PCA Actually Does.](https://medium.com/@ashtosh.shenoy/f04bede0f99c) | Change of Basis, Similarity, PCA, DCT, Iris dataset | ✅ Published |
| 4A | [The Math You Already Know Is Running Your RAG Pipeline](https://medium.com/@ashtosh.shenoy/the-math-you-already-know-is-running-your-rag-pipeline-298eb4efa7a1) | Inner Product, Norm, Orthogonality, Cosine Similarity, Attention, RAG | ✅ Published |
| 4B | Orthogonal Projection, Best Approximation & Gram-Schmidt | Projection, Best Approximation Theorem, Gram-Schmidt Process | 🔜 Coming Soon |
| 5 | QR Factorization | QR Decomposition, Orthonormal Bases, Numerical Stability | 🔜 Coming Soon |
| 6 | SVD + PCA + House Price Dataset | Singular Value Decomposition, PCA via SVD, Full ML Pipeline | 🔜 Coming Soon |
| 7 | Probability Distributions | Binomial, Normal, Confidence Scores in ML | 🔜 Coming Soon |
| 8 | Sampling Theory | Central Limit Theorem, Training on Sample Data | 🔜 Coming Soon |

---

## 📁 Repository Structure

```
the-missing-link/
│
├── article-01-regression/          # Article 1 — see README inside
├── article-02-curve-fitting/       # Articles 2A & 2B — see README inside
├── article-03-linear-mapping/      # Articles 3A & 3B — see README inside
├── article-04-inner-products/      # Article 4A — see README inside
├── article-04-df-corr/             # df.corr() bridge article — see README inside
│
├── requirements.txt
└── README.md
```

Each article directory contains its own README with full file descriptions, run instructions, and expected output.

---

## 🚀 Setup

```bash
git clone https://github.com/ashtosh-dev/the-missing-link
cd the-missing-link
pip install -r requirements.txt
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

## 🤝 Connect

- 📝 Medium: [medium.com/@ashtosh.shenoy](https://medium.com/@ashtosh.shenoy)
- 💼 LinkedIn: [linkedin.com/in/ashutosh-shenoy](https://www.linkedin.com/in/ashutosh-shenoy/)

---

*Written in real time as I move through my AIML syllabus. New articles and code drop as new modules are covered in class.*
