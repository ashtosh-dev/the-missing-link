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
|---|---------|---------------|--------|
| 1 | [Your Stats Professor Is Teaching You Machine Learning. They Just Forgot to Mention It.](https://medium.com/@ashtosh.shenoy/your-stats-professor-is-teaching-you-machine-learning-they-just-forgot-to-mention-it-53a30bcfe02f) | Correlation, Lines of Regression, r, sklearn bridge | ✅ Published |
| 2A | [When a Straight Line Lies: The Curve Fitting Chapter Nobody Explains](https://medium.com/@ashtosh.shenoy/when-a-straight-line-lies-the-curve-fitting-chapter-nobody-explains-93abcb329c6a) | Curve Fitting (3 types), Normal Equations, Cost Function, Log Trick, Manual vs sklearn | ✅ Published |
| 2B | [Which Curve Do You Trust? The Part of Curve Fitting Nobody Teaches](https://medium.com/@ashtosh.shenoy/which-curve-do-you-trust-the-part-of-curve-fitting-nobody-teaches-d0c211703707) | Underfitting, Overfitting, R², Train/Test Split, Gradient Descent, Industry Gap | ✅ Published |
| 3A | [Your AIML Class Is Teaching You Neural Networks. Nobody Mentioned It.](https://medium.com/@ashtosh.shenoy/your-professor-is-teaching-you-neural-networks-theyre-calling-it-linear-mappings-bf0a07e1861d) | Linear Transformations, Matrix Representation [T]\_S, ML connection | ✅ Published |
| 3B | [Same Data. Different Coordinates. This Is What PCA Actually Does.](https://medium.com/@ashtosh.shenoy/f04bede0f99c) | Change of Basis, Similarity, PCA, DCT, Iris dataset | ✅ Published |
| 4A | [The Math You Already Know Is Running Your RAG Pipeline](https://medium.com/p/298eb4efa7a1) | Inner Products, Norms, Orthogonality, the measuring instrument hiding in plain sight | ✅ Published |
| — | [Your Features Are Not as Different as You Think](https://medium.com/@ashtosh.shenoy/what-df-corr-is-actually-computing-10cbe1b90234) | df.corr(), Dot Product, Gram-Schmidt, QR, Eigen Decomposition, SVD, PCA | ✅ Published · Standalone |
| 4B | The Missing Link — Inner Product Spaces (Part 2) | Gram-Schmidt, QR Factorization, Stability | 🔜 Coming Soon |
| 5 | The Missing Link — SVD | Singular Value Decomposition, Dimensionality Reduction | 🔜 Coming Soon |
| 6 | The Missing Link — Probability Distributions | Binomial, Normal, Confidence Scores in ML | 🔜 Coming Soon |
| 7 | The Missing Link — Sampling Theory | Central Limit Theorem, Training on Sample Data | 🔜 Coming Soon |

> **Standalone articles** are not numbered in the series arc but connect directly to upcoming numbered articles. They will be backlinked from the relevant numbered pieces.

---

## 📁 Repository Structure

```
the-missing-link/
│
├── article-01-regression/          # Article 1 — see README inside
├── article-02-curve-fitting/       # Articles 2A & 2B — see README inside
├── article-03-linear-mapping/      # Articles 3A & 3B — see README inside
├── article-04-df-corr/             # Standalone — df.corr() — see README inside
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
