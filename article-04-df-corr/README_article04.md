# Article 04 — df.corr() · What It's Actually Computing

**Standalone article in The Missing Link series.**

📝 [Read on Medium](https://medium.com/@ashtosh.shenoy/what-df-corr-is-actually-computing-10cbe1b90234)

---

## What This Article Covers

Most people run `df.corr()` as EDA decoration. This article opens the function and shows the exact linear algebra it executes — and follows that thread all the way to multicollinearity, Gram-Schmidt, QR factorization, eigen decomposition, SVD, and PCA.

| Concept | What the article shows |
|---------|----------------------|
| `df.corr()` | Computes the cosine of the angle between every pair of mean-centered feature columns — a Gram matrix of cosines |
| Dot product | `corr(x, y) = (x−x̄)ᵀ(y−ȳ) / (‖x−x̄‖·‖y−ȳ‖)` — verified manually against pandas output |
| Orthogonality | Zero correlation = perpendicular vectors, not statistical independence |
| Gram-Schmidt | Projecting out the "size effect" from `bedrooms` — creating `bedrooms_residual` with corr = 0.000000 |
| Multicollinearity | Why corr > 0.999 causes `(XᵀX)⁻¹` to blow up — weight signs flip under 1% noise |
| QR factorization | `Rw* = Qᵀy` solved by back-substitution — why `model.fit()` never inverts a matrix |
| Eigen decomposition | `C = QΛQᵀ` on the correlation matrix — near-zero eigenvalues as the multicollinearity signal |
| SVD | `X = UΣVᵀ` — why `σᵢ²/n = λᵢ` (StandardScaler uses ddof=0) |
| PCA | Globally optimal orthogonal directions — not the same as Gram-Schmidt on existing columns |

---

## Files

```text
article-04-df-corr/
│
├── df_corr.py                  # Main visualization script
│   │                           # Generates all 8 figures in Visuals/
│   └── run: python df_corr.py
│
├── Visuals/
│   ├── fig0_heatmap.png                  # df.corr() output — house price dataset
│   ├── fig1_three_angles.png             # cos(θ) = corr for 3 feature pairs
│   ├── fig2_zero_corr_not_indep.png      # y = x² counterexample
│   ├── fig3_weight_instability.png       # Multicollinearity catastrophe
│   ├── fig4_gram_schmidt.png             # Projection + orthogonalization
│   ├── fig5_eigenvalues.png              # Eigenvalue bar + cumulative variance
│   ├── fig6_svd_vs_eigen.png             # Two paths, same eigenvalues
│   └── fig7_conceptual_chain.png         # Full article arc
│
├── 4b.py                        # Article 4B visualization script
│
└── README_article04.md         # Combined documentation for Articles 4A and 4B
```

---

## Dataset

Synthetic house price dataset — 200 samples, 6 features, 1 target. Same seed (`np.random.seed(42)`) used throughout.

| Feature | Description | Notes |
|---------|-------------|-------|
| `size` | Floor area (sq ft) | Normally distributed |
| `bedrooms` | Number of bedrooms | Derived from size — correlated |
| `bathrooms` | Number of bathrooms | Derived from bedrooms — correlated |
| `age` | Age of house (years) | Independent |
| `distance` | Distance from city (km) | Independent |
| `plot_area` | Plot area (sq ft) | Derived from size — correlated |
| `price` | House price (₹) | Target |

---

## Run Instructions

```bash
cd article-04-df-corr
mkdir -p Visuals
python df_corr.py
```

Expected output:

```text
✓  fig0_heatmap.png
✓  fig1_three_angles.png
✓  fig2_zero_corr_not_indep.png
✓  fig3_weight_instability.png
✓  fig4_gram_schmidt.png
✓  fig5_eigenvalues.png
✓  fig6_svd_vs_eigen.png
✓  fig7_conceptual_chain.png

All 8 figures saved to Visuals/
```

---

## Key Technical Notes

- **`StandardScaler` uses `ddof=0`** (population std). So `XᵀX/n = C` exactly, and `σᵢ²/n = λᵢ`. Using `n-1` here is wrong.
- **`df.corr()` ≠ eigen decomposition.** `df.corr()` computes C. Eigen decomposition is a separate analysis applied to C.
- **Zero correlation ≠ independence.** `y = x²` has corr = 0 with x but is a deterministic function of it.
- **Gram-Schmidt ≠ PCA.** Gram-Schmidt orthogonalizes existing columns in an arbitrary order. PCA finds globally optimal new directions that maximize variance.
- **QR solve: `Rw* = Qᵀy`**, not `w* = R⁻¹Qᵀy`. Back-substitution on an upper triangular system — no inverse computed.
- **`fig3` uses a local demo pair** (`size_demo`, `plot_demo` with corr = 0.999998), not the global dataset. This keeps the global figures clean while demonstrating the catastrophic instability dramatically.

---

## Dependencies

See `requirements.txt` in the root directory.

```text
numpy
pandas
matplotlib
scikit-learn
```

---

# Article 4B — What Happens When a Matrix Has an Inverse That Is Just Its Transpose?

**Companion article in The Missing Link series.**

*Orthogonal matrices, QR factorisation, eigendecomposition, PCA and the mathematics behind numerical stability.*

> Article 4B extends the discussion of orthogonality from feature relationships to orthogonal transformations and matrices. It focuses on the identity `Q⁻¹ = Qᵀ` for square real orthogonal matrices, then connects that property to geometry, QR factorisation, symmetric eigendecomposition, PCA and least-squares numerical stability.

## What This Article Covers

| Concept | What the article explains |
|---------|---------------------------|
| Orthogonal matrices | The defining condition `QᵀQ = QQᵀ = I` and why it implies `Q⁻¹ = Qᵀ` |
| Geometric preservation | Orthogonal transformations preserve vector norms, inner products, angles and Euclidean distances |
| Rectangular orthonormal matrices | In reduced QR, `QᵀQ = I` but Q is generally not square and has no ordinary inverse |
| QR factorisation | Decomposing a full-column-rank matrix as `A = QR`, with orthonormal columns in Q and an upper-triangular R |
| Least squares | Using QR to solve `Rx = Qᵀb` without explicitly forming `(AᵀA)⁻¹` |
| Symmetric eigendecomposition | Expressing a real symmetric matrix as `A = QΛQᵀ` |
| PCA | Using covariance-matrix eigenvectors as principal directions and projecting data with `Z = XQₖ` |
| Numerical conditioning | Why forming normal equations squares the 2-norm condition number: `κ₂(AᵀA) = κ₂(A)²` |
| SVD preview | Introducing `A = UΣVᵀ` as the next decomposition in the series |

## Visualizations

The Article 4B visualization script is `4b.py`. The figures support the article's geometric explanations.

| Figure | What it demonstrates |
|--------|----------------------|
| Figure 1 | Original orthonormal basis and the same basis after an orthogonal rotation, illustrating preservation of lengths and angles |
| Figure 2 | Orthonormal columns and the QR factorisation relationship |
| Figure 3 | Perpendicular eigenvector directions of a real symmetric matrix and the associated scaling |

The figures follow the visual style of the earlier article: dark navy background, light labels, restrained grid lines, and distinct colours for original and transformed directions.

**Figure 1 mathematical connection:**

- `QᵀQ = I`
- `‖Qe₁‖ = ‖e₁‖`
- `‖Qe₂‖ = ‖e₂‖`
- `(Qe₁)ᵀ(Qe₂) = 0`

## Files

```text
article-04-df-corr/
│
├── df_corr.py                  # Article 4A visualizations
├── 4b.py                       # Article 4B visualizations
├── Visuals/
│   ├── ...                     # Article 4A figures
│   └── ...                     # Article 4B figures
│
└── README_article04.md         # Combined documentation for Articles 4A and 4B
```

## Running the Article 4B Visualizations

From the repository root:

```bash
cd article-04-df-corr
python 4b.py
```

The actual output filenames and any additional dependencies should be recorded here once the implementation is finalized.

The project-level dependencies are listed in the root `requirements.txt`.

## Mathematical and Implementation Notes

- **The inverse-transpose identity applies to square orthogonal matrices.** For a real square matrix Q, `QᵀQ = QQᵀ = I`, so `Q⁻¹ = Qᵀ`.
- **Do not apply ordinary inverse notation to thin Q in reduced QR.** For a rectangular matrix with orthonormal columns, `QᵀQ = I`, but Q has no ordinary two-sided inverse. `Qᵀ` is a left inverse.
- **Orthogonality is not statistical independence.** A zero inner product, or zero covariance under the relevant centering and scaling, does not generally establish independence.
- **QR does not make an ill-conditioned problem well-conditioned.** It avoids the additional conditioning penalty caused by explicitly forming `AᵀA`; the original problem may still be sensitive.
- **Condition-number identity.** For a full-column-rank matrix, `κ₂(AᵀA) = κ₂(A)²`. The general 2-norm condition number is `κ₂(A) = σₘₐₓ(A) / σₘᵢₙ(A)`.
- **PCA data convention.** The covariance expression `C = XᵀX/(n−1)` assumes X has been mean-centred and uses sample covariance.
- **Full versus truncated PCA.** Keeping all principal components is a coordinate transformation; retaining only the first k components gives a lower-dimensional approximation.

## Article Links

- **Article 4A:** [What df.corr() Is Actually Computing](https://medium.com/@ashtosh.shenoy/what-df-corr-is-actually-computing-10cbe1b90234)
- **Article 4B:** Add the Medium URL here after publication.

---

## Shared Dependencies

See `requirements.txt` in the repository root. The Article 4A visualizations use:

```text
numpy
pandas
matplotlib
scikit-learn
```
