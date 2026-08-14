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

```
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
└── README_article04.md         # This file
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

```
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

```
numpy
pandas
matplotlib
scikit-learn
```
