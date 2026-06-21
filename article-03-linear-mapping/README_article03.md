# Articles 3A & 3B — Linear Mappings and Matrices

**3A: Your AIML Class Is Teaching You Neural Networks. Nobody Mentioned It.**
📝 [Read on Medium](https://medium.com/@ashtosh.shenoy/bf0a07e1861d)

**3B: Same Data. Different Coordinates. This Is What PCA Actually Does.**
📝 [Read on Medium](https://medium.com/@ashtosh.shenoy)

**Topics:** Linear Transformations, Matrix Representation [T]_S, Change of Basis, Similarity of Matrices, PCA, StandardScaler, DCT, Iris dataset, ML/DL/CV connections

---

## 📁 Files

```
article-03-linear-mappings/
├── Visuals/
│   ├── fig0_vector_mapping.png            # 3A: A vector before and after transformation F
│   ├── fig1_grid_transformation.png       # 3A: Grid before and after F (side by side)
│   ├── fig2_T_S_construction.png          # 3A: Building [F]_E column by column
│   ├── fig0_same_point_two_bases.png      # 3B: Same point, two coordinate systems
│   ├── fig1_standard_scaler.png           # 3B: Data before/after StandardScaler
│   ├── fig2_pca_axes.png                  # 3B: PCA axes overlaid on data
│   ├── fig3_P_construction.png            # 3B: Building P matrix column by column
│   ├── fig4_similarity.png                # 3B: B = P⁻¹AP diagram
│   ├── fig5_iris_pca.png                  # 3B: Iris dataset in PCA coordinates
│   └── fig6_dct.png                       # 3B: Pixel basis vs DCT frequency basis
├── linear_mapping_visualization.py        # Generates all Article 3A figures
└── change_of_basis_visualization.py       # Generates all Article 3B figures
```

---

## ▶️ Run

**Article 3A figures:**

```bash
cd article-03-linear-mappings
python linear_mapping_visualization.py
```

**Expected output:**
```
fig0 saved → Visuals/fig0_vector_mapping.png
fig1 saved → Visuals/fig1_grid_transformation.png
fig2 saved → Visuals/fig2_T_S_construction.png

Done. All figures in Visuals/
```

**Article 3B figures:**

```bash
cd article-03-linear-mappings
python change_of_basis_visualization.py
```

**Expected output:**
```
fig0 saved → Visuals/fig0_same_point_two_bases.png
fig1 saved → Visuals/fig1_standard_scaler.png
fig2 saved → Visuals/fig2_pca_axes.png
fig3 saved → Visuals/fig3_P_construction.png
fig4 saved → Visuals/fig4_similarity.png
fig5 saved → Visuals/fig5_iris_pca.png
fig6 saved → Visuals/fig6_dct.png

Done. All figures in Visuals/
```

---

## 🧠 Key Concepts (3B)

- **StandardScaler** is an affine transformation (centers + rescales) — not a full change of basis
- **PCA** finds new orthogonal basis vectors from data variance — a genuine change of basis
- **`pca.transform(X)`** internally computes `(X − μ) · Pᵀ` — mean-centering matters
- **Similarity** (`B = P⁻¹AP`) connects two matrices representing the same transformation in different bases — trace and determinant are invariant
- **DCT vs PCA**: DCT (used in JPEG) is data-independent (fixed cosine basis); PCA is data-dependent (learned from data). DCT approximates PCA fast enough for real-time compression.
