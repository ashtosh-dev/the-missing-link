# Articles 2A & 2B — Curve Fitting

**2A: When a Straight Line Lies: The Curve Fitting Chapter Nobody Explains**
📝 [Read on Medium](https://medium.com/@ashtosh.shenoy/when-a-straight-line-lies-the-curve-fitting-chapter-nobody-explains-93abcb329c6a)

**2B: Which Curve Do You Trust? The Part of Curve Fitting Nobody Teaches**
📝 [Read on Medium](https://medium.com/@ashtosh.shenoy/which-curve-do-you-trust-the-part-of-curve-fitting-nobody-teaches-d0c211703707)

**Topics:** Curve Fitting (3 types), Normal Equations, Cost Function, Log Trick, Manual vs sklearn, R², Train/Test Split, Underfitting, Overfitting, Gradient Descent

---

## 📁 Files

```
article-02-curve-fitting/
├── Visuals/
│   ├── fig0_straight_line_fail.png    # Hook: straight line breaks on curved data
│   ├── fig1_three_types.png           # All 3 curve types on same data
│   ├── fig2_overfit_underfit.png      # Underfitting vs just right vs overfitting
│   ├── fig3_bias_variance.png         # Train R² vs Test R² across degrees
│   └── fig4_manual_vs_sklearn.png     # Manual table vs sklearn: identical
├── curve_fitting_examples.py          # All teaching code from 2A and 2B, runnable
├── curve_fitting_visualization.py     # Generates fig1–fig4
└── fig0_straight_line_fail.py         # Generates fig0
```

---

## ▶️ Run

**Teaching code** — follow along with both articles:

```bash
cd article-02-curve-fitting
python curve_fitting_examples.py
```

**Expected output (excerpt):**
```
Type 1 (manual):  y = 6.3455x + 28.5273
Type 1 (sklearn): y = 6.3455x + 28.5273
→ Identical: True

Type 2 (manual):  y = 32.1874 · x^0.5312
Type 2 (sklearn): y = 32.1874 · x^0.5312
→ Identical: True

Type 3 (manual):  y = -0.8182x² + 14.2727x + 21.9091
Type 3 (sklearn): y = -0.8182x² + 14.2727x + 21.9091
→ Identical: True

Degree 1: Train R²=0.8971  Test R²=0.9456  underfit
Degree 2: Train R²=0.9996  Test R²=0.9996  just right ✓
Degree 9: Train R²=1.0000  Test R²=0.6658  completely overfit

→ Same destination. Different journey.
```

**Generate figures:**

```bash
python curve_fitting_visualization.py  # fig1 to fig4
python fig0_straight_line_fail.py      # fig0
```
