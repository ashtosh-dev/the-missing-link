# Article 1 — Regression

**Your Stats Professor Is Teaching You Machine Learning. They Just Forgot to Mention It.**

📝 [Read on Medium](https://medium.com/@ashtosh.shenoy)

**Topics:** Correlation, Lines of Regression, r, sklearn bridge

---

## 📁 Files

```
article-01-regression/
├── Visuals/
│   ├── fig00_regression_lines.png     # Two regression lines + r gauge
│   ├── fig01_angle_proof.png          # Angle between lines = visual proof of r
│   ├── fig02_sklearn_vs_manual.png    # sklearn vs manual: identical result
│   └── fig04_straight_line_fail.png   # Straight line predicting above 100
└── 00regression_visualization.py      # Generates all Article 1 figures
```

---

## ▶️ Run

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
