"""
The Missing Link — Article 2: Curve Fitting
All code snippets from the article, in order, runnable as a single file.

Data: Study hours vs Exam score
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [35, 48, 60, 70, 78, 84, 89, 92, 94, 95]
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ─── Data ─────────────────────────────────────────────────────────────────────
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([35, 48, 60, 70, 78, 84, 89, 92, 94, 95], dtype=float)
n = len(x)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TYPE 1: STRAIGHT LINE  y = ax + b
# The classroom method vs sklearn
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TYPE 1: STRAIGHT LINE  y = ax + b")
print("=" * 60)

# ── Manual method (exactly what the classroom table computes) ──
sum_x  = np.sum(x)        # Σx
sum_y  = np.sum(y)        # Σy
sum_x2 = np.sum(x**2)     # Σx²
sum_xy = np.sum(x * y)    # Σxy

# Normal equations:
# Σy  = aΣx  + nb      →  745  = 55a  + 10b
# Σxy = aΣx² + bΣx     →  4261 = 385a + 55b
# Solve simultaneously:
a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
b1 = (sum_y - a1 * sum_x) / n

print(f"\nManual (normal equations):")
print(f"  Σx={sum_x}, Σy={sum_y}, Σx²={sum_x2}, Σxy={sum_xy}")
print(f"  a = {a1:.4f},  b = {b1:.4f}")
print(f"  y = {a1:.4f}x + {b1:.4f}")

# ── sklearn equivalent ──
model1 = LinearRegression().fit(x.reshape(-1, 1), y)
print(f"\nsklearn:")
print(f"  y = {model1.coef_[0]:.4f}x + {model1.intercept_:.4f}")

print(f"\n→ Identical: {np.isclose(a1, model1.coef_[0])}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TYPE 2: POWER CURVE  y = ax^b
# The log linearisation trick
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TYPE 2: POWER CURVE  y = ax^b")
print("Log linearisation: ln y = ln a + b·ln x  →  Y = A + bX")
print("=" * 60)

# ── Manual method ──
# Step 1: Transform to log space
X_log = np.log(x)    # X = ln x
Y_log = np.log(y)    # Y = ln y

# Step 2: Apply straight line normal equations on transformed data
sum_X  = np.sum(X_log)
sum_Y  = np.sum(Y_log)
sum_X2 = np.sum(X_log**2)
sum_XY = np.sum(X_log * Y_log)

b2 = (n * sum_XY - sum_X * sum_Y) / (n * sum_X2 - sum_X**2)
A2 = (sum_Y - b2 * sum_X) / n

# Step 3: Convert back — a = e^A
a2 = np.exp(A2)

print(f"\nManual (log transform + normal equations):")
print(f"  After log transform: ΣX={sum_X:.4f}, ΣY={sum_Y:.4f}")
print(f"  A = {A2:.4f}  →  a = e^A = {a2:.4f}")
print(f"  b = {b2:.4f}")
print(f"  y = {a2:.4f} · x^{b2:.4f}")

# ── sklearn equivalent ──
model2 = LinearRegression().fit(X_log.reshape(-1, 1), Y_log)
a2_sk  = np.exp(model2.intercept_)
b2_sk  = model2.coef_[0]

print(f"\nsklearn (fit in log space, convert back):")
print(f"  model.fit(log(x), log(y))")
print(f"  a = exp(intercept) = {a2_sk:.4f}")
print(f"  b = coef[0]        = {b2_sk:.4f}")
print(f"  y = {a2_sk:.4f} · x^{b2_sk:.4f}")

print(f"\n→ Identical: {np.isclose(a2, a2_sk) and np.isclose(b2, b2_sk)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TYPE 3: PARABOLA  y = ax² + bx + c
# 7-column table → 3 normal equations → 3 unknowns
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TYPE 3: PARABOLA  y = ax² + bx + c")
print("7-column table: x, y, x², x³, x⁴, xy, x²y")
print("=" * 60)

# ── Manual method (7-column table) ──
sum_x3  = np.sum(x**3)      # Σx³
sum_x4  = np.sum(x**4)      # Σx⁴
sum_x2y = np.sum(x**2 * y)  # Σx²y

print(f"\n7-column table sums:")
print(f"  Σx={sum_x}, Σy={sum_y}, Σx²={sum_x2}")
print(f"  Σx³={sum_x3}, Σx⁴={sum_x4}")
print(f"  Σxy={sum_xy}, Σx²y={sum_x2y}")

# Normal equations (3x3 system):
# Σy   = aΣx²  + bΣx  + nc
# Σxy  = aΣx³  + bΣx² + cΣx
# Σx²y = aΣx⁴  + bΣx³ + cΣx²
A_mat = np.array([
    [sum_x2, sum_x,  n      ],
    [sum_x3, sum_x2, sum_x  ],
    [sum_x4, sum_x3, sum_x2 ]
])
b_vec = np.array([sum_y, sum_xy, sum_x2y])
a3, b3, c3 = np.linalg.solve(A_mat, b_vec)

print(f"\nManual (solve 3×3 normal equations):")
print(f"  a = {a3:.4f},  b = {b3:.4f},  c = {c3:.4f}")
print(f"  y = {a3:.4f}x² + {b3:.4f}x + {c3:.4f}")

# ── sklearn equivalent ──
poly    = PolynomialFeatures(degree=2, include_bias=False)
X_poly  = poly.fit_transform(x.reshape(-1, 1))
model3  = LinearRegression().fit(X_poly, y)
a3_sk   = model3.coef_[1]   # coefficient of x²
b3_sk   = model3.coef_[0]   # coefficient of x
c3_sk   = model3.intercept_ # constant

print(f"\nsklearn (PolynomialFeatures + LinearRegression):")
print(f"  poly = PolynomialFeatures(degree=2)")
print(f"  model.fit(poly.fit_transform(x), y)")
print(f"  a = coef_[1]   = {a3_sk:.4f}")
print(f"  b = coef_[0]   = {b3_sk:.4f}")
print(f"  c = intercept_ = {c3_sk:.4f}")
print(f"  y = {a3_sk:.4f}x² + {b3_sk:.4f}x + {c3_sk:.4f}")

print(f"\n→ Identical: {np.isclose(a3, a3_sk) and np.isclose(b3, b3_sk) and np.isclose(c3, c3_sk)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ALL THREE SIDE BY SIDE (the article's main code block)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ALL THREE TYPES — SUMMARY")
print("=" * 60)

print(f"\nType 1 (manual):  y = {a1:.4f}x + {b1:.4f}")
print(f"Type 1 (sklearn): y = {model1.coef_[0]:.4f}x + {model1.intercept_:.4f}")

print(f"\nType 2 (manual):  y = {a2:.4f} · x^{b2:.4f}")
print(f"Type 2 (sklearn): y = {a2_sk:.4f} · x^{b2_sk:.4f}")

print(f"\nType 3 (manual):  y = {a3:.4f}x² + {b3:.4f}x + {c3:.4f}")
print(f"Type 3 (sklearn): y = {a3_sk:.4f}x² + {b3_sk:.4f}x + {c3_sk:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OVERFITTING: DEGREE 9
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("OVERFITTING — Degree 9 polynomial on 10 data points")
print("=" * 60)

poly9   = PolynomialFeatures(degree=9)
X_poly9 = poly9.fit_transform(x.reshape(-1, 1))
model9  = LinearRegression().fit(X_poly9, y)

r2_train_9 = r2_score(y, model9.predict(X_poly9))
print(f"\nDegree 9 on training data:")
print(f"  R² = {r2_train_9:.6f}  ← perfect fit, memorised the data")
print(f"  J  = {np.sum((y - model9.predict(X_poly9))**2):.8f}  ← near zero")
print(f"\nBut ask it to predict at x=11:")
print(f"  y = {model9.predict(poly9.transform([[11]]))[0]:.2f}  ← absurd answer")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAIN/TEST SPLIT: R² ACROSS DEGREES
# The bias-variance tradeoff made visible
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TRAIN vs TEST R² ACROSS POLYNOMIAL DEGREES")
print("R² on training data = lie. R² on test data = truth.")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    x.reshape(-1, 1), y, test_size=0.3, random_state=42
)

print(f"\n{'Degree':<10} {'Train R²':<15} {'Test R²':<15} {'Verdict'}")
print("-" * 55)

for degree in [1, 2, 3, 5, 9]:
    p      = PolynomialFeatures(degree=degree)
    X_tr   = p.fit_transform(X_train)
    X_te   = p.transform(X_test)
    mdl    = LinearRegression().fit(X_tr, y_train)
    tr_r2  = r2_score(y_train, mdl.predict(X_tr))
    te_r2  = r2_score(y_test,  mdl.predict(X_te))

    if degree == 1:
        verdict = "underfit"
    elif degree == 2:
        verdict = "just right ✓"
    elif degree == 3:
        verdict = "slightly complex"
    elif degree == 5:
        verdict = "overfitting begins"
    else:
        verdict = "completely overfit"

    print(f"{degree:<10} {tr_r2:<15.4f} {te_r2:<15.4f} {verdict}")

print(f"\n→ Degree 2 wins — not highest training R², but highest test R².")
print(f"→ .score() in sklearn returns R². Always evaluate on test data.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — GRADIENT DESCENT FROM SCRATCH (Type 1)
# Shows what .fit() is doing iteratively
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("GRADIENT DESCENT FROM SCRATCH — Type 1 (y = ax + b)")
print("Same answer as normal equations. Different journey.")
print("=" * 60)

# Cost function
def cost(a, b, x, y):
    return np.mean((y - (a * x + b))**2)

# Gradient descent
a_gd, b_gd = 0.0, 0.0   # start at random point
alpha       = 0.01        # learning rate
iterations  = 5000

for i in range(iterations):
    y_pred  = a_gd * x + b_gd
    error   = y_pred - y
    # Update rule: move in direction that reduces J
    a_gd -= alpha * (2/n) * np.sum(error * x)
    b_gd -= alpha * (2/n) * np.sum(error)

print(f"\nAfter {iterations} iterations (α = {alpha}):")
print(f"  Gradient descent: y = {a_gd:.4f}x + {b_gd:.4f}")
print(f"  Normal equations: y = {a1:.4f}x + {b1:.4f}")
print(f"  sklearn:          y = {model1.coef_[0]:.4f}x + {model1.intercept_:.4f}")
print(f"\n→ Same destination. Different journey.")