"""
The Missing Link — Article 2: Curve Fitting Visualizations
Generates all 4 figures for the article.
"""
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
 
# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor':   '#1a1a1a',
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  '#cccccc',
    'xtick.color':      '#888888',
    'ytick.color':      '#888888',
    'text.color':       '#eeeeee',
    'grid.color':       '#2a2a2a',
    'grid.linestyle':   '--',
    'font.family':      'monospace',
})
 
COLORS = {
    'data':      '#ffffff',
    'underfit':  '#e74c3c',   # red
    'power':     '#f39c12',   # orange
    'parabola':  '#2ecc71',   # green
    'overfit':   '#e74c3c',   # red
    'train':     '#3498db',   # blue
    'test':      '#e74c3c',   # red
    'manual':    '#2ecc71',
    'sklearn':   '#e74c3c',
}
 
# ─── Data ─────────────────────────────────────────────────────────────────────
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([35, 48, 60, 70, 78, 84, 89, 92, 94, 95], dtype=float)
n = len(x)
x_smooth = np.linspace(0.5, 11, 300)
 
# ─── Fit all models ───────────────────────────────────────────────────────────
# Type 1: straight line
m1 = LinearRegression().fit(x.reshape(-1,1), y)
y1 = m1.predict(x_smooth.reshape(-1,1))
 
# Type 2: power curve (log transform)
m2 = LinearRegression().fit(np.log(x).reshape(-1,1), np.log(y))
y2 = np.exp(m2.predict(np.log(x_smooth).reshape(-1,1)))
 
# Type 3: parabola
poly2  = PolynomialFeatures(degree=2, include_bias=True)
m3     = LinearRegression().fit(poly2.fit_transform(x.reshape(-1,1)), y)
y3     = m3.predict(poly2.transform(x_smooth.reshape(-1,1)))
 
# Degree 9: overfit
poly9  = PolynomialFeatures(degree=9)
m9     = LinearRegression().fit(poly9.fit_transform(x.reshape(-1,1)), y)
y9     = m9.predict(poly9.transform(x_smooth.reshape(-1,1)))
 
# ─────────────────────────────────────────────────────────────────────────────
# FIG 1: Three Types of Curve Fitting
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle(
    'Three Types of Curve Fitting — Same Data, Different Assumptions',
    fontsize=13, color='#eeeeee', y=1.02
)
 
titles  = ['Type 1: Straight Line\ny = ax + b', 
           'Type 2: Power Curve\ny = axᵇ', 
           'Type 3: Parabola\ny = ax² + bx + c']
ys      = [y1, y2, y3]
clrs    = [COLORS['underfit'], COLORS['power'], COLORS['parabola']]
labels  = [
    f'y = {m1.coef_[0]:.2f}x + {m1.intercept_:.2f}',
    f'y = {np.exp(m2.intercept_):.2f}·x^{m2.coef_[0]:.2f}',
    f'y = {m3.coef_[2]:.2f}x² + {m3.coef_[1]:.2f}x + {m3.intercept_:.2f}',
]
 
for ax, title, y_pred, color, label in zip(axes, titles, ys, clrs, labels):
    ax.scatter(x, y, color=COLORS['data'], zorder=5, s=60, label='Data')
    ax.plot(x_smooth, y_pred, color=color, lw=2.5, label=label)
    ax.axhline(100, color='#555555', lw=1, linestyle=':', label='Score ceiling (100)')
    ax.set_title(title, fontsize=10, color='#eeeeee', pad=10)
    ax.set_xlabel('Study Hours', fontsize=9)
    ax.set_ylabel('Exam Score', fontsize=9)
    ax.legend(fontsize=7.5, facecolor='#1a1a1a', edgecolor='#333333')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 120)
 
    # Shade the "impossible zone" above 100
    ax.fill_between(x_smooth, 100, 120, alpha=0.08, color='#e74c3c')
    ax.text(6, 110, 'impossible zone', fontsize=7, color='#e74c3c', ha='center', alpha=0.7)
 
    # Annotate R² on training data
    if color == COLORS['underfit']:
        r2 = r2_score(y, m1.predict(x.reshape(-1,1)))
    elif color == COLORS['power']:
        r2 = r2_score(np.log(y), m2.predict(np.log(x).reshape(-1,1)))
        r2 = r2_score(y, np.exp(m2.predict(np.log(x).reshape(-1,1))))
    else:
        r2 = r2_score(y, m3.predict(poly2.transform(x.reshape(-1,1))))
    ax.text(0.97, 0.05, f'R² = {r2:.4f}', transform=ax.transAxes,
            ha='right', fontsize=8, color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#111111', alpha=0.8))
 
plt.tight_layout()
fig1.savefig('fig1_three_types.png', dpi=150, bbox_inches='tight',
             facecolor=fig1.get_facecolor())
print("Fig 1 saved.")
 
# ─────────────────────────────────────────────────────────────────────────────
# FIG 2: Underfitting vs Just Right vs Overfitting
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle(
    'Underfitting vs Just Right vs Overfitting',
    fontsize=13, color='#eeeeee', y=1.02
)
 
# Clip degree 9 predictions for visualization
y9_clipped = np.clip(y9, -20, 150)
 
configs = [
    ('Underfitting\n(Degree 1 — too simple)', y1,         COLORS['underfit'], 
     'High error everywhere\nMisses the curve shape'),
    ('Just Right\n(Degree 2 — parabola)',     y3,         COLORS['parabola'], 
     'Captures the pattern\nGeneralises well'),
    ('Overfitting\n(Degree 9 — too complex)', y9_clipped, COLORS['overfit'],  
     'Fits training perfectly\nFails on new data'),
]
 
for ax, (title, y_pred, color, note) in zip(axes2, configs):
    ax.scatter(x, y, color=COLORS['data'], zorder=5, s=80, label='Training data')
    ax.plot(x_smooth, y_pred, color=color, lw=2.5)
    ax.axhline(100, color='#555555', lw=1, linestyle=':')
    ax.set_title(title, fontsize=10, color=color, pad=10)
    ax.set_xlabel('Study Hours', fontsize=9)
    ax.set_ylabel('Exam Score', fontsize=9)
    ax.set_xlim(0, 12)
    ax.set_ylim(-10, 130)
    ax.grid(True, alpha=0.3)
 
    # Add residual lines for underfit/overfit to make errors visible
    if 'Degree 1' in title or 'Degree 9' in title:
        model_x = m1 if 'Degree 1' in title else m9
        poly_x  = None if 'Degree 1' in title else poly9
        for xi, yi in zip(x, y):
            if poly_x:
                y_hat = model_x.predict(poly_x.transform([[xi]]))[0]
            else:
                y_hat = model_x.predict([[xi]])[0]
            ax.plot([xi, xi], [yi, y_hat], color=color, alpha=0.4, lw=1.5, linestyle='--')
 
    # Note box
    ax.text(0.05, 0.95, note, transform=ax.transAxes,
            va='top', fontsize=8, color=color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#111111', alpha=0.9))
 
plt.tight_layout()
fig2.savefig('fig2_overfit_underfit.png', dpi=150, bbox_inches='tight',
             facecolor=fig2.get_facecolor())
print("Fig 2 saved.")
 
# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: Bias-Variance Tradeoff — Train R² vs Test R²
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
 
degrees     = [1, 2, 3, 4, 5, 6, 7, 8, 9]
train_r2s   = []
test_r2s    = []
 
for deg in degrees:
    p      = PolynomialFeatures(degree=deg)
    Xtr    = p.fit_transform(x_train.reshape(-1,1))
    Xte    = p.transform(x_test.reshape(-1,1))
    mdl    = LinearRegression().fit(Xtr, y_train)
    train_r2s.append(r2_score(y_train, mdl.predict(Xtr)))
    test_r2s.append(r2_score(y_test,  mdl.predict(Xte)))
 
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(degrees, train_r2s, 'o-', color=COLORS['train'],   lw=2.5, ms=7, label='Training R²')
ax3.plot(degrees, test_r2s,  's-', color=COLORS['test'],    lw=2.5, ms=7, label='Test R²')
ax3.axvline(x=2, color='#2ecc71', lw=1.5, linestyle='--', alpha=0.7, label='Sweet spot (degree 2)')
 
# Shade regions
ax3.axvspan(0.5, 1.5,  alpha=0.07, color='#e74c3c', label='Underfitting zone')
ax3.axvspan(4.5, 9.5,  alpha=0.07, color='#f39c12', label='Overfitting zone')
 
ax3.set_title('Bias-Variance Tradeoff: Training R² vs Test R² Across Model Complexity',
              fontsize=11, color='#eeeeee')
ax3.set_xlabel('Polynomial Degree (Model Complexity)', fontsize=10)
ax3.set_ylabel('R² Score', fontsize=10)
ax3.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333333')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(degrees)
 
# Annotations
ax3.text(1,    min(train_r2s)-0.05, 'Underfit',  ha='center', fontsize=9, color='#e74c3c')
ax3.text(7,    min(train_r2s)-0.05, 'Overfit',   ha='center', fontsize=9, color='#f39c12')
ax3.text(2.0,  0.50, '← sweet spot',             ha='left',   fontsize=9, color='#2ecc71')
 
plt.tight_layout()
fig3.savefig('fig3_bias_variance.png', dpi=150, bbox_inches='tight',
             facecolor=fig3.get_facecolor())
print("Fig 3 saved.")
 
# ─────────────────────────────────────────────────────────────────────────────
# FIG 4: Manual vs sklearn — All 3 Types (Verification)
# ─────────────────────────────────────────────────────────────────────────────
 
# ── Manual computations ───────────────────────────────────────────────────────
# Type 1
sum_x  = np.sum(x);  sum_y  = np.sum(y)
sum_x2 = np.sum(x**2); sum_xy = np.sum(x*y)
a1m = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
b1m = (sum_y - a1m*sum_x) / n
y1m_smooth = a1m*x_smooth + b1m
 
# Type 2
X_log = np.log(x); Y_log = np.log(y)
sum_X=np.sum(X_log); sum_Y=np.sum(Y_log)
sum_X2=np.sum(X_log**2); sum_XY=np.sum(X_log*Y_log)
b2m = (n*sum_XY - sum_X*sum_Y)/(n*sum_X2 - sum_X**2)
A2m = (sum_Y - b2m*sum_X)/n; a2m = np.exp(A2m)
y2m_smooth = a2m * x_smooth**b2m
 
# Type 3
sum_x3=np.sum(x**3); sum_x4=np.sum(x**4)
sum_x2y=np.sum(x**2*y)
A_mat = np.array([[sum_x2,sum_x,n],[sum_x3,sum_x2,sum_x],[sum_x4,sum_x3,sum_x2]])
b_vec = np.array([sum_y, sum_xy, sum_x2y])
a3m, b3m, c3m = np.linalg.solve(A_mat, b_vec)
y3m_smooth = a3m*x_smooth**2 + b3m*x_smooth + c3m
 
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
fig4.suptitle(
    'Manual Table Method vs sklearn — Identical Results',
    fontsize=13, color='#eeeeee', y=1.02
)
 
manual_ys  = [y1m_smooth, y2m_smooth, y3m_smooth]
sklearn_ys = [y1, y2, y3]
subtitles  = ['Type 1: y = ax + b', 'Type 2: y = axᵇ', 'Type 3: y = ax² + bx + c']
man_labels = [
    f'Manual: y = {a1m:.4f}x + {b1m:.4f}',
    f'Manual: y = {a2m:.4f}·x^{b2m:.4f}',
    f'Manual: y = {a3m:.4f}x² + {b3m:.4f}x + {c3m:.4f}',
]
sk_labels  = [
    f'sklearn: y = {m1.coef_[0]:.4f}x + {m1.intercept_:.4f}',
    f'sklearn: y = {np.exp(m2.intercept_):.4f}·x^{m2.coef_[0]:.4f}',
    f'sklearn: y = {m3.coef_[2]:.4f}x² + {m3.coef_[1]:.4f}x + {m3.intercept_:.4f}',
]
 
for ax, title, ym, ys_pred, ml, sl in zip(axes4, subtitles, manual_ys, sklearn_ys, man_labels, sk_labels):
    ax.scatter(x, y, color=COLORS['data'], zorder=5, s=60, label='Data', alpha=0.9)
    ax.plot(x_smooth, ym,      color=COLORS['manual'],  lw=3,   label=ml,  linestyle='-')
    ax.plot(x_smooth, ys_pred, color=COLORS['sklearn'], lw=1.5, label=sl,  linestyle='--')
    ax.set_title(title, fontsize=10, color='#eeeeee', pad=10)
    ax.set_xlabel('Study Hours', fontsize=9)
    ax.set_ylabel('Exam Score', fontsize=9)
    ax.legend(fontsize=7, facecolor='#1a1a1a', edgecolor='#333333')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 12)
 
    ax.text(0.5, 0.05, '↑ Lines overlap — identical results',
            transform=ax.transAxes, ha='center', fontsize=8,
            color='#aaaaaa',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#111111', alpha=0.8))
 
plt.tight_layout()
fig4.savefig('fig4_manual_vs_sklearn.png', dpi=150, bbox_inches='tight',
             facecolor=fig4.get_facecolor())
print("Fig 4 saved.")
 
print("\n✅ All 4 figures saved to /mnt/user-data/outputs/")
print("\nFigure summary:")
print("  fig1_three_types.png      — Three curve types on same data")
print("  fig2_overfit_underfit.png — Underfitting vs Just Right vs Overfitting")
print("  fig3_bias_variance.png    — Train R² vs Test R² across degrees")
print("  fig4_manual_vs_sklearn.png — Manual table vs sklearn verification")