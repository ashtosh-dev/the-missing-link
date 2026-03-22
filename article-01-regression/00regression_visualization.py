"""
regression_visualization.py
============================
Visualization code for the Medium article:
"The Missing Link Between My Stats Class and My AI Degree"

Author  : Ashutosh Shenoy
GitHub  : https://github.com/ashtosh-shenoy
Article : https://medium.com/@ashtosh.shenoy

Description:
    Reproduces the exact classroom regression calculations manually,
    verifies them against sklearn, and generates three publication-ready
    figures used in the article.

Figures generated:
    fig1_regression_lines.png  — Two regression lines + correlation gauge
    fig2_angle_proof.png       — Angle between lines as visual proof of r
    fig3_sklearn_vs_manual.png — sklearn vs manual: identical result

Requirements:
    pip install numpy matplotlib scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ── Colour palette ────────────────────────────────────────────────────────────
BG_DARK   = '#0f0f1a'
BG_PANEL  = '#1a1a2e'
SPINE_COL = '#444444'
C_BLUE    = '#00d4ff'   # data points
C_RED     = '#ff6b6b'   # y on x line
C_YELLOW  = '#ffd93d'   # x on y line
C_GREEN   = '#6bcb77'   # sklearn / perfect correlation


def set_dark_axes(ax):
    """Apply consistent dark-theme styling to an axes object."""
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)


# ── Classroom data ─────────────────────────────────────────────────────────────
#    Source: Module 3, Statistical Methods — worked example from class notes
x = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
y = np.array([9, 8, 10, 12, 11, 13, 14], dtype=float)
n = len(x)


# ── Manual calculation (Type 1 method — z = x − y shortcut) ───────────────────
x_mean = np.mean(x)         # x̄ = 4.0
y_mean = np.mean(y)         # ȳ = 11.0
z      = x - y              # shortcut column

sigma_x2 = np.sum(x**2)/n - x_mean**2          # σx² = 4.0
sigma_y2 = np.sum(y**2)/n - y_mean**2          # σy² = 4.0
sigma_z2 = np.sum(z**2)/n - np.mean(z)**2      # σz² = 0.5714

# Karl Pearson's correlation coefficient
r = (sigma_x2 + sigma_y2 - sigma_z2) / (2 * np.sqrt(sigma_x2 * sigma_y2))

# Regression coefficient (y on x) and line
byx         = r * np.sqrt(sigma_y2) / np.sqrt(sigma_x2)   # slope
a_yonx      = y_mean - byx * x_mean                        # intercept

# Regression coefficient (x on y) — expressed as y = f(x) for plotting
bxy                  = r * np.sqrt(sigma_x2) / np.sqrt(sigma_y2)
a_xony               = x_mean - bxy * y_mean
slope_xony_as_y      = 1 / bxy
intercept_xony_as_y  = -a_xony / bxy

print("=" * 50)
print("MANUAL CALCULATION RESULTS")
print("=" * 50)
print(f"  x̄ = {x_mean},  ȳ = {y_mean}")
print(f"  σx² = {sigma_x2},  σy² = {sigma_y2},  σz² = {sigma_z2:.4f}")
print(f"  r   = {r:.4f}")
print(f"  Line y on x : y = {byx:.4f}x + {a_yonx:.4f}")
print(f"  Line x on y : x = {bxy:.4f}y + {a_xony:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Two Regression Lines + Correlation Gauge
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.patch.set_facecolor(BG_DARK)
for ax in axes1:
    set_dark_axes(ax)

x_line = np.linspace(0, 8, 200)

# Left panel — scatter plot with both regression lines
ax = axes1[0]
ax.scatter(x, y, color=C_BLUE, s=90, zorder=5, label='Data points')
ax.plot(x_line, byx * x_line + a_yonx,
        color=C_RED, lw=2.5, label=f'y on x  →  y = {byx:.2f}x + {a_yonx:.2f}')
ax.plot(x_line, slope_xony_as_y * x_line + intercept_xony_as_y,
        color=C_YELLOW, lw=2.5, linestyle='--',
        label=f'x on y  →  y = {slope_xony_as_y:.2f}x + {intercept_xony_as_y:.2f}')
# Mean lines and intersection
ax.axvline(x_mean, color='#777', linestyle=':', lw=1)
ax.axhline(y_mean, color='#777', linestyle=':', lw=1)
ax.scatter([x_mean], [y_mean], color='white', s=150, zorder=6,
           marker='*', label=f'(x̄, ȳ) = ({x_mean:.0f}, {y_mean:.0f})')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title(f'Two Lines of Regression  |  r = {r:.2f}', fontsize=13, fontweight='bold')
ax.legend(facecolor=BG_PANEL, labelcolor='white', fontsize=9, loc='upper left')
ax.set_xlim(0, 8)
ax.set_ylim(6, 16)

# Right panel — correlation gauge bar
ax2 = axes1[1]
r_range  = np.linspace(-1, 1, 300)
for rv in r_range:
    col = plt.cm.RdYlGn((rv + 1) / 2)
    ax2.bar(rv, 1, width=0.01, color=col, alpha=0.85)
ax2.axvline(r, color='white', lw=3)
ax2.text(r + 0.04, 1.15, f'r = {r:.2f}\n(Strong +ve)',
         color='white', fontsize=11, fontweight='bold')
ax2.text(-0.97, 0.45, 'Perfect\nNegative', color='white', fontsize=8, ha='left')
ax2.text( 0.97, 0.45, 'Perfect\nPositive', color='white', fontsize=8, ha='right')
ax2.text(-0.02, 0.45, 'No\nCorr.', color='white', fontsize=8, ha='center')
ax2.set_xlim(-1.05, 1.05)
ax2.set_ylim(0, 1.6)
ax2.set_xlabel('Correlation Coefficient (r)', fontsize=12)
ax2.set_title('Where Does Our r Land?', fontsize=13, fontweight='bold')
ax2.set_yticks([])

plt.tight_layout()
fig1.savefig('fig1_regression_lines.png', dpi=150,
             bbox_inches='tight', facecolor=BG_DARK)
print("\n[✓] fig1_regression_lines.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Angle Between Lines = Visual Proof of r
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.patch.set_facecolor(BG_DARK)
fig2.suptitle(
    'The "Two Lines" Mystery — Angle Between Lines is the Visual Proof of r',
    color='white', fontsize=13, fontweight='bold', y=1.02
)

scenarios = [
    {
        'r':     0.30,
        'title': 'Weak Correlation  (r = 0.3)\nWide angle — model is guessing',
    },
    {
        'r':     0.93,
        'title': f'Our Classroom Data  (r = 0.93)\nNarrow angle — model is confident',
    },
    {
        'r':     1.00,
        'title': 'Perfect Correlation  (r = 1.0)\nLines overlap — model is certain',
    },
]

xl = np.linspace(-3, 3, 200)

for ax3, sc in zip(axes2, scenarios):
    set_dark_axes(ax3)
    rv   = sc['r']
    sx   = sy = 2.0                         # equal variances for clean visuals
    byx_ = rv * sy / sx                     # slope y on x
    bxy_ = rv * sx / sy                     # slope x on y
    slope_inv = 1 / bxy_ if abs(bxy_) > 1e-9 else 1e9

    ax3.plot(xl, byx_ * xl,   color=C_RED,    lw=2.5, label='y on x')
    ax3.plot(xl, slope_inv * xl, color=C_YELLOW, lw=2.5,
             linestyle='--', label='x on y')
    ax3.axhline(0, color='#555', lw=0.5)
    ax3.axvline(0, color='#555', lw=0.5)
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-4, 4)
    ax3.set_title(sc['title'], color='white', fontsize=10, fontweight='bold')
    ax3.legend(facecolor=BG_PANEL, labelcolor='white', fontsize=9)

plt.tight_layout()
fig2.savefig('fig2_angle_proof.png', dpi=150,
             bbox_inches='tight', facecolor=BG_DARK)
print("[✓] fig2_angle_proof.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — sklearn vs Manual: Identical Result
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax4 = plt.subplots(figsize=(9, 5))
fig3.patch.set_facecolor(BG_DARK)
set_dark_axes(ax4)

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

print("\n" + "=" * 50)
print("SKLEARN VERIFICATION")
print("=" * 50)
print(f"  slope     = {model.coef_[0]:.4f}  (manual: {byx:.4f})")
print(f"  intercept = {model.intercept_:.4f}  (manual: {a_yonx:.4f})")
match = np.isclose(model.coef_[0], byx, atol=1e-4) and \
        np.isclose(model.intercept_, a_yonx, atol=1e-4)
print(f"  Results match: {match}")

ax4.scatter(x, y, color=C_BLUE, s=100, zorder=5, label='Classroom data points')
ax4.plot(x_line,
         model.predict(x_line.reshape(-1, 1)),
         color=C_RED, lw=2.5, linestyle='--',
         label=f'sklearn  →  y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}')
ax4.plot(x_line, byx * x_line + a_yonx,
         color=C_GREEN, lw=2.5,
         label=f'Manual   →  y = {byx:.4f}x + {a_yonx:.4f}')

ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.set_title('sklearn vs. Manual Calculation — Same Line, Same Numbers',
              fontsize=13, fontweight='bold', color='white')
ax4.legend(facecolor=BG_PANEL, labelcolor='white', fontsize=10)
ax4.set_xlim(0, 8)
ax4.set_ylim(6, 16)

plt.tight_layout()
fig3.savefig('fig3_sklearn_vs_manual.png', dpi=150,
             bbox_inches='tight', facecolor=BG_DARK)
print("[✓] fig3_sklearn_vs_manual.png saved")

print("\n✅ All figures generated successfully.")
print("   → fig1_regression_lines.png")
print("   → fig2_angle_proof.png")
print("   → fig3_sklearn_vs_manual.png")