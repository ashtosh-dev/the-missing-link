"""
The Missing Link — "Your Features Are Not as Different as You Think"
Visualization script v3: incorporates all five Gemini fixes.

Changes from v2:
  - fix1: SVD section uses S**2/n (not n-1) — matches StandardScaler (ddof=0)
  - fix2: multicollinearity demo uses a LOCAL near-identical pair (corr>0.9999)
           so coefficient catastrophe is visually dramatic; global dataset unchanged
  - fix3: QR section framing updated in article; no figure change needed
  - fix4: PCA/GS nuance is textual; no figure change needed
  - fix5: formatting is Medium-side; no figure change needed

Run from inside article-04-df-corr/ after: mkdir -p Visuals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# ── THEME ───────────────────────────────────────────────────────────────
BG     = "#0f0f0f"
BLUE   = "#57c1ff"
RED    = "#ff6b6b"
YELLOW = "#ffd966"
GREEN  = "#b5e6a2"
PURPLE = "#c9b1ff"
GREY   = "#444444"
WHITE  = "#e8e8e8"
DIM    = "#888888"
FONT   = "monospace"

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_color(GREY)

def style_fig(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in np.array(axes).flat:
        style_ax(ax)

def save(name):
    plt.savefig(f"Visuals/{name}", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"✓  {name}")

# ── GLOBAL DATASET (unchanged from article) ─────────────────────────────
# This dataset feeds fig0, fig1, fig2, fig4, fig5, fig6, fig7.
# fig3 uses its OWN local pair — see fix2 below.
np.random.seed(42)
n = 200

size      = np.random.normal(1500, 400, n)
bedrooms  = (size / 400 + np.random.normal(0, 0.5, n)).clip(1, 6).round()
bathrooms = (bedrooms * 0.6 + np.random.normal(0, 0.3, n)).clip(1, 4).round()
age       = np.random.uniform(1, 50, n)
distance  = np.random.uniform(2, 30, n)
plot_area = size * np.random.uniform(1.1, 1.4, n)
price     = (size*80 + bedrooms*5000 - age*200
             - distance*300 + np.random.normal(0, 15000, n))

df = pd.DataFrame({
    "size": size, "bedrooms": bedrooms, "bathrooms": bathrooms,
    "age": age, "distance": distance, "plot_area": plot_area,
    "price": price
})

features = df.drop("price", axis=1)
corr     = features.corr()

# ══════════════════════════════════════════════════════════════════════════
# FIG 0 — Correlation heatmap
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5.8))
style_fig(fig, [ax])

cols = corr.columns.tolist()
nf   = len(cols)
data = corr.values

for i in range(nf):
    for j in range(nf):
        v = data[i, j]
        if v >= 0:
            r = int(BLUE[1:3],16)/255; g = int(BLUE[3:5],16)/255; b = int(BLUE[5:7],16)/255
            bg = (r*v+0.06*(1-v), g*v+0.06*(1-v), b*v+0.06*(1-v))
        else:
            iv = abs(v)
            r = int(RED[1:3],16)/255; g = int(RED[3:5],16)/255; b = int(RED[5:7],16)/255
            bg = (r*iv+0.06*(1-iv), g*iv+0.06*(1-iv), b*iv+0.06*(1-iv))
        rect = plt.Rectangle([j, nf-1-i], 1, 1,
                             facecolor=bg, edgecolor=BG, linewidth=1.5)
        ax.add_patch(rect)
        tc = WHITE if abs(v) > 0.3 else DIM
        ax.text(j+0.5, nf-0.5-i, f"{v:.2f}",
                ha="center", va="center",
                fontsize=9.5, fontfamily=FONT, color=tc, fontweight="bold")

ax.set_xlim(0, nf); ax.set_ylim(0, nf)
ax.set_xticks(np.arange(nf)+0.5); ax.set_yticks(np.arange(nf)+0.5)
ax.set_xticklabels(cols, fontfamily=FONT, fontsize=9, color=WHITE)
ax.set_yticklabels(cols[::-1], fontfamily=FONT, fontsize=9, color=WHITE)
ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("df.corr()  —  house price dataset", fontfamily=FONT,
             fontsize=12, color=WHITE, pad=14)
ax.text(0, -0.75,
        "correlated cluster:  size · bedrooms · bathrooms · plot_area   (top-left block)",
        fontfamily=FONT, fontsize=7.8, color=BLUE, ha="left")
ax.text(0, -1.15,
        "independent features:  age · distance   (near-zero off-diagonal)",
        fontfamily=FONT, fontsize=7.8, color=DIM, ha="left")
plt.tight_layout()
save("fig0_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Three angles: the geometric meaning of correlation values
# ══════════════════════════════════════════════════════════════════════════
pairs = [
    ("size",  "plot_area", BLUE,   RED,    "r = 0.96  →  nearly parallel"),
    ("size",  "bedrooms",  BLUE,   GREEN,  "r = 0.87  →  moderate angle"),
    ("age",   "distance",  YELLOW, PURPLE, "r ≈ 0.00  →  nearly perpendicular"),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
style_fig(fig, axes)

for ax, (f1, f2, c1, c2, label) in zip(axes, pairs):
    a  = df[f1].values - df[f1].mean()
    b  = df[f2].values - df[f2].mean()
    ua = a / np.linalg.norm(a)
    ub = b / np.linalg.norm(b)
    theta = np.arccos(np.clip(np.dot(ua, ub), -1, 1))

    v1 = np.array([1.0, 0.0])
    v2 = np.array([np.cos(theta), np.sin(theta)])

    ax.annotate("", xy=v1*1.08, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=c1, lw=2.2))
    ax.annotate("", xy=v2*1.08, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=c2, lw=2.2))
    ax.text(v1[0]*1.18, v1[1]-0.05, f1, fontfamily=FONT, fontsize=9,
            color=c1, ha="center")
    ax.text(v2[0]*1.2,  v2[1]+0.06, f2, fontfamily=FONT, fontsize=9,
            color=c2, ha="center")

    arc_t = np.linspace(0, theta, 80)
    ax.plot(0.32*np.cos(arc_t), 0.32*np.sin(arc_t), color=YELLOW, lw=1.4)
    deg = np.degrees(theta)
    ax.text(0.32*np.cos(theta/2)*1.35, 0.32*np.sin(theta/2)*1.35,
            f"θ ≈ {deg:.1f}°", fontfamily=FONT, fontsize=8.5,
            color=YELLOW, ha="center")

    r_real = df[f1].corr(df[f2])
    ax.text(0.05, 0.07, f"corr = {r_real:.2f}  =  cos({deg:.1f}°)",
            transform=ax.transAxes, fontfamily=FONT, fontsize=8, color=GREEN)

    ax.set_xlim(-0.2, 1.5); ax.set_ylim(-0.25, 1.15)
    ax.axhline(0, color=GREY, lw=0.5, ls="--")
    ax.axvline(0, color=GREY, lw=0.5, ls="--")
    ax.set_aspect("equal")
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_title(label, fontfamily=FONT, fontsize=9.5, color=WHITE)

plt.suptitle("correlation  =  cos(θ)  between mean-centered unit feature vectors",
             fontfamily=FONT, fontsize=11.5, color=WHITE, y=1.02)
plt.tight_layout()
save("fig1_three_angles.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Zero correlation ≠ statistical independence
# y = x² has exactly zero Pearson correlation but perfect dependence.
# Right panel shows WHY: the positive/negative dot product contributions
# cancel symmetrically, so the sum (= numerator of corr) is zero.
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
style_fig(fig, axes)

x_demo = np.linspace(-2, 2, 400)
y_demo = x_demo ** 2
r_demo = np.corrcoef(x_demo, y_demo)[0, 1]

ax = axes[0]
ax.scatter(x_demo[::4], y_demo[::4], color=BLUE, s=14, alpha=0.6, linewidths=0)
ax.plot(x_demo, y_demo, color=YELLOW, lw=1.5, alpha=0.7)
ax.axhline(0, color=GREY, lw=0.5, ls="--")
ax.axvline(0, color=GREY, lw=0.5, ls="--")
ax.set_xlabel("x", fontfamily=FONT, fontsize=10)
ax.set_ylabel("y = x²", fontfamily=FONT, fontsize=10)
ax.set_title("a perfect functional relationship", fontfamily=FONT,
             fontsize=10, color=WHITE)
ax.text(0.05, 0.88, f"corr(x, x²) = {r_demo:.4f}",
        transform=ax.transAxes, fontfamily=FONT, fontsize=9,
        color=RED, fontweight="bold")
ax.text(0.05, 0.78, "yet y is a deterministic\nfunction of x",
        transform=ax.transAxes, fontfamily=FONT, fontsize=8.5, color=DIM)

ax = axes[1]
x_c = x_demo - x_demo.mean()
y_c = y_demo - y_demo.mean()
contributions = x_c * y_c
pos_mask = contributions >= 0
neg_mask = contributions < 0
ax.bar(np.where(pos_mask)[0], contributions[pos_mask],
       color=BLUE, alpha=0.6, width=1.0, label="positive contribution")
ax.bar(np.where(neg_mask)[0], contributions[neg_mask],
       color=RED,  alpha=0.6, width=1.0, label="negative contribution")
ax.axhline(0, color=WHITE, lw=0.8)
ax.set_xlabel("sample index", fontfamily=FONT, fontsize=9)
ax.set_ylabel("(x−x̄)·(y−ȳ)  contribution", fontfamily=FONT, fontsize=9)
ax.set_title("positive and negative contributions cancel exactly",
             fontfamily=FONT, fontsize=10, color=WHITE)
ax.text(0.05, 0.88, f"sum = {np.sum(contributions):.4f}  →  corr = 0",
        transform=ax.transAxes, fontfamily=FONT, fontsize=8.5, color=YELLOW)
leg = ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor=GREY, labelcolor=WHITE)

plt.suptitle("zero correlation  ≠  independence\n"
             "orthogonality rules out linear overlap — not all relationships",
             fontfamily=FONT, fontsize=11, color=WHITE, y=1.04)
plt.tight_layout()
save("fig2_zero_corr_not_indep.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Weight instability: the multicollinearity catastrophe
# FIX 2: uses a LOCAL near-identical pair (corr > 0.9999), not the global
# dataset — so global figures are unaffected but the instability is dramatic.
# With corr > 0.9999, even noise_std=5 on a feature with std≈480 causes
# weights to flip sign and swing by >1000 units.
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
style_fig(fig, axes)

# LOCAL near-identical pair — separate from global dataset
np.random.seed(42)
size_demo  = np.random.normal(1500, 400, n)
plot_demo  = size_demo * 1.2 + np.random.normal(0, 1, n)  # corr ≈ 0.999998
price_demo = size_demo * 80  + np.random.normal(0, 15000, n)
X_demo     = np.column_stack([size_demo, plot_demo])

r_demo = np.corrcoef(size_demo, plot_demo)[0, 1]

# Run 200 trials with noise_std=5 (≈1% of feature std)
noise_std = 5
weight_history = []
for _ in range(200):
    Xn = X_demo.copy()
    Xn[:, 1] += np.random.normal(0, noise_std, n)
    weight_history.append(LinearRegression().fit(Xn, price_demo).coef_)
weights = np.array(weight_history)

# Left: scatter of weight pairs across trials — should be wildly spread
ax = axes[0]
ax.scatter(weights[:, 0], weights[:, 1],
           color=BLUE, alpha=0.4, s=18, linewidths=0)

# Mark the original (no-noise) weights
m_orig = LinearRegression().fit(X_demo, price_demo)
ax.scatter(m_orig.coef_[0], m_orig.coef_[1],
           color=YELLOW, s=90, zorder=5, label="original (no noise)",
           edgecolors=BG, linewidths=0.8)

ax.axhline(0, color=GREY, lw=0.5, ls="--")
ax.axvline(0, color=GREY, lw=0.5, ls="--")
ax.set_xlabel("w(size_demo)", fontfamily=FONT, fontsize=9)
ax.set_ylabel("w(plot_demo)", fontfamily=FONT, fontsize=9)
ax.set_title(f"200 regression runs  |  noise std = {noise_std}  |  feature std ≈ 480",
             fontfamily=FONT, fontsize=9, color=WHITE)
ax.text(0.05, 0.92,
        f"corr(size, plot) = {r_demo:.6f}",
        transform=ax.transAxes, fontfamily=FONT, fontsize=8, color=RED)
ax.text(0.05, 0.83,
        f"w(size) range: [{weights[:,0].min():.0f},  {weights[:,0].max():.0f}]\n"
        f"w(plot) range: [{weights[:,1].min():.0f}, {weights[:,1].max():.0f}]",
        transform=ax.transAxes, fontfamily=FONT, fontsize=8, color=WHITE)
leg = ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor=GREY, labelcolor=WHITE)

# Right: compare original vs one noisy run — sign flip
np.random.seed(7)
Xn_bad = X_demo.copy()
Xn_bad[:, 1] += np.random.normal(0, noise_std, n)
m_bad = LinearRegression().fit(Xn_bad, price_demo)

ax = axes[1]
labels = ["w(size_demo)", "w(plot_demo)"]
x_pos  = np.arange(2)
width  = 0.35

b1 = ax.bar(x_pos - width/2, m_orig.coef_, width,
            color=BLUE, alpha=0.85, label="original", edgecolor=BG)
b2 = ax.bar(x_pos + width/2, m_bad.coef_,  width,
            color=RED,  alpha=0.85, label=f"noise std={noise_std}", edgecolor=BG)

for bar, val in zip(b1, m_orig.coef_):
    offset = 12 if val >= 0 else -35
    ax.text(bar.get_x()+bar.get_width()/2, val+offset,
            f"{val:.1f}", ha="center", fontfamily=FONT, fontsize=8.5, color=WHITE)
for bar, val in zip(b2, m_bad.coef_):
    offset = 12 if val >= 0 else -35
    ax.text(bar.get_x()+bar.get_width()/2, val+offset,
            f"{val:.1f}", ha="center", fontfamily=FONT, fontsize=8.5, color=WHITE)

ax.axhline(0, color=WHITE, lw=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontfamily=FONT, fontsize=9)
ax.set_ylabel("learned weight", fontfamily=FONT, fontsize=9)
ax.set_title("original  vs  one noisy run\nsigns flip — model is meaningless",
             fontfamily=FONT, fontsize=9.5, color=WHITE)
leg = ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor=GREY, labelcolor=WHITE)

plt.suptitle("multicollinearity catastrophe  —  near-parallel features  →  (XᵀX)⁻¹ unstable\n"
             f"noise std = {noise_std}  on a feature with std ≈ 480  (< 2% perturbation)",
             fontfamily=FONT, fontsize=10.5, color=WHITE, y=1.04)
plt.tight_layout()
save("fig3_weight_instability.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Gram-Schmidt: the geometric fix
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
style_fig(fig, axes)

xc = df["size"].values     - df["size"].mean()
yc = df["bedrooms"].values - df["bedrooms"].mean()
u1       = xc / np.linalg.norm(xc)
proj     = (yc @ u1) * u1
residual = yc - proj
df["bedrooms_residual"] = residual

theta_bed = np.arccos(np.clip(np.dot(xc/np.linalg.norm(xc),
                                     yc/np.linalg.norm(yc)), -1, 1))
v_size = np.array([1.0, 0.0])
v_bed  = np.array([np.cos(theta_bed), np.sin(theta_bed)])
v_proj = np.dot(v_bed, v_size) * v_size
v_res  = v_bed - v_proj

ax = axes[0]
ax.annotate("", xy=v_size*1.05, xytext=(0,0),
            arrowprops=dict(arrowstyle="-|>", color=BLUE,   lw=2.2))
ax.annotate("", xy=v_bed*1.05,  xytext=(0,0),
            arrowprops=dict(arrowstyle="-|>", color=RED,    lw=2.2))
ax.annotate("", xy=v_proj*1.05, xytext=(0,0),
            arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=1.8,
                            linestyle="dashed"))
ax.annotate("", xy=v_bed*1.05, xytext=v_proj*1.05,
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))

foot  = v_proj
sq    = 0.045
perp  = np.array([0, 1])
sq_pts = [foot, foot+sq*v_size, foot+sq*v_size+sq*perp, foot+sq*perp]
ax.add_patch(plt.Polygon(sq_pts, closed=True, fill=False,
                         edgecolor=GREY, linewidth=0.9))
ax.plot([v_bed[0], v_proj[0]], [v_bed[1], v_proj[1]],
        color=GREY, lw=0.9, ls=":")

ax.text(v_size[0]*1.13, v_size[1]-0.05, "u₁  (size direction)",
        fontfamily=FONT, fontsize=8.5, color=BLUE)
ax.text(v_bed[0]*1.1,   v_bed[1]+0.05, "bedrooms_c",
        fontfamily=FONT, fontsize=8.5, color=RED)
ax.text(v_proj[0]*0.45, v_proj[1]-0.11,
        "projection\n(size effect in bedrooms)",
        fontfamily=FONT, fontsize=7.5, color=YELLOW, ha="center")
ax.text((v_proj[0]+v_bed[0])/2+0.07, (v_proj[1]+v_bed[1])/2+0.02,
        "residual = bedrooms_residual\n(⊥ to size, by construction)",
        fontfamily=FONT, fontsize=7.5, color=GREEN)

ax.set_xlim(-0.15, 1.45); ax.set_ylim(-0.25, 1.05)
ax.axhline(0, color=GREY, lw=0.5, ls="--")
ax.axvline(0, color=GREY, lw=0.5, ls="--")
ax.set_aspect("equal")
ax.tick_params(labelbottom=False, labelleft=False, length=0)
ax.set_title("Gram-Schmidt: project out the size direction from bedrooms",
             fontfamily=FONT, fontsize=9.5, color=WHITE)

ax = axes[1]
ax.scatter(xc[:120], yc[:120],
           color=RED, alpha=0.35, s=14, linewidths=0,
           label="bedrooms_c  (original)")
ax.scatter(xc[:120], residual[:120],
           color=GREEN, alpha=0.5, s=14, linewidths=0,
           label="bedrooms_residual")
ax.axhline(0, color=GREY, lw=0.5, ls="--")
ax.axvline(0, color=GREY, lw=0.5, ls="--")
ax.set_xlabel("size  (centered)", fontfamily=FONT, fontsize=9)
ax.set_ylabel("feature value  (centered)", fontfamily=FONT, fontsize=9)
ax.set_title("before  vs  after orthogonalization",
             fontfamily=FONT, fontsize=9.5, color=WHITE)

r_orig  = df["size"].corr(df["bedrooms"])
r_resid = np.corrcoef(xc, residual)[0, 1]
ax.text(0.05, 0.92,
        f"corr(size, bedrooms)          = {r_orig:.6f}",
        transform=ax.transAxes, fontfamily=FONT, fontsize=8, color=RED)
ax.text(0.05, 0.83,
        f"corr(size, bedrooms_residual) = {r_resid:.6f}",
        transform=ax.transAxes, fontfamily=FONT, fontsize=8, color=GREEN)
leg = ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor=GREY,
                labelcolor=WHITE, loc="lower right")

plt.suptitle("Gram-Schmidt orthogonalization  —  the geometric fix for feature overlap",
             fontfamily=FONT, fontsize=11.5, color=WHITE, y=1.02)
plt.tight_layout()
save("fig4_gram_schmidt.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Eigenvalues of the correlation matrix
# Near-zero last eigenvalue highlighted red — the multicollinearity signal.
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
style_fig(fig, axes)

X_s   = StandardScaler().fit_transform(features)
C     = np.corrcoef(X_s.T)
evals, evecs = np.linalg.eigh(C)
idx   = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]
nf2   = len(evals)
lbls  = [f"PC{i+1}" for i in range(nf2)]
expv  = evals / evals.sum()
cumv  = np.cumsum(expv)

ax = axes[0]
colors_bar = [BLUE if i < nf2-1 else RED for i in range(nf2)]
bars = ax.bar(lbls, evals, color=colors_bar, edgecolor=BG, lw=0.8)
for bar, val in zip(bars, evals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04,
            f"{val:.3f}", ha="center", va="bottom",
            fontfamily=FONT, fontsize=8.5, color=WHITE)
ax.axhline(1.0, color=YELLOW, lw=1.0, ls="--", alpha=0.7)
ax.text(nf2-0.55, 1.06, "Kaiser criterion (λ=1)",
        fontfamily=FONT, fontsize=7.5, color=YELLOW, ha="right")
ax.set_ylabel("eigenvalue  λ", fontfamily=FONT, fontsize=9)
ax.set_title("eigenvalues of C  (sum = number of features = 6)",
             fontfamily=FONT, fontsize=9.5, color=WHITE)
ax.annotate("near-zero eigenvalue\n→ (XᵀX) nearly singular\n→ multicollinearity signal",
            xy=(nf2-1, evals[-1]), xytext=(nf2-2.5, 0.5),
            fontfamily=FONT, fontsize=7.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

ax = axes[1]
ax.bar(lbls, expv*100, color=colors_bar, edgecolor=BG,
       lw=0.8, alpha=0.75, label="individual")
ax.plot(lbls, cumv*100, color=YELLOW, lw=2.0,
        marker="o", ms=5, markerfacecolor=YELLOW, label="cumulative")
for i, (pct, cum) in enumerate(zip(expv, cumv)):
    ax.text(i, pct*100+1.5, f"{pct*100:.1f}%",
            ha="center", fontfamily=FONT, fontsize=7.5, color=WHITE)
ax.axhline(90, color=GREEN, lw=0.9, ls="--", alpha=0.8)
ax.text(nf2-0.55, 91.5, "90% threshold",
        fontfamily=FONT, fontsize=7.5, color=GREEN, ha="right")
ax.set_ylabel("variance explained (%)", fontfamily=FONT, fontsize=9)
ax.set_title("3 components capture >90% of the variance",
             fontfamily=FONT, fontsize=9.5, color=WHITE)
leg = ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor=GREY,
                labelcolor=WHITE, loc="center right")

plt.suptitle("eigen decomposition of df.corr()  —  C = QΛQᵀ",
             fontfamily=FONT, fontsize=11.5, color=WHITE, y=1.02)
plt.tight_layout()
save("fig5_eigenvalues.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 6 — SVD vs eigen decomposition: two paths, one result
# FIX 1: uses S**2/n (not n-1) — StandardScaler divides by n (ddof=0),
# so XᵀX/n = C exactly, meaning σᵢ²/n = λᵢ.
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
style_fig(fig, axes)

U, S, Vt = np.linalg.svd(X_s, full_matrices=False)

# FIX 1: divide by n, not n-1
evals_from_svd   = S**2 / n
evals_from_eigen = evals

ax = axes[0]
x_pos = np.arange(nf2)
width = 0.38
b1 = ax.bar(x_pos - width/2, evals_from_eigen, width,
            color=BLUE,  alpha=0.85, label="λ  from eigh(C)",     edgecolor=BG)
b2 = ax.bar(x_pos + width/2, evals_from_svd,   width,
            color=GREEN, alpha=0.85, label="σ²/n  from SVD(X_s)", edgecolor=BG)

for bar, val in zip(b1, evals_from_eigen):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
            f"{val:.3f}", ha="center", fontfamily=FONT, fontsize=7.5, color=WHITE)
for bar, val in zip(b2, evals_from_svd):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
            f"{val:.3f}", ha="center", fontfamily=FONT, fontsize=7.5, color=GREEN)

ax.set_xticks(x_pos); ax.set_xticklabels(lbls, fontfamily=FONT, fontsize=9)
ax.set_ylabel("eigenvalue  λ", fontfamily=FONT, fontsize=9)
ax.set_title("two computational paths  →  identical eigenvalues",
             fontfamily=FONT, fontsize=9.5, color=WHITE)
leg = ax.legend(fontsize=8.5, facecolor="#1a1a1a", edgecolor=GREY,
                labelcolor=WHITE)

ax = axes[1]
ax.scatter(evals_from_eigen, evals_from_svd,
           color=YELLOW, s=80, zorder=5,
           edgecolors=BG, linewidths=0.5)
diag = np.linspace(0, max(evals_from_eigen)*1.05, 100)
ax.plot(diag, diag, color=GREY, lw=1.2, ls="--", label="y = x  (perfect agreement)")
for i, (xe, xs) in enumerate(zip(evals_from_eigen, evals_from_svd)):
    ax.text(xe+0.03, xs+0.03, f"PC{i+1}",
            fontfamily=FONT, fontsize=8, color=WHITE)
ax.set_xlabel("eigenvalue from  eigh(C)", fontfamily=FONT, fontsize=9)
ax.set_ylabel("σ²/n  from  SVD(X_scaled)", fontfamily=FONT, fontsize=9)
ax.set_title("XᵀX/n = C  →  σᵢ²/n = λᵢ  (exact, not approximate)",
             fontfamily=FONT, fontsize=9, color=WHITE)
ax.set_aspect("equal")

max_diff = np.max(np.abs(evals_from_eigen - evals_from_svd))
ax.text(0.05, 0.88,
        f"max |difference| = {max_diff:.2e}\n(floating-point precision only)",
        transform=ax.transAxes, fontfamily=FONT, fontsize=8, color=GREEN)
ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor=GREY, labelcolor=WHITE)

plt.suptitle("SVD  and  eigen decomposition  are  the same object\n"
             "StandardScaler uses ddof=0  →  XᵀX/n = C  →  σᵢ²/n = λᵢ",
             fontfamily=FONT, fontsize=10.5, color=WHITE, y=1.04)
plt.tight_layout()
save("fig6_svd_vs_eigen.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 7 — Conceptual chain: the full article arc
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 4.0))
style_fig(fig, [ax])
ax.set_xlim(0, 14); ax.set_ylim(0, 4.0)
ax.axis("off")

nodes = [
    (0.85,  "df.corr()",     BLUE,   "Gram matrix\nof cosines"),
    (2.55,  "dot product",   BLUE,   "corr = cos(θ)\nbetween vectors"),
    (4.25,  "orthogonality", GREEN,  "θ = 90°\nzero linear overlap"),
    (5.95,  "Gram-Schmidt",  GREEN,  "project & subtract\nremove shared direction"),
    (7.65,  "QR",            YELLOW, "A = QR\nRw* = Qᵀy  (back-sub)"),
    (9.35,  "eigen decomp",  YELLOW, "C = QΛQᵀ\ndirections of variance"),
    (11.05, "SVD",           RED,    "X = UΣVᵀ\nσᵢ²/n = λᵢ"),
    (12.75, "PCA",           PURPLE, "globally optimal\northogonal directions"),
]

box_w, box_h = 1.3, 0.75
y_box = 2.55

for x, label, color, sub in nodes:
    rect = mpatches.FancyBboxPatch(
        (x-box_w/2, y_box-box_h/2), box_w, box_h,
        boxstyle="round,pad=0.07", linewidth=1.4,
        edgecolor=color, facecolor=BG
    )
    ax.add_patch(rect)
    ax.text(x, y_box+0.04, label, ha="center", va="center",
            fontfamily=FONT, fontsize=8.8, color=color, fontweight="bold")
    ax.text(x, y_box-0.72, sub, ha="center", va="top",
            fontfamily=FONT, fontsize=6.8, color=DIM,
            multialignment="center", linespacing=1.45)

for i in range(len(nodes)-1):
    x1 = nodes[i][0]   + box_w/2
    x2 = nodes[i+1][0] - box_w/2
    ax.annotate("", xy=(x2, y_box), xytext=(x1, y_box),
                arrowprops=dict(arrowstyle="-|>", color=GREY,
                                lw=1.3, mutation_scale=10))

ax.annotate("stable back-sub\nbehind model.fit()",
            xy=(7.65, y_box-box_h/2), xytext=(7.65, 0.7),
            fontfamily=FONT, fontsize=7.5, color=YELLOW, ha="center",
            arrowprops=dict(arrowstyle="->", color=YELLOW, lw=0.9))
ax.annotate("multicollinearity\ndiagnosis",
            xy=(9.35, y_box-box_h/2), xytext=(9.8, 0.7),
            fontfamily=FONT, fontsize=7.5, color=YELLOW, ha="center",
            arrowprops=dict(arrowstyle="->", color=YELLOW, lw=0.9))

ax.text(7.0, 3.78,
        "one connected idea  —  seen from eight angles",
        ha="center", fontfamily=FONT, fontsize=11.5, color=WHITE)

plt.tight_layout()
save("fig7_conceptual_chain.png")

print("\nAll 8 figures saved to Visuals/")
print("Changes from v2:")
print("  fig3: local near-identical pair (corr>0.9999) — catastrophic weight swings")
print("  fig6: σ²/n (not n-1) — exact match with StandardScaler (ddof=0)")
print("  fig7: QR node updated to 'Rw* = Qᵀy (back-sub)'")