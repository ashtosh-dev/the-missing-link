import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# -------------------------------------------------
# The Missing Link visual style
# -------------------------------------------------
BG = "#0B1020"
PANEL = "#111827"
GRID = "#263244"
WHITE = "#E5E7EB"
MUTED = "#94A3B8"
CYAN = "#38BDF8"
BLUE = "#60A5FA"
ORANGE = "#F59E0B"
GREEN = "#34D399"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "text.color": WHITE,
    "axes.labelcolor": WHITE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titleweight": "bold",
})

def style_ax(ax, title):
    ax.set_title(title, color=WHITE, pad=12, fontsize=13)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color(GRID)

def arrow(ax, start, end, color, label=None, lw=2.5):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=14
        )
    )
    if label:
        ax.text(
            end[0] + 0.08,
            end[1] + 0.08,
            label,
            color=color,
            fontsize=10,
            weight="bold"
        )

def save_fig(fig, filename):
    fig.savefig(
        filename,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.18
    )
    plt.close(fig)
    print(f"Saved: {filename}")


# =================================================
# FIGURE 1: Orthogonal transformation preserves shape
# =================================================
theta = np.deg2rad(40)

Q = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

square = np.array([
    [-1, -1],
    [ 1, -1],
    [ 1,  1],
    [-1,  1]
])

rotated = square @ Q.T

# Increased height and added subplots_adjust to prevent title overlaps
fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
fig.subplots_adjust(top=0.76, bottom=0.18)

for ax in axes:
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

style_ax(axes[0], "Original basis")
style_ax(axes[1], "After orthogonal transformation")

axes[0].add_patch(
    Polygon(square, closed=True, fill=False,
            edgecolor=CYAN, linewidth=2.5)
)
axes[1].add_patch(
    Polygon(rotated, closed=True, fill=False,
            edgecolor=ORANGE, linewidth=2.5)
)

# Original perpendicular unit vectors
arrow(axes[0], [0, 0], [1, 0], CYAN, "e₁")
arrow(axes[0], [0, 0], [0, 1], BLUE, "e₂")

# Rotated vectors
e1_rot = Q @ np.array([1, 0])
e2_rot = Q @ np.array([0, 1])

arrow(axes[1], [0, 0], e1_rot, ORANGE, "Qe₁")
arrow(axes[1], [0, 0], e2_rot, GREEN, "Qe₂")

fig.suptitle(
    "Orthogonal transformations preserve lengths and angles",
    fontsize=15, color=WHITE, weight="bold", y=0.92
)
fig.text(
    0.5, 0.04,
    "Rotation changes direction, not geometry",
    ha="center", color=MUTED, fontsize=10
)

save_fig(fig, "figure1_orthogonal_transformation.png")


# =================================================
# FIGURE 2: Eigenvectors of a symmetric matrix
# =================================================
A = np.array([
    [3.0, 1.0],
    [1.0, 3.0]
])

eigenvalues, eigenvectors = np.linalg.eigh(A)

# Unit circle and its image under A
t = np.linspace(0, 2*np.pi, 400)
circle = np.vstack([np.cos(t), np.sin(t)])
ellipse = A @ circle

fig, ax = plt.subplots(figsize=(6.8, 6.2))
fig.subplots_adjust(top=0.88, bottom=0.12)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
style_ax(ax, "A symmetric transformation")

ax.plot(circle[0], circle[1],
        color=CYAN, linewidth=2,
        label="Unit circle")

ax.plot(ellipse[0], ellipse[1],
        color=ORANGE, linewidth=2.5,
        label="Transformed circle")

# Eigenvectors and their scaled images
for i, color in enumerate([GREEN, BLUE]):
    v = eigenvectors[:, i]
    Av = A @ v
    lam = eigenvalues[i]

    arrow(ax, [0, 0], v, color, f"v{i+1}")
    arrow(ax, [0, 0], Av, color, f"λ{i+1}v{i+1}", lw=2)

ax.legend(
    facecolor=PANEL,
    edgecolor=GRID,
    labelcolor=WHITE,
    loc="lower right"
)

fig.text(
    0.5, 0.02,
    "Av = λv: eigenvectors keep their direction, up to sign",
    ha="center", color=MUTED, fontsize=10
)

save_fig(fig, "figure2_symmetric_eigenvectors.png")


# =================================================
# FIGURE 3: PCA directions in correlated data
# =================================================
rng = np.random.default_rng(7)

# Create correlated 2D data
mean = np.array([0.0, 0.0])
cov = np.array([
    [3.0, 2.4],
    [2.4, 2.5]
])

X = rng.multivariate_normal(mean, cov, size=350)
X_centered = X - X.mean(axis=0)

# Covariance eigenvectors = principal directions
C = np.cov(X_centered, rowvar=False)
vals, vecs = np.linalg.eigh(C)

# Sort largest variance first
order = np.argsort(vals)[::-1]
vals = vals[order]
vecs = vecs[:, order]

pc1 = vecs[:, 0]
pc2 = vecs[:, 1]

# Projection onto first principal component
scores = X_centered @ pc1
X_proj = np.outer(scores, pc1)

fig, ax = plt.subplots(figsize=(7, 6.2))
fig.subplots_adjust(top=0.88, bottom=0.12)
style_ax(ax, "PCA finds orthogonal directions of variation")

ax.scatter(
    X_centered[:, 0],
    X_centered[:, 1],
    s=18, alpha=0.55,
    color=CYAN, edgecolors="none",
    label="Centered data"
)

# Show selected projections
for i in range(0, len(X_centered), 12):
    ax.plot(
        [X_centered[i, 0], X_proj[i, 0]],
        [X_centered[i, 1], X_proj[i, 1]],
        color=MUTED, alpha=0.28, linewidth=0.8
    )

scale1 = 3 * np.sqrt(vals[0])
scale2 = 3 * np.sqrt(vals[1])

arrow(ax, [0, 0], pc1 * scale1, ORANGE, "PC1")
arrow(ax, [0, 0], pc2 * scale2, GREEN, "PC2")

ax.legend(
    facecolor=PANEL,
    edgecolor=GRID,
    labelcolor=WHITE,
    loc="upper left"
)

fig.text(
    0.5, 0.02,
    "PC1 captures the greatest variance; PC2 is orthogonal to PC1",
    ha="center", color=MUTED, fontsize=10
)

save_fig(fig, "figure3_pca_directions.png")