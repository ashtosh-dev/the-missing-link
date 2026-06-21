"""
The Missing Link — Article 3B
Same Data. Different Coordinates. This Is What PCA Actually Does.
Visualization script — generates fig0 through fig6

Run:
    python change_of_basis_visualization.py

Output:
    Visuals/fig0_same_point_two_bases.png
    Visuals/fig1_standard_scaler.png
    Visuals/fig2_pca_axes.png
    Visuals/fig3_P_construction.png
    Visuals/fig4_similarity.png
    Visuals/fig5_iris_pca.png
    Visuals/fig6_dct.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from scipy.fft import dctn, idctn
import os

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor':   '#0f0f0f',
    'axes.edgecolor':   '#444444',
    'axes.labelcolor':  '#cccccc',
    'xtick.color':      '#888888',
    'ytick.color':      '#888888',
    'text.color':       '#cccccc',
    'grid.color':       '#2a2a2a',
    'grid.linewidth':   0.6,
    'font.family':      'monospace',
})

os.makedirs('Visuals', exist_ok=True)

# ── shared data ───────────────────────────────────────────────────────────────
np.random.seed(42)
mean = [2, 3]
cov  = [[3, 2], [2, 2]]
X    = np.random.multivariate_normal(mean, cov, 200)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# ── fig0: same point, two coordinate systems ──────────────────────────────────
# Placed in article: after "The Problem With Coordinates" section
def fig0_same_point_two_bases():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    point = np.array([3, 1])

    configs = [
        ('Standard basis E',   np.eye(2),                    '(3, 1) in E'),
        ('Custom basis S',     np.array([[1, 0.5],[0, 1]]),  '(2.5, 1) in S'),
    ]

    for ax, (title, basis, label) in zip(axes, configs):
        ax.set_facecolor('#0f0f0f')
        ax.axhline(0, color='#444', lw=0.8)
        ax.axvline(0, color='#444', lw=0.8)
        ax.set_xlim(-1, 5); ax.set_ylim(-1, 4)
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_title(title, color='#aaaaaa', pad=8)

        b1 = basis[:, 0] if basis.shape == (2,2) else basis[0]
        b2 = basis[:, 1] if basis.shape == (2,2) else basis[1]

        ax.annotate('', xy=b1, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#57c1ff', lw=2))
        ax.annotate('', xy=b2, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=2))

        ax.plot(*point, 'o', color='#ffd966', ms=8, zorder=5)
        ax.text(point[0]+0.1, point[1]+0.1, label,
                color='#ffd966', fontsize=10)

    fig.suptitle('Same point. Different coordinate systems. Different labels.',
                 color='#888', fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig('Visuals/fig0_same_point_two_bases.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("fig0 saved → Visuals/fig0_same_point_two_bases.png")


# ── fig1: StandardScaler before/after ────────────────────────────────────────
# Placed in article: after the StandardScaler code block
def fig1_standard_scaler():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in zip(axes,
        [X, X_scaled],
        ['Original data', 'After StandardScaler']):
        ax.set_facecolor('#0f0f0f')
        ax.scatter(data[:, 0], data[:, 1],
                   alpha=0.4, s=15, color='#57c1ff')
        ax.axhline(0, color='#555', lw=0.8)
        ax.axvline(0, color='#555', lw=0.8)
        ax.set_title(title, color='#aaaaaa', pad=8)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    fig.suptitle('StandardScaler: same shape, different axis labels',
                 color='#888', fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig('Visuals/fig1_standard_scaler.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("fig1 saved → Visuals/fig1_standard_scaler.png")


# ── fig2: PCA axes overlaid on data ──────────────────────────────────────────
# Placed in article: after the PCA code block
def fig2_pca_axes():
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#0f0f0f')
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1],
               alpha=0.3, s=15, color='#888888')
    ax.axhline(0, color='#444', lw=0.8)
    ax.axvline(0, color='#444', lw=0.8)

    scale = 2.5
    for comp, color, label in zip(
        pca.components_,
        ['#57c1ff', '#ff6b6b'],
        ['PC1 (90.8% variance)', 'PC2 (9.2% variance)']
    ):
        ax.annotate('', xy=scale * comp, xytext=-scale * comp,
            arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.text(*(scale * comp + 0.1), label, color=color, fontsize=9)

    ax.set_title('PCA finds new basis vectors aligned with data variance',
                 color='#aaaaaa', pad=8)
    ax.set_xlabel('Feature 1 (scaled)')
    ax.set_ylabel('Feature 2 (scaled)')
    plt.tight_layout()
    plt.savefig('Visuals/fig2_pca_axes.png', dpi=150)
    plt.close()
    print("fig2 saved → Visuals/fig2_pca_axes.png")


# ── fig3: P matrix construction table ────────────────────────────────────────
# Placed in article: after the change of basis worked example
def fig3_P_construction():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_facecolor('#0f0f0f')
    ax.axis('off')

    headers = ['Step', 'New basis vector', 'Express in S',
               'Coefficients', 'Column of P']
    rows = [
        ['1', 'u₁ = (1, −1)', 'x(1,2) + y(3,5) = (1,−1)', 'x=−8, y=3',  '[−8,  3]ᵀ'],
        ['2', 'u₂ = (1, −2)', 'x(1,2) + y(3,5) = (1,−2)', 'x=−11, y=4', '[−11, 4]ᵀ'],
    ]
    col_colors = ['#888888', '#57c1ff', '#ffd966', '#b5e6a2', '#ff6b6b']
    col_x      = [0.02, 0.10, 0.27, 0.60, 0.80]
    row_y      = [0.82, 0.55, 0.28]

    for hdr, cx, cc in zip(headers, col_x, col_colors):
        ax.text(cx, row_y[0], hdr,
                color=cc, fontsize=9, fontweight='bold',
                transform=ax.transAxes, va='top')

    ax.axhline(0.68, color='#444', lw=0.8, xmin=0.01, xmax=0.99)

    for ri, row in enumerate(rows):
        for val, cx, cc in zip(row, col_x, col_colors):
            ax.text(cx, row_y[ri + 1], val,
                    color=cc, fontsize=9,
                    transform=ax.transAxes, va='top',
                    fontfamily='monospace')

    ax.text(0.5, 0.02,
            'P = | −8  −11 |\n    |  3    4 |',
            ha='center', va='bottom', fontsize=11,
            color='#ffffff', transform=ax.transAxes,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                      edgecolor='#57c1ff', lw=1.5))

    ax.set_title('Building the change of basis matrix P — column by column',
                 color='#aaaaaa', pad=10)
    plt.tight_layout()
    plt.savefig('Visuals/fig3_P_construction.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("fig3 saved → Visuals/fig3_P_construction.png")


# ── fig4: B = P⁻¹AP similarity diagram ───────────────────────────────────────
# Placed in article: after the similarity code block
def fig4_similarity():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_facecolor('#0f0f0f')
    ax.axis('off')

    boxes = [
        (0.12, 0.5, '[F]_E\n| 2   3 |\n| 4  -5 |',   '#57c1ff', 'Standard basis E'),
        (0.88, 0.5, '[F]_S\n|  52  129 |\n| -22  -55 |', '#ff6b6b', 'Basis S'),
    ]
    for x, y, text, color, label in boxes:
        ax.text(x, y, text,
                ha='center', va='center', fontsize=10,
                color=color, fontfamily='monospace',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#1a1a2e',
                          edgecolor=color, lw=1.5))
        ax.text(x, y - 0.32, label,
                ha='center', color='#888', fontsize=8,
                transform=ax.transAxes)

    ax.annotate('', xy=(0.72, 0.62), xytext=(0.28, 0.62),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='#ffd966', lw=2))
    ax.text(0.5, 0.68, 'B = P⁻¹AP',
            ha='center', color='#ffd966', fontsize=10,
            transform=ax.transAxes, fontfamily='monospace')

    ax.annotate('', xy=(0.28, 0.38), xytext=(0.72, 0.38),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='#b5e6a2', lw=2))
    ax.text(0.5, 0.27, 'A = PBP⁻¹',
            ha='center', color='#b5e6a2', fontsize=10,
            transform=ax.transAxes, fontfamily='monospace')

    ax.text(0.5, 0.08,
            'tr(A) = tr(B) = −3     det(A) = det(B) = −22',
            ha='center', color='#aaaaaa', fontsize=9,
            transform=ax.transAxes, fontfamily='monospace')

    ax.set_title('Similarity: same transformation, different coordinate system',
                 color='#aaaaaa', pad=10)
    plt.tight_layout()
    plt.savefig('Visuals/fig4_similarity.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("fig4 saved → Visuals/fig4_similarity.png")


# ── fig5: Iris PCA scatter ────────────────────────────────────────────────────
# Placed in article: after the Iris code block
def fig5_iris_pca():
    iris         = load_iris()
    X_iris_sc    = StandardScaler().fit_transform(iris.data)
    pca_iris     = PCA(n_components=2)
    X_iris_pca   = pca_iris.fit_transform(X_iris_sc)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#0f0f0f')

    colors = ['#57c1ff', '#ff6b6b', '#ffd966']
    for i, (species, color) in enumerate(zip(iris.target_names, colors)):
        mask = iris.target == i
        ax.scatter(X_iris_pca[mask, 0], X_iris_pca[mask, 1],
                   alpha=0.7, s=40, color=color, label=species)

    ax.axhline(0, color='#444', lw=0.8)
    ax.axvline(0, color='#444', lw=0.8)
    ax.set_xlabel('PC1 (72.96% variance)', color='#888')
    ax.set_ylabel('PC2 (22.85% variance)', color='#888')
    ax.set_title(
        'Iris dataset in PCA coordinates\n'
        '4 features → 2 components (95.8% variance retained)',
        color='#aaaaaa', pad=8)
    ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='#cccccc')
    plt.tight_layout()
    plt.savefig('Visuals/fig5_iris_pca.png', dpi=150)
    plt.close()
    print("fig5 saved → Visuals/fig5_iris_pca.png")


# ── fig6: DCT compression ─────────────────────────────────────────────────────
# Placed in article: after the DCT code block
def fig6_dct():
    np.random.seed(0)
    patch       = np.random.randint(0, 256, (8, 8)).astype(float)
    dct_coeffs  = dctn(patch, norm='ortho')
    compressed  = np.zeros((8, 8))
    compressed[:4, :4] = dct_coeffs[:4, :4]
    reconstructed = idctn(compressed, norm='ortho')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles    = [
        'Original patch\n(pixel basis)',
        'DCT coefficients\n(frequency basis)',
        'Reconstructed\n(25% coefficients kept)',
    ]
    data_list = [patch, dct_coeffs, reconstructed]
    cmaps     = ['gray', 'RdBu_r', 'gray']

    for ax, data, title, cmap in zip(axes, data_list, titles, cmaps):
        ax.set_facecolor('#0f0f0f')
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        ax.set_title(title, color='#aaaaaa', pad=6, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Same image patch — pixel basis vs frequency basis (DCT)',
                 color='#888', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig('Visuals/fig6_dct.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("fig6 saved → Visuals/fig6_dct.png")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig0_same_point_two_bases()
    fig1_standard_scaler()
    fig2_pca_axes()
    fig3_P_construction()
    fig4_similarity()
    fig5_iris_pca()
    fig6_dct()
    print("\nDone. All figures in Visuals/")