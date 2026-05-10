import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
 
# The transformation from the article: F(x,y) = (2x+3y, 4x-5y)
# Matrix representation relative to standard basis E:
#   [F]_E = | 2   3 |
#           | 4  -5 |
T = np.array([[2,  3],
              [4, -5]])
 
 
# ─────────────────────────────────────────────────────────────────────────────
# fig0 — A single vector before and after transformation
# Shows: input v = (3,2) and output F(v) = (12, 2)
# Placed in article: after the code block that verifies the matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig0_vector_mapping():
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-4, 6)
    ax.axhline(0, color='#444', lw=0.8)
    ax.axvline(0, color='#444', lw=0.8)
    ax.grid(True, ls='--', alpha=0.4)
 
    v  = np.array([3, 2])
    Tv = T @ v          # [12, 2]
 
    ax.annotate('', xy=v, xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='#57c1ff', lw=2.5))
    ax.text(v[0] + 0.2, v[1] + 0.2,
            'v = (3, 2)', color='#57c1ff', fontsize=11)
 
    ax.annotate('', xy=Tv, xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=2.5))
    ax.text(Tv[0] + 0.2, Tv[1] + 0.2,
            'F(v) = (12, 2)', color='#ff6b6b', fontsize=11)
 
    ax.plot(0, 0, 'o', color='white', ms=5, zorder=5)
 
    handles = [
        mpatches.Patch(color='#57c1ff', label='Input vector v'),
        mpatches.Patch(color='#ff6b6b', label='Output F(v)'),
    ]
    ax.legend(handles=handles, facecolor='#1a1a1a',
              edgecolor='#444', labelcolor='#cccccc', fontsize=9)
 
    ax.set_title('A vector before and after transformation F',
                 color='#aaaaaa', pad=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
 
    plt.tight_layout()
    plt.savefig('Visuals/fig0_vector_mapping.png', dpi=150)
    plt.close()
    print("fig0 saved → Visuals/fig0_vector_mapping.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# fig1 — Grid before and after transformation (side by side)
# Shows: how F stretches and shears the plane
# Placed in article: after the "What the grid actually looks like" section
# ─────────────────────────────────────────────────────────────────────────────
def fig1_grid_transformation():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
 
    configs = [
        ('Input space',
         False, (-3.5, 3.5), (-3.5, 3.5)),
        ('Output space  —  F(x,y) = (2x+3y,  4x−5y)',
         True,  (-14, 14),   (-18, 18)),
    ]
 
    grid_vals = np.linspace(-3, 3, 7)
    fine      = np.linspace(-3, 3, 120)
 
    for ax, (title, do_transform, xlim, ylim) in zip(axes, configs):
        ax.set_facecolor('#0f0f0f')
        ax.set_title(title, color='#aaaaaa', pad=8, fontsize=9)
        ax.axhline(0, color='#555', lw=0.8)
        ax.axvline(0, color='#555', lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel('x', color='#888')
        ax.set_ylabel('y', color='#888')
 
        for val in grid_vals:
            # horizontal grid lines  (y = val)
            pts = np.column_stack([fine, np.full_like(fine, val)])
            if do_transform:
                pts = (T @ pts.T).T
            ax.plot(pts[:, 0], pts[:, 1], color='#1f5f8b', lw=0.9, alpha=0.8)
 
            # vertical grid lines  (x = val)
            pts = np.column_stack([np.full_like(fine, val), fine])
            if do_transform:
                pts = (T @ pts.T).T
            ax.plot(pts[:, 0], pts[:, 1], color='#1f5f8b', lw=0.9, alpha=0.8)
 
        # basis vectors
        e1 = T @ np.array([1, 0]) if do_transform else np.array([1, 0])
        e2 = T @ np.array([0, 1]) if do_transform else np.array([0, 1])
 
        ax.annotate('', xy=e1, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#57c1ff', lw=2.2))
        ax.annotate('', xy=e2, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=2.2))
 
        offset = 0.6 if do_transform else 0.12
        lbl1 = (f'e₁ → ({int(e1[0])}, {int(e1[1])})'
                if do_transform else 'e₁ = (1, 0)')
        lbl2 = (f'e₂ → ({int(e2[0])}, {int(e2[1])})'
                if do_transform else 'e₂ = (0, 1)')
        ax.text(e1[0] + offset, e1[1] + offset, lbl1,
                color='#57c1ff', fontsize=9)
        ax.text(e2[0] + offset, e2[1] + offset, lbl2,
                color='#ff6b6b', fontsize=9)
 
    fig.suptitle('How F stretches and shears the plane',
                 color='#888888', y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig('Visuals/fig1_grid_transformation.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("fig1 saved → Visuals/fig1_grid_transformation.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# fig2 — [T]_S construction table
# Shows: the four-step procedure visualised as a table with resulting matrix
# Placed in article: right after the four-step procedure explanation
# ─────────────────────────────────────────────────────────────────────────────
def fig2_matrix_construction():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor('#0f0f0f')
    ax.axis('off')
 
    headers = ['Step', 'Basis vector',
               'Apply F', 'Express in basis E', 'Column of [F]_E']
    rows = [
        ['1', 'e₁ = (1, 0)', 'F(1,0) = (2, 4)',
         '2·e₁ + 4·e₂',     '[2,  4]ᵀ'],
        ['2', 'e₂ = (0, 1)', 'F(0,1) = (3, −5)',
         '3·e₁ + (−5)·e₂',  '[3, −5]ᵀ'],
    ]
 
    col_colors = ['#888888', '#57c1ff', '#ffd966', '#b5e6a2', '#ff6b6b']
    col_x      = [0.02, 0.10, 0.28, 0.54, 0.80]
    row_y      = [0.82, 0.55, 0.30]
 
    # header row
    for hdr, cx, cc in zip(headers, col_x, col_colors):
        ax.text(cx, row_y[0], hdr,
                color=cc, fontsize=9, fontweight='bold',
                transform=ax.transAxes, va='top')
 
    ax.axhline(0.68, color='#444', lw=0.8, xmin=0.01, xmax=0.99)
 
    # data rows
    for ri, row in enumerate(rows):
        for val, cx, cc in zip(row, col_x, col_colors):
            ax.text(cx, row_y[ri + 1], val,
                    color=cc, fontsize=10,
                    transform=ax.transAxes, va='top',
                    fontfamily='monospace')
 
    # resulting matrix
    ax.text(0.5, 0.04,
            '[F]_E  =  | 2    3 |\n           | 4   −5 |',
            ha='center', va='bottom', fontsize=11,
            color='#ffffff', transform=ax.transAxes,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                      edgecolor='#57c1ff', lw=1.5))
 
    ax.set_title('Building [F]_E — column by column',
                 color='#aaaaaa', pad=10)
    plt.tight_layout()
    plt.savefig('Visuals/fig2_T_S_construction.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("fig2 saved → Visuals/fig2_T_S_construction.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig0_vector_mapping()
    fig1_grid_transformation()
    fig2_matrix_construction()
    print("\nDone. All figures in Visuals/")