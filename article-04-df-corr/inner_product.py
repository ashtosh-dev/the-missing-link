"""
The Missing Link — Article 4A 
Inner Products, Norms, and Orthogonality
Narrative: Recognition across disguises — same instrument, different contexts

Figures:
  fig0 — The three disguises: same dot product in class, cosine similarity, attention
  fig1 — Weights formula as two explicit operations
  fig2 — Orthogonality: what dot product = 0 actually means (three cases)
  fig3 — Cosine similarity = normalized dot product (physics class -> ML bridge)
  fig4 — Attention mechanism: query.key is the weights formula
  fig5 — RAG pipeline: dot product as retrieval engine

Dark theme:
  BG #0f0f0f | BLUE #57c1ff | RED #ff6b6b | YELLOW #ffd966 | GREEN #b5e6a2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

BG     = "#0f0f0f"
BLUE   = "#57c1ff"
RED    = "#ff6b6b"
YELLOW = "#ffd966"
GREEN  = "#b5e6a2"
GREY   = "#555555"
WHITE  = "#e8e8e8"
DIMMED = "#888888"

plt.rcParams.update({
    "font.family":      "monospace",
    "text.color":       WHITE,
    "axes.facecolor":   BG,
    "figure.facecolor": BG,
    "axes.edgecolor":   GREY,
    "axes.labelcolor":  WHITE,
    "xtick.color":      DIMMED,
    "ytick.color":      DIMMED,
    "grid.color":       "#222222",
    "grid.linewidth":   0.5,
})

os.makedirs("Visuals", exist_ok=True)

def arrow(ax, origin, vec, color, label="", lw=2.5, offset=(0.06, 0.06)):
    ax.annotate("", xy=(origin[0]+vec[0], origin[1]+vec[1]), xytext=origin,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=18))
    if label:
        ax.text(origin[0]+vec[0]+offset[0], origin[1]+vec[1]+offset[1],
                label, color=color, fontsize=11, fontweight="bold")

def save(fig, name):
    path = f"Visuals/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {path}")
    plt.close(fig)


def fig0_three_disguises():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("One Operation. Three Different Names. Nobody Drew the Line.",
                 fontsize=13, color=WHITE, y=1.02)

    ax = axes[0]
    ax.set_facecolor(BG); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("In your linear algebra class", color=DIMMED, fontsize=10, pad=10)
    ax.text(5, 8.5, "Weights formula", color=WHITE, fontsize=11, ha="center", fontweight="bold")
    ax.text(5, 7.0, "cj = y.uj / uj.uj", color=YELLOW, fontsize=14, ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor=YELLOW))
    ax.text(5, 5.2, "Expand y in an\northogonal basis", color=DIMMED, fontsize=9, ha="center")
    u = np.array([1.0, 0.0]); y = np.array([0.7, 0.6])
    dot_val = np.dot(y, u); norm_sq = np.dot(u, u)
    ax.text(5, 3.5, f"y.uj = {dot_val:.1f}", color=BLUE, fontsize=10, ha="center")
    ax.text(5, 2.5, f"uj.uj = {norm_sq:.1f}", color=BLUE, fontsize=10, ha="center")
    ax.text(5, 1.5, f"cj = {dot_val/norm_sq:.1f}", color=GREEN, fontsize=10, ha="center", fontweight="bold")
    ax.add_patch(patches.FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.1",
                 edgecolor=YELLOW, facecolor="none", lw=1.5, alpha=0.4))

    ax2 = axes[1]
    ax2.set_facecolor(BG); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
    ax2.set_title("In your ML libraries", color=DIMMED, fontsize=10, pad=10)
    ax2.text(5, 8.5, "Cosine Similarity", color=WHITE, fontsize=11, ha="center", fontweight="bold")
    ax2.text(5, 7.0, "cos(u,v) = u.v / (||u|| ||v||)", color=BLUE, fontsize=11, ha="center",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor=BLUE))
    ax2.text(5, 5.2, "Rank documents in\na vector database", color=DIMMED, fontsize=9, ha="center")
    u2 = np.array([0.8, 0.6]); v2 = np.array([0.9, 0.4])
    dot2 = np.dot(u2, v2); sim = dot2 / (np.linalg.norm(u2) * np.linalg.norm(v2))
    ax2.text(5, 3.5, f"u.v = {dot2:.3f}", color=BLUE, fontsize=10, ha="center")
    ax2.text(5, 2.5, f"||u||.||v|| = {np.linalg.norm(u2)*np.linalg.norm(v2):.3f}", color=BLUE, fontsize=10, ha="center")
    ax2.text(5, 1.5, f"similarity = {sim:.3f}", color=GREEN, fontsize=10, ha="center", fontweight="bold")
    ax2.add_patch(patches.FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.1",
                  edgecolor=BLUE, facecolor="none", lw=1.5, alpha=0.4))

    ax3 = axes[2]
    ax3.set_facecolor(BG); ax3.set_xlim(0, 10); ax3.set_ylim(0, 10); ax3.axis("off")
    ax3.set_title("In transformer attention", color=DIMMED, fontsize=10, pad=10)
    ax3.text(5, 8.5, "Attention Score", color=WHITE, fontsize=11, ha="center", fontweight="bold")
    ax3.text(5, 7.0, "score = q.k / sqrt(d)", color=RED, fontsize=14, ha="center",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor=RED))
    ax3.text(5, 5.2, "How much should this\ntoken attend to that one?", color=DIMMED, fontsize=9, ha="center")
    q = np.array([0.6, 0.8]); k = np.array([0.7, 0.5]); d = 2
    score = np.dot(q, k) / np.sqrt(d)
    ax3.text(5, 3.5, f"q.k = {np.dot(q,k):.3f}", color=BLUE, fontsize=10, ha="center")
    ax3.text(5, 2.5, f"sqrt(d) = {np.sqrt(d):.3f}", color=BLUE, fontsize=10, ha="center")
    ax3.text(5, 1.5, f"score = {score:.3f}", color=GREEN, fontsize=10, ha="center", fontweight="bold")
    ax3.add_patch(patches.FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.1",
                  edgecolor=RED, facecolor="none", lw=1.5, alpha=0.4))

    fig.text(0.5, -0.04,
             "All three reduce to: multiply components, sum them up. The dot product, wearing different clothes.",
             ha="center", color=DIMMED, fontsize=10, style="italic")
    plt.tight_layout()
    save(fig, "fig0_three_disguises.png")

    print(f"[fig0] weights formula cj = {dot_val/norm_sq:.4f}")
    print(f"[fig0] cosine similarity  = {sim:.6f}")
    print(f"[fig0] attention score    = {score:.6f}")


def fig1_weights_two_operations():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("The Weights Formula: Two Operations, Not One", fontsize=13, color=WHITE, y=1.02)

    u = np.array([2.0, 0.0])
    y = np.array([1.6, 1.4])
    u_hat = u / np.linalg.norm(u)
    proj_scalar = np.dot(y, u_hat)
    c_j = np.dot(y, u) / np.dot(u, u)
    proj_vec = c_j * u

    ax = axes[0]
    ax.set_facecolor(BG); ax.set_xlim(-0.3, 2.8); ax.set_ylim(-0.4, 2.2)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5)
    ax.set_title("Step 1: Scalar projection\ny.uj / ||uj||", color=DIMMED, fontsize=10)
    arrow(ax, (0,0), u, BLUE, "uj", offset=(0.06, -0.2))
    arrow(ax, (0,0), y, YELLOW, "y", offset=(0.06, 0.06))
    ax.plot([proj_scalar, y[0]], [0, y[1]], color=RED, lw=1.5, linestyle="--", alpha=0.7)
    ax.plot([0, proj_scalar], [0, 0], color=GREEN, lw=4, alpha=0.8)
    ax.text(proj_scalar/2, -0.18, f"shadow = {proj_scalar:.3f}", color=GREEN, fontsize=9, ha="center")
    ax.text(0.05, 1.9,
            f"y.uj = {np.dot(y,u):.3f}\n||uj|| = {np.linalg.norm(u):.3f}\n-> {proj_scalar:.3f}",
            color=WHITE, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor=GREY))

    ax2 = axes[1]
    ax2.set_facecolor(BG); ax2.set_xlim(-0.3, 2.8); ax2.set_ylim(-0.4, 2.2)
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color=GREY, lw=0.5); ax2.axvline(0, color=GREY, lw=0.5)
    ax2.set_title("Step 2: Divide by ||uj|| again\nto shrink uj to unit length", color=DIMMED, fontsize=10)
    arrow(ax2, (0,0), u, BLUE, "uj", offset=(0.06, -0.2))
    arrow(ax2, (0,0), u_hat, GREEN, "uj-hat (unit)", offset=(0.06, -0.2))
    ax2.text(1.05, -0.32, f"||uj||={np.linalg.norm(u):.1f} -> 1", color=GREEN, fontsize=8)
    ax2.text(0.05, 1.9,
             f"cj = shadow / ||uj||\n   = {proj_scalar:.3f} / {np.linalg.norm(u):.3f}\n   = {c_j:.3f}",
             color=WHITE, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor=GREY))
    ax2.text(1.3, 1.0, "Why divide twice?\nSo cj x uj lands\nexactly on y's\ncomponent",
             color=DIMMED, fontsize=8.5, style="italic")

    ax3 = axes[2]
    ax3.set_facecolor(BG); ax3.set_xlim(-0.3, 2.8); ax3.set_ylim(-0.4, 2.2)
    ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color=GREY, lw=0.5); ax3.axvline(0, color=GREY, lw=0.5)
    ax3.set_title("Verify: cj x uj = component of y along uj", color=DIMMED, fontsize=10)
    arrow(ax3, (0,0), y, YELLOW, "y", offset=(0.06, 0.06))
    arrow(ax3, (0,0), proj_vec, GREEN, "cj*uj [OK]", offset=(0.06, -0.2))
    ax3.plot([proj_vec[0], y[0]], [proj_vec[1], y[1]], color=RED, lw=1.5, linestyle="--", alpha=0.7)
    ax3.text(0.05, 1.9,
             f"cj = {c_j:.3f}\ncj x uj = [{proj_vec[0]:.3f}, {proj_vec[1]:.3f}]\n(x-component matches y [OK])",
             color=WHITE, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor=GREY))

    fig.text(0.5, -0.04,
             "cj = (y.uj / ||uj||) x (1/||uj||)  =  y.uj / ||uj||^2  =  y.uj / uj.uj",
             ha="center", color=YELLOW, fontsize=11, style="italic")
    plt.tight_layout()
    save(fig, "fig1_weights_two_operations.png")

    print(f"[fig1] scalar projection = {proj_scalar:.6f}")
    print(f"[fig1] c_j               = {c_j:.6f}")
    print(f"[fig1] c_j * u           = {c_j*u}")
    print(f"[fig1] y x-component     = {y[0]:.6f}  (should match c_j*u[0] = {c_j*u[0]:.6f})")


def fig2_orthogonality():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("What Does u.v = 0 Actually Mean?", fontsize=13, color=WHITE, y=1.02)

    configs = [
        (np.array([2.0, 0.5]), np.array([2.2, 1.8]), BLUE, YELLOW,
         "Pointing together", "Large positive dot product\nHigh directional overlap"),
        (np.array([2.0, 0.0]), np.array([0.0, 2.0]), GREEN, RED,
         "Perpendicular (orthogonal)", "Dot product = 0\nShare no direction — fully independent"),
        (np.array([2.0, 0.5]), np.array([-1.8, 0.8]), BLUE, RED,
         "Pointing apart", "Negative dot product\nOpposing directional components"),
    ]

    for ax, (u, v, cu, cv, title, subtitle) in zip(axes, configs):
        ax.set_facecolor(BG); ax.set_xlim(-2.8, 2.8); ax.set_ylim(-0.5, 2.8)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5)
        ax.set_title(title, color=WHITE, fontsize=11, pad=8)
        arrow(ax, (0,0), u, cu, "u", offset=(0.08, -0.22))
        arrow(ax, (0,0), v, cv, "v", offset=(0.08,  0.08))
        dot = np.dot(u, v)
        dot_color = GREEN if abs(dot) < 0.01 else (YELLOW if dot > 0 else RED)
        ax.text(0, -0.38, f"u.v = {dot:.2f}", ha="center", color=dot_color, fontsize=12, fontweight="bold")
        ax.text(0, -0.65, subtitle, ha="center", color=DIMMED, fontsize=8.5, style="italic")

    fig.text(0.5, -0.08,
             "In ML: orthogonal features carry zero redundant information. PCA finds orthogonal directions for exactly this reason.",
             ha="center", color=DIMMED, fontsize=10, style="italic")
    plt.tight_layout()
    save(fig, "fig2_orthogonality.png")

    for i, (u, v, *_) in enumerate(configs):
        print(f"[fig2] pair {i+1}: u.v = {np.dot(u,v):.4f}")


def fig3_cosine_bridge():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cosine Similarity: The Formula You Know From Physics Class",
                 fontsize=13, color=WHITE, y=1.02)

    u = np.array([2.4, 1.0]); v = np.array([1.8, 2.0])
    dot = np.dot(u, v); nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    theta = np.degrees(np.arccos(np.clip(dot/(nu*nv), -1, 1)))
    sim = dot / (nu * nv)

    ax = axes[0]
    ax.set_facecolor(BG); ax.set_xlim(-0.3, 3.2); ax.set_ylim(-0.5, 2.8)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5)
    ax.set_title("Physics class: u.v = ||u|| ||v|| cos(theta)", color=DIMMED, fontsize=10)
    arrow(ax, (0,0), u, BLUE,   "u", offset=(0.06, -0.2))
    arrow(ax, (0,0), v, YELLOW, "v", offset=(0.06,  0.06))
    ang1 = np.degrees(np.arctan2(u[1], u[0])); ang2 = np.degrees(np.arctan2(v[1], v[0]))
    arc = patches.Arc((0,0), 0.8, 0.8, angle=0,
                      theta1=min(ang1,ang2), theta2=max(ang1,ang2), color=GREEN, lw=1.5)
    ax.add_patch(arc)
    mid = np.radians((ang1+ang2)/2)
    ax.text(0.52*np.cos(mid)-0.08, 0.52*np.sin(mid), f"theta={theta:.1f}", color=GREEN, fontsize=9)
    ax.text(0.1, 2.55,
            f"u.v = {dot:.3f}\n||u||*||v|| = {nu:.3f}x{nv:.3f} = {nu*nv:.3f}\ncos(theta) = {np.cos(np.radians(theta)):.3f}",
            color=WHITE, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a", edgecolor=GREY))

    ax2 = axes[1]
    ax2.set_facecolor(BG); ax2.set_xlim(-0.3, 3.2); ax2.set_ylim(-0.5, 2.8)
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color=GREY, lw=0.5); ax2.axvline(0, color=GREY, lw=0.5)
    ax2.set_title("ML Library: cosine_similarity(u, v)", color=DIMMED, fontsize=10)
    arrow(ax2, (0,0), u/nu, BLUE,   "u-hat", offset=(0.06, -0.2))
    arrow(ax2, (0,0), v/nv, YELLOW, "v-hat", offset=(0.06,  0.06))
    ax2.text(0.1, 2.55,
             f"Normalize both to unit length:\nu-hat = u/||u||,  v-hat = v/||v||\nu-hat . v-hat = cos(theta) = {sim:.4f}",
             color=WHITE, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a", edgecolor=GREY))
    ax2.text(0.6, 0.4, f"cosine_similarity = {sim:.4f}",
             color=GREEN, fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a", edgecolor=GREEN))

    fig.text(0.5, -0.04,
             "cosine_similarity(u,v) = u.v / (||u||*||v||) = cos(theta)  --  the physics formula, rearranged.",
             ha="center", color=DIMMED, fontsize=10, style="italic")
    plt.tight_layout()
    save(fig, "fig3_cosine_bridge.png")

    print(f"[fig3] dot(u,v)          = {dot:.6f}")
    print(f"[fig3] ||u|| * ||v||     = {nu*nv:.6f}")
    print(f"[fig3] cosine_similarity = {sim:.6f}")
    print(f"[fig3] theta             = {theta:.4f} degrees")


def fig4_attention_connection():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Transformer Attention: The Weights Formula at Scale",
                 fontsize=13, color=WHITE, y=1.02)

    ax = axes[0]
    ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    ax.text(5, 9.3, "Linear Algebra Class", color=YELLOW, fontsize=12, ha="center", fontweight="bold")
    ax.text(5, 8.3, "cj = y.uj / uj.uj", color=YELLOW, fontsize=13, ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor=YELLOW))
    ax.text(5, 7.2, "How much of y points in the uj direction?", color=DIMMED, fontsize=9, ha="center")
    ax.plot([2, 8], [6.5, 6.5], color=GREY, lw=1, linestyle="--", alpha=0.5)
    ax.text(5, 6.1, "same operation", color=GREY, fontsize=9, ha="center", style="italic")
    ax.text(5, 5.3, "Transformer Attention", color=RED, fontsize=12, ha="center", fontweight="bold")
    ax.text(5, 4.3, "score(q,k) = q.k / sqrt(d)", color=RED, fontsize=13, ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor=RED))
    ax.text(5, 3.2, "How much should this token\nattend to that token?", color=DIMMED, fontsize=9, ha="center")
    ax.text(5, 1.8,
            "y  ->  query vector (q)\nuj ->  key vector (k)\nsqrt(d)  ->  scaling (replaces uj.uj normalization)",
            color=WHITE, fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor=GREY))
    ax.add_patch(patches.FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1",
                 edgecolor=GREY, facecolor="none", lw=1, alpha=0.3))

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.set_title('Attention scores: dot product decides who attends to whom', color=DIMMED, fontsize=9)
    ax2.grid(True, alpha=0.3, axis="x")

    np.random.seed(7)
    d = 4
    q = np.array([0.8, 0.3, 0.6, 0.1])
    tokens = {
        "river": np.array([0.7, 0.2, 0.5, 0.2]),
        "money": np.array([0.2, 0.8, 0.1, 0.7]),
        "the":   np.array([0.1, 0.1, 0.2, 0.1]),
        "sat":   np.array([0.2, 0.3, 0.1, 0.2]),
        "on":    np.array([0.1, 0.2, 0.1, 0.1]),
    }
    scores = {tok: np.dot(q, k) / np.sqrt(d) for tok, k in tokens.items()}
    raw = np.array(list(scores.values()))
    exp = np.exp(raw - raw.max())
    softmax_scores = exp / exp.sum()
    softmax_dict = dict(zip(scores.keys(), softmax_scores))
    sorted_items = sorted(softmax_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k,_ in sorted_items]; vals = [v for _,v in sorted_items]
    bar_colors = [BLUE if v > 0.25 else (YELLOW if v > 0.15 else DIMMED) for v in vals]

    bars = ax2.barh(labels, vals, color=bar_colors, height=0.5, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax2.text(val+0.005, bar.get_y()+bar.get_height()/2,
                 f"{val:.3f}", va="center", color=WHITE, fontsize=10)
    ax2.set_xlabel("Attention weight (after softmax)", color=WHITE, fontsize=9)
    ax2.set_xlim(0, max(vals)*1.25)
    ax2.tick_params(colors=WHITE)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.text(max(vals)*0.5, len(labels)-0.3, 'query token: "bank"', color=GREEN, fontsize=9, style="italic")

    plt.tight_layout()
    save(fig, "fig4_attention_connection.png")

    print(f"[fig4] Attention scores (softmax) for query 'bank':")
    for tok, sc in sorted_items:
        print(f"  {tok:10s}: {sc:.6f}")


def fig5_rag_pipeline():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Your RAG Pipeline Is Running This Formula Millions of Times",
                 fontsize=13, color=WHITE, y=1.02)

    np.random.seed(42)
    query_vec = np.array([0.85, 0.52])
    documents = {
        "Attention Mechanism\n(transformers)": np.array([0.82, 0.57]),
        "Self-Attention\n(BERT paper)":        np.array([0.78, 0.63]),
        "Convolutional Nets\n(image CNNs)":    np.array([0.55, 0.84]),
        "K-Means Clustering":                  np.array([0.95, 0.31]),
        "Decision Trees\n(ML basics)":         np.array([0.30, 0.95]),
    }

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = {doc: cosine_sim(query_vec, vec) for doc, vec in documents.items()}
    sorted_docs = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    col_map = {
        "Attention Mechanism\n(transformers)": BLUE,
        "Self-Attention\n(BERT paper)":        YELLOW,
        "Convolutional Nets\n(image CNNs)":    DIMMED,
        "K-Means Clustering":                  DIMMED,
        "Decision Trees\n(ML basics)":         RED,
    }

    ax = axes[0]
    ax.set_facecolor(BG); ax.set_xlim(-0.05, 1.15); ax.set_ylim(-0.05, 1.15)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Document vectors in embedding space (2D projection)", color=DIMMED, fontsize=9)
    ax.annotate("", xy=query_vec, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.5, mutation_scale=18))
    ax.text(query_vec[0]+0.02, query_vec[1]+0.02, "query", color=GREEN, fontsize=10, fontweight="bold")
    for doc, vec in documents.items():
        col = col_map[doc]
        alpha = 0.9 if sims[doc] > 0.99 else (0.65 if sims[doc] > 0.9 else 0.4)
        ax.annotate("", xy=vec, xytext=(0,0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=14, alpha=alpha))
        ax.text(vec[0]+0.02, vec[1]+0.01, doc.split("\n")[0][:18], color=col, fontsize=7.5, alpha=alpha+0.1)

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.set_title("Retrieved — ranked by cosine similarity", color=DIMMED, fontsize=9)
    ax2.grid(True, alpha=0.3, axis="x")
    labels_s = [k.replace("\n"," ") for k,_ in sorted_docs]
    scores_s = [v for _,v in sorted_docs]
    bar_cols  = [col_map[k] for k,_ in sorted_docs]
    bars = ax2.barh(range(len(labels_s)), scores_s, color=bar_cols, height=0.5, alpha=0.85)
    ax2.set_yticks(range(len(labels_s))); ax2.set_yticklabels(labels_s, fontsize=8.5, color=WHITE)
    ax2.set_xlim(0.0, 1.12); ax2.set_xlabel("Cosine Similarity to Query", color=WHITE, fontsize=9)
    for bar, score, (label, _) in zip(bars, scores_s, sorted_docs):
        rank_color = GREEN if score > 0.97 else (YELLOW if score > 0.88 else RED)
        ax2.text(score+0.005, bar.get_y()+bar.get_height()/2,
                 f"{score:.4f}", va="center", color=rank_color, fontsize=9, fontweight="bold")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.text(0.5, -0.04,
             "The retrieval step is u.v / (||u||*||v||), run once per document per query. That is all it is.",
             ha="center", color=DIMMED, fontsize=10, style="italic")
    plt.tight_layout()
    save(fig, "fig5_rag_pipeline.png")

    print(f"[fig5] Cosine similarities:")
    for doc, sim in sorted_docs:
        print(f"  {doc.split(chr(10))[0][:35]:35s}: {sim:.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Article 4A  — Inner Products, Norms, Orthogonality")
    print("Generating all figures with verified outputs...")
    print("=" * 60)
    print()
    fig0_three_disguises()
    print()
    fig1_weights_two_operations()
    print()
    fig2_orthogonality()
    print()
    fig3_cosine_bridge()
    print()
    fig4_attention_connection()
    print()
    fig5_rag_pipeline()
    print()
    print("=" * 60)
    print("All 6 figures generated. Check Visuals/ directory.")
    print("=" * 60)