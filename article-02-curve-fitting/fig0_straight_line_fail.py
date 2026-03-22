import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
 
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
 
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([35, 48, 60, 70, 78, 84, 89, 92, 94, 95], dtype=float)
 
model = LinearRegression().fit(x.reshape(-1,1), y)
a = model.coef_[0]
b = model.intercept_
 
x_smooth = np.linspace(0, 13, 300)
y_line   = model.predict(x_smooth.reshape(-1,1))
 
fig, ax = plt.subplots(figsize=(10, 5))
 
# Impossible zone
ax.fill_between(x_smooth, 100, 130, alpha=0.12, color='#e74c3c')
ax.axhline(100, color='#e74c3c', lw=1.5, linestyle='--', alpha=0.8,
           label='Score ceiling (100) — impossible above this')
 
# Line and data
ax.plot(x_smooth, y_line, color='#e74c3c', lw=2.5,
        label=f'Straight line: y = {a:.1f}x + {b:.1f}')
ax.scatter(x, y, color='white', zorder=5, s=70, label='Actual data')
 
# Error lines
for xi, yi in zip(x, y):
    y_hat = model.predict([[xi]])[0]
    ax.plot([xi, xi], [yi, y_hat], color='#ffffff', alpha=0.25, lw=1.2, linestyle=':')
 
# Three key annotations
checks = [
    (1,  model.predict([[1]])[0],  y[0],
     'x=1: predicts 35.3\nactual 35  [OK]',     '#2ecc71',  (1.4,  28)),
    (5,  model.predict([[5]])[0],  y[4],
     'x=5: predicts 60.5\nactual 78  [WRONG]',  '#f39c12',  (5.4,  48)),
    (12, model.predict([[12]])[0], None,
     'x=12: predicts 104.6\nImpossible score!', '#e74c3c',  (9.5, 108)),
]
 
for (xi, y_pred, _, label, color, txtpos) in checks:
    ax.scatter([xi], [y_pred], color=color, zorder=6, s=100, marker='D')
    ax.annotate(label,
                xy=(xi, y_pred),
                xytext=txtpos,
                fontsize=8, color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#111111', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
 
ax.text(11, 116, 'IMPOSSIBLE\nZONE', fontsize=8, color='#e74c3c',
        ha='center', alpha=0.8, style='italic')
 
ax.set_title('A Straight Line on Curved Data — Confidently Wrong at the Extremes',
             fontsize=11, color='#eeeeee', pad=12)
ax.set_xlabel('Study Hours', fontsize=10)
ax.set_ylabel('Exam Score', fontsize=10)
ax.legend(fontsize=8.5, facecolor='#1a1a1a', edgecolor='#333333', loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 13)
ax.set_ylim(20, 130)
 
plt.tight_layout()
plt.savefig('fig0_straight_line_fail.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved.")