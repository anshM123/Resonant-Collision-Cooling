import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
# beta*omega = 8 -> 16, 41 machines. Resonant: 40-digit re-evaluation (d=3: pi_C every step; d>=4: d-cycle in 7 of
# every 8 steps + swap). Swap: optimum with exact endpoints (c17_swap_exact.py).
ds = np.array([3, 4, 5, 6, 7, 8])
res = np.array([8.19266e-6, 6.06049e-6, 4.61264e-6, 3.72964e-6, 3.13679e-6, 2.71421e-6])
swp = np.array([1.6287630e-5, 1.6287626e-5, 1.6287627e-5, 1.6287627e-5, 1.6287623e-5, 1.6287623e-5])
ratios = res / swp
dd = np.linspace(2.8, 8.2, 200)
blue, muted, ink = '#2a78d6', '#6b6a63', '#1f1f1e'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': muted, 'xtick.color': muted,
                     'ytick.color': muted, 'axes.labelcolor': ink, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.plot(dd, 1 / (dd - 1), color=muted, lw=1.0, ls='--', zorder=1)
ax.text(5.1, 0.15, r'$1/(d-1)$', color=muted, fontsize=7)
ax.axhline(1.0, color=muted, lw=0.6, ls=':')
ax.text(3.0, 0.965, 'optimal swap protocols', color=muted, fontsize=6.5, va='top')
ax.plot(ds, ratios, color=blue, lw=1.6, marker='o', ms=4.5, mfc=blue, mec='white', mew=0.6, zorder=3)
for d, r in zip(ds, ratios):
    ax.text(d + 0.08, r + 0.04, f'{r:.2f}', color=ink, fontsize=6.5, ha='left')
ax.set_xlabel(r'dimension $d$ of system and machines')
ax.set_ylabel('dissipation / optimal swap')
ax.set_xticks(ds); ax.set_ylim(0, 1.05); ax.set_xlim(2.7, 8.5)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
fig.savefig(os.path.join(here, 'fig2.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(here, 'fig2.png'), dpi=200, bbox_inches='tight')
print(dict(zip(ds.tolist(), np.round(ratios, 4).tolist())))
