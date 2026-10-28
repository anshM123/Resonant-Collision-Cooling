import numpy as np, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
E = np.array([0., 1., 2.])
def gibbs(b):
    v = np.exp(-b * E); return v / v.sum()
def Lstar2half(b0, lam):
    L = 2 * np.arccos(np.sqrt(gibbs(b0)) @ np.sqrt(gibbs(lam * b0))); return L * L / 2

# (a) local ratio along the Gibbs curve: exact hull gauge, primal-dual bracket from c22_hull_bounds.py
import re
pts = {}
logs = sorted(glob.glob(os.path.join(root, 'fig1a_exact*.log')) + glob.glob(os.path.join(root, 'code', 'fig1a_exact*.log')))
for line in (l for f in logs for l in open(f)):
    m = re.match(r'beta=\s*([0-9.]+)\s+exact hull ratio in \[([0-9.]+), ([0-9.]+)\]\s+\(gap ([-0-9.e+]+)', line)
    if m:
        lo, up, gap = float(m.group(2)), float(m.group(3)), float(m.group(4))
        if abs(gap) > 2e-7:              # only certified points (primal-dual gap < 2e-7) are plotted
            print('skipping uncertified point:', line.strip()[:60]); continue
        pts[float(m.group(1))] = 0.5 * (lo + up)
bs = np.array(sorted(pts)); rs = np.array([pts[b] for b in bs])

# (b) exact protocols: N * Sigma_N / (L*^2/2)
Ns = np.array([40, 80, 160])
win = {
    r'$\beta\omega\!: 0.01\to0.05$': dict(L=Lstar2half(0.01, 5),
        res=[7.05825808497648e-6, 3.53064167749038e-6, 1.76498116706219e-6],
        swp=[1.3326366e-5, 6.663167e-6, 3.331579e-6]),
    r'$0.1\to0.5$': dict(L=Lstar2half(0.1, 5),
        res=[9.67451134726498e-4, 4.8342849759049e-4, 0.038663 / 160],
        swp=[1.266846059e-3, 6.33268414e-4, 3.16595604e-4]),
    r'$3\to9$': dict(L=Lstar2half(3, 3),
        res=[1.79413727268603e-3, 8.8700188072431e-4, 4.41011879457423e-4],
        swp=[2.3471376482e-3, 1.1586947207e-3, 5.7564366548e-4]),
}
cols = ['#2a78d6', '#eb6834', '#1baf7a']
marks = ['o', 's', 'D']
ink, muted = '#1f1f1e', '#6b6a63'

plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.edgecolor': muted, 'xtick.color': muted, 'ytick.color': muted, 'axes.labelcolor': ink,
                     'font.family': 'serif'})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.5), gridspec_kw=dict(height_ratios=[1, 1.05], hspace=0.42))

ax1.axhline(1.0, color=muted, lw=0.8, ls='--')
ax1.axhline(1 / np.sqrt(2), color=muted, lw=0.8, ls=':')
ax1.text(0.021, 1.008, 'swaps (thermodynamic length)', color=muted, fontsize=6.5, va='bottom')
ax1.text(0.021, 0.692, r'$1/\sqrt{2}$', color=muted, fontsize=6.5, va='top')
ax1.plot(bs, rs, color=cols[0], lw=1.6, marker='o', ms=3.2, mfc=cols[0], mec='white', mew=0.5, zorder=3)
ax1.set_xscale('log'); ax1.set_xlim(0.015, 20); ax1.set_ylim(0.66, 1.05)
ax1.set_xlabel(r'inverse temperature $\beta\omega$')
ax1.set_ylabel(r'$\|\dot p\|_{\rm eff}/\|\dot p\|_F$')
ax1.text(-0.2, 1.02, '(a)', transform=ax1.transAxes, fontsize=9, fontweight='bold', color=ink)
for s in ['top', 'right']: ax1.spines[s].set_visible(False)

for (lab, w), c, mk, dy in zip(win.items(), cols, marks, [0.0, -0.032, 0.032]):
    yr = Ns * np.array(w['res']) / w['L']; ys = Ns * np.array(w['swp']) / w['L']
    ax2.plot(Ns, ys, color=c, lw=1.2, ls='--', marker=mk, ms=4.5, mfc='white', mec=c, mew=1.0, zorder=2)
    ax2.plot(Ns, yr, color=c, lw=1.6, marker=mk, ms=4.5, mfc=c, mec='white', mew=0.6, zorder=3)
    ax2.text(Ns[-1] * 1.12, yr[-1] + dy, lab, color=ink, fontsize=6.5, va='center')
ax2.axhline(1.0, color=muted, lw=0.6, ls=':')
ax2.text(42, 0.925, 'optimal swap protocols (open)', color=muted, fontsize=6.5)
ax2.text(42, 0.60, 'resonant protocols (filled)', color=muted, fontsize=6.5)
ax2.set_xscale('log', base=2); ax2.set_xticks(Ns); ax2.set_xticklabels([str(n) for n in Ns])
ax2.set_xlim(34, 420); ax2.set_ylim(0.45, 1.1)
ax2.set_xlabel(r'number of machines $N$')
ax2.set_ylabel(r'$N\Sigma_N\,/\,[(L^*)^2/2]$')
ax2.text(-0.2, 1.02, '(b)', transform=ax2.transAxes, fontsize=9, fontweight='bold', color=ink)
for s in ['top', 'right']: ax2.spines[s].set_visible(False)

fig.savefig(os.path.join(here, 'fig1.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(here, 'fig1.png'), dpi=200, bbox_inches='tight')
for lab, w in win.items():
    print(lab, 'res/L:', np.round(Ns * np.array(w['res']) / w['L'], 4), 'swap/L:', np.round(Ns * np.array(w['swp']) / w['L'], 4))
