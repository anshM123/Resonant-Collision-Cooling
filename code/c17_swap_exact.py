"""Robust optimum of swap-only protocols with EXACT endpoints.
M machines: p_0 = Gibbs(beta) -> p_1 -> ... -> p_M = Gibbs(lam*beta) exactly; minimise sum_n D(p_{n-1}||p_n) over the
M-1 intermediate states (log-parametrised, so tiny populations are handled), then re-evaluate in 40-digit arithmetic.
The problem is jointly convex in (p_1..p_{M-1}), so the minimum found from the Gibbs-path start is the global one.
usage: c17_swap_exact.py d beta lam M [spectrum]
"""
import sys
import numpy as np
import torch
import mpmath as mp
torch.set_default_dtype(torch.float64)
mp.mp.dps = 40

d, beta, lam, M = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
E = np.arange(d, dtype=float) if len(sys.argv) <= 5 else np.array([float(v) for v in sys.argv[5].split(',')])
Et = torch.tensor(E)
def gibbs_log(b):
    l = -b * Et; return l - torch.logsumexp(l, 0)
lp0, lpT = gibbs_log(beta), gibbs_log(beta * lam)
bs = np.linspace(beta, beta * lam, M + 1)
# start on the Fisher-Rao geodesic (great circle in sqrt(p)), uniformly spaced -- the swap optimum is close to it
a = torch.exp(0.5 * lp0); b_ = torch.exp(0.5 * lpT); th = torch.arccos(torch.clamp(a @ b_, -1, 1))
ts = torch.linspace(0, 1, M + 1)[1:-1]
geo = [((torch.sin((1 - t) * th) * a + torch.sin(t * th) * b_) / torch.sin(th)) ** 2 for t in ts]
z = torch.stack([torch.log(g / g.sum()) for g in geo]).clone().requires_grad_(True)

def total(z):
    lps = [lp0] + [zz - torch.logsumexp(zz, 0) for zz in z] + [lpT]
    tot = torch.zeros(())
    for a, b in zip(lps[:-1], lps[1:]):
        tot = tot + torch.sum(torch.exp(a) * (a - b))
    return tot

opt = torch.optim.LBFGS([z], lr=1, max_iter=4000, tolerance_grad=1e-16, tolerance_change=1e-20, history_size=50,
                        line_search_fn='strong_wolfe')
def closure():
    opt.zero_grad(); t = total(z); t.backward(); return t
for _ in range(3):
    opt.step(closure)
with torch.no_grad():
    val = total(z).item()
    lps = [lp0.numpy()] + [(zz - torch.logsumexp(zz, 0)).numpy() for zz in z] + [lpT.numpy()]
# 40-digit re-evaluation of the same path (endpoints exact Gibbs in mpmath)
def gibbs_mp(b):
    w = [mp.e ** (-mp.mpf(b) * mp.mpf(e)) for e in E]; Z = sum(w); return [x / Z for x in w]
P = [gibbs_mp(beta)] + [[mp.e ** mp.mpf(float(v)) for v in l] for l in lps[1:-1]] + [gibbs_mp(beta * lam)]
P = [[x / sum(row) for x in row] for row in P]
tot = sum(sum(a[i] * mp.log(a[i] / b[i]) for i in range(d)) for a, b in zip(P[:-1], P[1:]))
print(f'd={d} beta={beta} lam={lam} M={M} spectrum={list(E)}: swap optimum (exact endpoints) = {mp.nstr(tot, 10)}  (float {val:.10e})')
