"""Sharper adversarial test of the relaxed (marginal-only) bound of S11(d).
R(q, psi) = max over distributions P on maps s:[m]->[m] with sum_s P_s s#q = q of  sum P a_s^2 / sum P G_s
(Charnes-Cooper LP).  Conjecture: R <= 1 - 1/m.  Unlike c27 (objective flat at 0), R varies smoothly-ish."""
import itertools, sys
import numpy as np
from scipy.optimize import linprog, minimize
m = int(sys.argv[1]); restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
maps = np.array(list(itertools.product(range(m), repeat=m)))
push = np.zeros((len(maps), m))
def R(z):
    q = np.exp(z[:m]); q /= q.sum()
    psi = z[m:]; psi = psi - q @ psi; nrm = np.sqrt(q @ psi ** 2)
    if nrm < 1e-9: return 0.0
    psi = psi / nrm
    a = (q[None, :] * psi[maps]).sum(1); G = (q[None, :] * (psi[maps] - psi[None, :]) ** 2).sum(1)
    pu = np.zeros((len(maps), m))
    for i, s in enumerate(maps): np.add.at(pu[i], s, q)
    # variables Y (len maps) and t ; max a^2.Y  s.t. G.Y = 1, pu^T Y - t q = 0, 1.Y - t = 0, Y,t >= 0
    nv = len(maps) + 1
    Aeq = np.zeros((m + 2, nv)); beq = np.zeros(m + 2)
    Aeq[0, :-1] = G; beq[0] = 1
    Aeq[1:m + 1, :-1] = pu.T; Aeq[1:m + 1, -1] = -q
    Aeq[m + 1, :-1] = 1; Aeq[m + 1, -1] = -1
    c = np.concatenate([-(a ** 2), [0.0]])
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * nv, method='highs')
    return -res.fun if res.status == 0 else 0.0
best = 0.0; arg = None
for r in range(restarts):
    z0 = np.concatenate([rng.normal(0, 1.5, m), rng.normal(0, 1, m)])
    res = minimize(lambda z: -R(z), z0, method='Nelder-Mead', options=dict(maxiter=600 * m, xatol=1e-10, fatol=1e-13))
    if -res.fun > best:
        best = -res.fun; q = np.exp(res.x[:m]); arg = q / q.sum()
    print(f'restart {r}: R = {-res.fun:.12f}', flush=True)
samp = max(R(np.concatenate([rng.normal(0, 2, m), rng.normal(0, 1, m)])) for _ in range(300))
print(f'm={m}: max R found = {best:.12f}  (1-1/m = {1 - 1 / m:.12f});  best q = {np.round(arg, 4)};  random-sample max = {samp:.6f}')
