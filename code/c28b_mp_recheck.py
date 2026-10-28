"""Re-check in 50-digit arithmetic the mechanisms that c28 flags as infinite (ill-conditioned tiny weights)."""
import itertools, math, sys
import numpy as np, mpmath as mp
mp.mp.dps = 50
import importlib.util
d, dm, kmax, maxp, beta = 4, 4, 4, 300, 5.0
sys.argv = ['c28', str(d), str(dm), str(kmax), str(maxp)]
spec = importlib.util.spec_from_file_location('c28', 'c28_bound_large.py'); c28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c28)
E = np.arange(d, dtype=float)
p = np.exp(-beta * E); p /= p.sum()
def lam_mp(pi, e):
    pm = [mp.e ** (-beta * x) for x in range(d)]; Z = sum(pm); pm = [v / Z for v in pm]
    qm = [mp.e ** (-beta * v) for v in e]; Z = sum(qm); qm = [v / Z for v in qm]
    inv = np.argsort(pi)
    sig = [[int(inv[x * dm + y] % dm) for y in range(dm)] for x in range(d)]
    # basis psi_j = e_j - q_j 1, j < dm-1
    def Bq(y, j): return (1 if y == j else 0) - qm[j]
    A = mp.zeros(dm - 1, dm - 1); F = mp.zeros(dm - 1, dm - 1)
    for x in range(d):
        L = [[Bq(sig[x][y], j) - Bq(y, j) for j in range(dm - 1)] for y in range(dm)]
        m = [sum(qm[y] * L[y][j] for y in range(dm)) for j in range(dm - 1)]
        for i in range(dm - 1):
            for j in range(dm - 1):
                A[i, j] += pm[x] * (sum(qm[y] * L[y][i] * L[y][j] for y in range(dm)) - m[i] * m[j])
                F[i, j] += pm[x] * m[i] * m[j]
    w, V = mp.eigsy(A)
    scale = max(abs(v) for v in w)
    keep = [i for i in range(dm - 1) if w[i] > mp.mpf(10) ** (-40) * scale]
    Fn = V.T * F * V
    for i in range(dm - 1):
        if i not in keep and abs(Fn[i, i]) > mp.mpf(10) ** (-35) * max(abs(Fn[a, a]) for a in range(dm - 1)):
            return mp.inf
    W = mp.matrix(dm - 1, len(keep))
    for k, i in enumerate(keep):
        for r in range(dm - 1): W[r, k] = V[r, i] / mp.sqrt(w[i])
    M = W.T * F * W
    return max(mp.eigsy(M)[0])
flag = 0; worst = mp.mpf(0)
for e in [(0,) + t for t in itertools.combinations_with_replacement(range(kmax + 1), dm - 1)]:
    q = np.exp(-beta * np.array(e, float)); q /= q.sum()
    for pi in c28.perms_for(e):
        v = c28.lam(pi, p, q)
        if not np.isfinite(v) or v > 2.99:
            flag += 1
            vm = lam_mp(pi, e)
            worst = max(worst, vm)
            if flag <= 6: print(f'e={e} pi={list(pi)}: double={v}, 50-digit lambda={mp.nstr(vm, 12)}', flush=True)
print(f'flagged {flag} mechanisms; worst 50-digit lambda = {mp.nstr(worst, 15)} (bound 3)')
