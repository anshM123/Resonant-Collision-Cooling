"""Universal-bound test (Theorem 2 conjecture) for larger dimensions, beyond exhaustive enumeration.
On the Gibbs curve of the equally spaced d-level system, a permutation is resonant iff it conserves H_S + H_M for some
machine spectrum e (machine state q = Gibbs(beta, e)).  We take integer spectra e (e_0 = 0 <= ... <= kmax) and all (or
sampled) energy-conserving permutations, and compute lambda = max_Theta ||delta||_F^2 / 2Q in the stable variance form.
Claim tested: lambda <= d_M - 1.
usage: c28_bound_large.py d dM kmax max_perms_per_spectrum beta1 [beta2 ...]
"""
import itertools, math, sys
import numpy as np

d, dm, kmax, maxp = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
betas = [float(b) for b in sys.argv[5:]]
rng = np.random.default_rng(0)
n = d * dm

def perms_for(e):
    shells = {}
    for x in range(d):
        for y in range(dm):
            shells.setdefault(x + e[y], []).append(x * dm + y)
    groups = [g for g in shells.values() if len(g) > 1]
    count = math.prod(math.factorial(len(g)) for g in groups) if groups else 1
    out = []
    if count <= maxp:
        for combo in itertools.product(*[itertools.permutations(g) for g in groups]):
            pi = np.arange(n)
            for g, img in zip(groups, combo): pi[list(g)] = list(img)
            out.append(pi)
    else:
        seen = set()
        for _ in range(maxp):
            pi = np.arange(n)
            for g in groups: pi[list(g)] = rng.permutation(g)
            t = tuple(pi)
            if t not in seen: seen.add(t); out.append(pi)
    return out

def lam(pi, p, q):
    inv = np.argsort(pi)
    sig = np.array([[inv[x * dm + y] % dm for y in range(dm)] for x in range(d)])
    Bq = np.eye(dm)[:, :dm - 1] - np.outer(np.ones(dm), q[:dm - 1])
    A = np.zeros((dm - 1, dm - 1)); F = np.zeros((dm - 1, dm - 1))
    for x in range(d):
        L = Bq[sig[x]] - Bq
        m = q @ L
        A += p[x] * (L.T @ (q[:, None] * L) - np.outer(m, m))
        F += p[x] * np.outer(m, m)
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    keep = w > 1e-12 * max(np.abs(w).max(), 1e-300)
    if not keep.any(): return 0.0
    Fn = V.T @ F @ V
    if (~keep).any() and np.abs(Fn[np.ix_(~keep, ~keep)]).max() > 1e-9 * max(np.abs(Fn).max(), 1e-300):
        return np.inf
    W = V[:, keep] / np.sqrt(w[keep])
    return float(np.linalg.eigvalsh(W.T @ F @ W).max())

E = np.arange(d, dtype=float)
specs = [(0,) + e for e in itertools.combinations_with_replacement(range(kmax + 1), dm - 1)]
for beta in betas:
    p = np.exp(-beta * E); p /= p.sum()
    best, cnt, arg = 0.0, 0, None
    for e in specs:
        q = np.exp(-beta * np.array(e, float)); q /= q.sum()
        for pi in perms_for(e):
            v = lam(pi, p, q); cnt += 1
            if v > best: best, arg = v, (e, list(pi))
    print(f'd={d} dM={dm} beta={beta:6.2f}: {cnt} mechanisms, max lambda = {best:.6f} (bound {dm - 1})  at e={arg[0] if arg else None}',
          flush=True)
