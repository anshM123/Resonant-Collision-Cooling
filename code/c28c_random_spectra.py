"""Physical test of the Theorem 2 conjecture for d_M = 4 beyond the equally spaced Gibbs curve: random integer system
spectra E (so that many resonances exist), random temperatures, all integer machine spectra up to emax, and the
energy-conserving permutations (sampled). Reports max lambda vs the bound d_M - 1 (stable variance form, as c28).
usage: c28c_random_spectra.py d dM emax max_perms n_spectra seed
"""
import itertools, math, sys
import numpy as np
d, dm, emax, maxp, nspec, seed = (int(v) for v in sys.argv[1:7])
rng = np.random.default_rng(seed)
n = d * dm
def perms_for(Es, e):
    shells = {}
    for x in range(d):
        for y in range(dm):
            shells.setdefault(Es[x] + e[y], []).append(x * dm + y)
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
        L = Bq[sig[x]] - Bq; m = q @ L
        A += p[x] * (L.T @ (q[:, None] * L) - np.outer(m, m)); F += p[x] * np.outer(m, m)
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    keep = w > 1e-12 * max(np.abs(w).max(), 1e-300)
    if not keep.any(): return 0.0
    Fn = V.T @ F @ V
    if (~keep).any() and np.abs(Fn[np.ix_(~keep, ~keep)]).max() > 1e-9 * max(np.abs(Fn).max(), 1e-300): return np.inf
    W = V[:, keep] / np.sqrt(w[keep]); return float(np.linalg.eigvalsh(W.T @ F @ W).max())
specs = [(0,) + e for e in itertools.combinations_with_replacement(range(emax + 1), dm - 1)]
overall = 0.0; ninf = 0
for k in range(nspec):
    Es = np.concatenate([[0], np.sort(rng.integers(1, emax + 1, d - 1))]).astype(float)
    if len(set(Es)) < d: continue                      # non-degenerate system spectrum
    beta = float(rng.choice([0.05, 0.3, 1.0, 2.0, 3.5]))
    p = np.exp(-beta * Es); p /= p.sum()
    best = 0.0; cnt = 0
    for e in specs:
        q = np.exp(-beta * np.array(e, float)); q /= q.sum()
        if (p[:, None] * q[None, :]).min() < 1e-9: continue
        for pi in perms_for(Es, e):
            v = lam(pi, p, q); cnt += 1
            if np.isinf(v): ninf += 1; continue
            best = max(best, v)
    overall = max(overall, best)
    print(f'E={Es.astype(int)} beta={beta}: {cnt} mechanisms, max lambda = {best:.6f} (bound {dm - 1})', flush=True)
print(f'OVERALL d={d} dM={dm}: max lambda = {overall:.6f}; flagged infinite (ill-conditioned) = {ninf}')
