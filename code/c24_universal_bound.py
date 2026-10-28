"""Test the conjectured universal bound: for every system state p, every resonant permutation (any machine state q*)
and every machine perturbation Theta,
        ||delta||_F^2  <=  (d_M - 1) * 2Q            (local cost >= swap cost / (d_M - 1)),
i.e. lambda_max(pi, q*) := max_Theta ||delta||_F^2 / 2Q  <=  d_M - 1.   (Proved at uniform p; tested here elsewhere.)
For d = d_M = 3 we enumerate all 9! permutations, solve the resonance equations at p, and scan each resonance family.
usage: c24_universal_bound.py ntrials [seed]
"""
import itertools, sys
import numpy as np

d = dm = 3
n = d * dm
cx, cy = np.arange(n) // dm, np.arange(n) % dm
B = np.zeros((dm, dm - 1)); B[:dm - 1] = np.eye(dm - 1); B[dm - 1] = -1
PERMS = np.array(list(itertools.permutations(range(n))), dtype=np.int64)

def lam_max(pi, p, q):
    r = p[cx] * q[cy]
    D = (p[cx][:, None] * B[cy])                    # (9, 2): D = p x Theta, Theta = B v
    X = np.empty_like(D); X[pi] = D; X = X - D
    M = np.zeros((d, dm - 1))
    for c in range(n): M[cx[c]] += X[c]
    A = X.T @ (X / r[:, None]) - M.T @ (M / p[:, None])
    F = M.T @ (M / p[:, None])
    A = 0.5 * (A + A.T); F = 0.5 * (F + F.T)
    wa, Va = np.linalg.eigh(A)
    keep = wa > 1e-12 * max(1.0, np.abs(wa).max())
    # null directions of A must not move the system
    if (~keep).any() and np.linalg.norm(M @ Va[:, ~keep]) > 1e-9 * max(1.0, np.abs(M).max()):
        return np.inf
    W = Va[:, keep] / np.sqrt(wa[keep])
    Fr = W.T @ F @ W
    return float(np.linalg.eigvalsh(0.5 * (Fr + Fr.T)).max()) if Fr.size else 0.0

def resonant(p):
    """(pi, particular log q*, kernel basis without normalisation) for permutations resonant at p that move the system"""
    lp = np.log(p)
    out = []
    for pi in PERMS:
        moved = np.where(pi != np.arange(n))[0]
        if len(moved) == 0: continue
        x, y, x2, y2 = cx[moved], cy[moved], cx[pi[moved]], cy[pi[moved]]
        if np.any((y == y2) & (x != x2)): continue
        A = np.zeros((len(moved), dm)); A[np.arange(len(moved)), y] += 1; A[np.arange(len(moved)), y2] -= 1
        b = lp[x2] - lp[x]
        u0, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.max(np.abs(A @ u0 - b)) > 1e-9: continue
        _, s, vt = np.linalg.svd(A); ker = vt[int(np.sum(s > 1e-9)):]
        if len(ker):
            one = np.ones(dm) / np.sqrt(dm); ker = ker - np.outer(ker @ one, one)
            _, s2, vt2 = np.linalg.svd(ker); ker = vt2[:int(np.sum(s2 > 1e-9))]
        out.append((pi, u0, ker))
    return out

def worst_at(p, rng, nsamp=40):
    best = (0.0, None)
    fam = resonant(p)
    for pi, u0, K in fam:
        ts = [np.zeros(0)] if len(K) == 0 else [rng.normal(0, 4, len(K)) for _ in range(nsamp)]
        for t in ts:
            lq = u0 + (t @ K if len(K) else 0.0)
            q = np.exp(lq - lq.max()); q /= q.sum()
            v = lam_max(pi, p, q)
            if v > best[0]: best = (v, (list(pi), np.round(q, 4)))
    return best, len(fam)

if __name__ == '__main__':
    ntr = int(sys.argv[1]); rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    E = np.arange(d, dtype=float)
    tests = []
    for b in [0.0, 0.05, 0.3, 1.0, 1.6, 3.0, 6.0, 12.0]:            # equally spaced Gibbs states (resonance-rich)
        v = np.exp(-b * E); tests.append(('gibbs b=%g' % b, v / v.sum()))
    for k in range(ntr):                                             # random spectra / random states
        if k % 2 == 0:
            Ek = np.sort(rng.uniform(0, 3, d)); Ek -= Ek[0]; b = rng.uniform(0.1, 5)
            v = np.exp(-b * Ek); tests.append(('gibbs E=%s b=%.2f' % (np.round(Ek, 2), b), v / v.sum()))
        else:
            tests.append(('random p', rng.dirichlet(np.ones(d))))
    overall = 0.0
    for name, p in tests:
        (v, arg), nf = worst_at(p, rng)
        overall = max(overall, v)
        print(f'{name:32s} resonant movers={nf:4d}   max lambda = {v:.6f}  (bound d_M-1 = {dm-1})', flush=True)
    print('OVERALL max lambda =', overall)
