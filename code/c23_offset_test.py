"""Near-resonant collisions: can an offset of the incoming state from a resonance manifold help?

At a Gibbs point p* of the equally spaced qutrit (on the manifold p1^2 = p0 p2) take every permutation resonant there,
with its resonant machine q*, and allow the incoming state to be p = p* + eps*R with R along the normal of the manifold
(the only offset that is not equivalent to re-expanding at another resonant point).  With D = R x q* + p* x Theta,
X = P D P^T - D, delta = Tr_M X, 2Q = ||X||^2_{s*} - ||delta||^2_{p*}.  We report
  (a) whether 2Q has a null vector that moves the system (a second-order-free move), and
  (b) the hull gauge along the Gibbs tangent with the offset as an extra free parameter of each mechanism, compared
      with the exactly resonant hull (offset 0).  Treating the offset as free per mechanism only helps, so if (b) does
      not decrease, offsets cannot lower the local cost.
usage: c23_offset_test.py beta1 [beta2 ...]
"""
import sys
import numpy as np
import cvxpy as cp
import importlib.util
spec = importlib.util.spec_from_file_location('c21', 'c21_exact_hull.py'); c21 = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ['c21']; spec.loader.exec_module(c21); sys.argv = _argv
d, n = 3, 9
cx, cy = c21.cx, c21.cy
B = c21.B

def forms(pi, p, q, nrm, with_R):
    """columns: parameters (Theta_1, Theta_2[, R]); returns move map M (3 x k) and 2Q matrix A (k x k)"""
    cols = []
    for j in range(2):
        cols.append(p[cx] * B[cy, j])
    if with_R:
        cols.append(nrm[cx] * q[cy])
    Dcols = np.array(cols).T                                   # (9, k)
    Xc = np.empty_like(Dcols); Xc[pi] = Dcols; Xc = Xc - Dcols  # X = P D P^T - D on the diagonal
    M = np.zeros((d, Dcols.shape[1]))
    for c in range(n): M[cx[c]] += Xc[c]
    r = p[cx] * q[cy]
    A = Xc.T @ (Xc / r[:, None]) - M.T @ (M / p[:, None])
    return M, 0.5 * (A + A.T)

def run(beta, fam):
    E = np.arange(d, dtype=float)
    p = np.exp(-beta * E); p /= p.sum()
    w = -p * (E - p @ E)
    # normal to the manifold f(p) = log p1 - (log p0 + log p2)/2 = 0 in the sum-zero plane (Euclidean)
    grad = np.array([-0.5 / p[0], 1 / p[1], -0.5 / p[2]]); nrm = grad - grad.mean(); nrm /= np.linalg.norm(nrm)
    res = {}
    free = 0
    for with_R in (False, True):
        mechs = [(B, B.T @ (B / p[:, None]))]
        for pi, u0, K in fam:
            ts = [np.zeros(0)] if K.shape[0] == 0 else [np.array([t]) for t in np.arange(-14, 14.001, 0.5)]
            for t in ts:
                lq = beta * u0 + (t @ K if K.shape[0] else 0.0)
                q = np.exp(lq - lq.max()); q /= q.sum()
                M, A = forms(pi, p, q, nrm, with_R)
                if np.linalg.norm(M) < 1e-14: continue
                wv, V = np.linalg.eigh(A)
                null = V[:, wv < 1e-10 * max(1.0, abs(wv).max())]
                if null.size and np.linalg.norm(M @ null) > 1e-8:
                    free += 1
                    if free <= 3:
                        print(f'  beta={beta} R={with_R}: second-order-free move for {list(pi)}: delta =',
                              np.round(M @ null[:, 0], 6), flush=True)
                    continue
                mechs.append((M, A))
        ths = [cp.Variable(M.shape[1]) for M, _ in mechs]
        obj = 0; tot = 0
        for (M, A), th in zip(mechs, ths):
            wv, V = np.linalg.eigh(A); wv = np.clip(wv, 0, None)
            obj += cp.norm((V * np.sqrt(wv)).T @ th, 2); tot += M @ th
        prob = cp.Problem(cp.Minimize(obj), [tot == w]); prob.solve(solver=cp.CLARABEL)
        res[with_R] = prob.value / np.sqrt(np.sum(w ** 2 / p))
    print(f'beta={beta:6.2f}: hull ratio exact-resonant = {res[False]:.6f}, with free normal offsets = {res[True]:.6f}'
          f'   (mechanisms with a second-order-free move: {free})', flush=True)

if __name__ == '__main__':
    fam = c21.resonant_family()
    for b in [float(v) for v in sys.argv[1:]]:
        run(b, fam)
