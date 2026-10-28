"""High-precision recheck of lambda_max = max_Theta ||delta||_F^2 / 2Q <= d_M - 1 for (d,d_M)=(3,3) on the equally
spaced Gibbs curve, with EXACTLY resonant machines (integer potentials) and 60-digit arithmetic.
Also: generalized pi_ext attainment check for d_M != d at uniform p."""
import itertools, sys
import numpy as np, mpmath as mp
mp.mp.dps = 60
rng = np.random.default_rng(5)
d = dm = 3; n = 9
cx = [c // dm for c in range(n)]; cy = [c % dm for c in range(n)]

def potentials(pi):
    """machine-level potentials u (log q = beta*u + comp offset) or None if not resonant on the Gibbs curve"""
    adj = {y: [] for y in range(dm)}
    for c in range(n):
        c2 = pi[c]
        if c2 == c: continue
        x, y, x2, y2 = cx[c], cy[c], cx[c2], cy[c2]
        # p_x q_y = p_x2 q_y2, log p_x = -beta x  ->  u_y - u_y2 = x - x2
        if y == y2:
            if x != x2: return None
            continue
        adj[y].append((y2, x - x2)); adj[y2].append((y, x2 - x))
    u = [None] * dm; comp = [None] * dm; nc = 0
    for s in range(dm):
        if u[s] is not None: continue
        u[s] = 0; comp[s] = nc; stack = [s]
        while stack:
            a = stack.pop()
            for b, w in adj[a]:          # u_a - u_b = w
                if u[b] is None:
                    u[b] = u[a] - w; comp[b] = nc; stack.append(b)
                elif u[a] - u[b] != w:
                    return None
        nc += 1
    return u, comp, nc

def lam_max(pi, p, q):
    """exact-ish (60 digit) max generalized eigenvalue of F wrt A on Theta = B v"""
    inv = [0] * n
    for c, c2 in enumerate(pi): inv[c2] = c
    Bm = [(1, 0), (0, 1), (-1, -1)]
    X = []
    for c in range(n):
        cs = inv[c]
        X.append([p[cx[cs]] * Bm[cy[cs]][j] - p[cx[c]] * Bm[cy[c]][j] for j in range(2)])
    r = [p[cx[c]] * q[cy[c]] for c in range(n)]
    dl = [[sum(X[3 * x + y][j] for y in range(3)) for j in range(2)] for x in range(3)]
    A = [[sum(X[c][i] * X[c][j] / r[c] for c in range(n)) - sum(dl[x][i] * dl[x][j] / p[x] for x in range(3))
          for j in range(2)] for i in range(2)]
    F = [[sum(dl[x][i] * dl[x][j] / p[x] for x in range(3)) for j in range(2)] for i in range(2)]
    scale = max(abs(A[0][0]), abs(A[1][1]), abs(F[0][0]), abs(F[1][1]), mp.mpf(10) ** -200)
    detA = A[0][0] * A[1][1] - A[0][1] ** 2
    if abs(detA) > mp.mpf(10) ** -40 * scale ** 2:
        # det(F - l A) = 0 : detA l^2 - (F00 A11 + F11 A00 - 2 F01 A01) l + detF = 0
        detF = F[0][0] * F[1][1] - F[0][1] ** 2
        bq = F[0][0] * A[1][1] + F[1][1] * A[0][0] - 2 * F[0][1] * A[0][1]
        disc = bq * bq - 4 * detA * detF
        return (bq + mp.sqrt(max(disc, 0))) / (2 * detA)
    tr = A[0][0] + A[1][1]
    if abs(tr) < mp.mpf(10) ** -40 * scale:
        return mp.inf if (F[0][0] + F[1][1]) > mp.mpf(10) ** -40 * scale else mp.mpf(0)
    # rank one: range vector w, null v0
    w = (A[0][0], A[0][1]) if abs(A[0][0]) >= abs(A[1][1]) else (A[0][1], A[1][1])
    v0 = (-w[1], w[0])
    Fv0 = F[0][0] * v0[0] ** 2 + 2 * F[0][1] * v0[0] * v0[1] + F[1][1] * v0[1] ** 2
    if Fv0 > mp.mpf(10) ** -40 * scale * (v0[0] ** 2 + v0[1] ** 2):
        return mp.inf
    Aw = A[0][0] * w[0] ** 2 + 2 * A[0][1] * w[0] * w[1] + A[1][1] * w[1] ** 2
    Fw = F[0][0] * w[0] ** 2 + 2 * F[0][1] * w[0] * w[1] + F[1][1] * w[1] ** 2
    return Fw / Aw

# resonant movers on the Gibbs curve (beta > 0): independent of beta
fam = []
for pi in itertools.permutations(range(n)):
    if all(pi[c] == c for c in range(n)): continue
    res = potentials(pi)
    if res is None: continue
    # does it move the system at all? (some config changes system level)
    if all(cx[pi[c]] == cx[c] for c in range(n)): continue
    fam.append((pi, res))
print('resonant movers on Gibbs curve:', len(fam), ' with families:', sum(1 for _, r in fam if r[2] > 1), flush=True)

betas = [float(b) for b in sys.argv[1:]] or [0.05, 6.0, 12.0, 16.0, 24.0]
for beta in betas:
    x = mp.e ** (-mp.mpf(beta)); Z = 1 + x + x * x
    p = [1 / Z, x / Z, x * x / Z]
    best = mp.mpf(0); arg = None; ninf = 0; minr = mp.mpf(1)
    for pi, (u, comp, nc) in fam:
        offs = [np.zeros(nc)] + ([rng.normal(0, s, nc) for s in (4, 4, 4, 12, 12, 30) for _ in range(5)] if nc > 1 else [])
        for t in offs:
            lq = [mp.mpf(beta) * u[y] + mp.mpf(float(t[comp[y]])) for y in range(dm)]
            mx = max(lq); qv = [mp.e ** (l - mx) for l in lq]; s = sum(qv); q = [v / s for v in qv]
            v = lam_max(pi, p, q)
            minr = min(minr, min(p[cx[c]] * q[cy[c]] for c in range(n)))
            if v == mp.inf: ninf += 1; continue
            if v > best: best, arg = v, (pi, [mp.nstr(qq, 4) for qq in q])
    print(f'beta*omega={beta}: max lambda = {mp.nstr(best, 12)}  best-2 = {mp.nstr(best - 2, 6)}  (bound 2)  infinite={ninf}  min weight seen={mp.nstr(minr, 3)}  '
          f'argmax pi={arg[0] if arg else None} q={arg[1] if arg else None}', flush=True)

# generalized pi_ext attainment at uniform p for d_M != d (float is enough here)
def lam_float(d, dm, pairs):
    n = d * dm; p = np.ones(d) / d; q = np.ones(dm) / dm
    cxx, cyy = np.arange(n) // dm, np.arange(n) % dm
    pi = np.arange(n)
    for a, b in pairs: pi[a], pi[b] = b, a
    Bq = np.zeros((dm, dm - 1)); Bq[:dm - 1] = np.eye(dm - 1); Bq[dm - 1] = -1
    D = p[cxx][:, None] * Bq[cyy]
    Xf = np.empty_like(D); Xf[pi] = D; Xf = Xf - D
    M = np.zeros((d, dm - 1))
    for c in range(n): M[cxx[c]] += Xf[c]
    r = p[cxx] * q[cyy]
    A = Xf.T @ (Xf / r[:, None]) - M.T @ (M / p[:, None]); F = M.T @ (M / p[:, None])
    w, V = np.linalg.eigh(A); k = w > 1e-12
    W = V[:, k] / np.sqrt(w[k]); return np.linalg.eigvalsh(W.T @ F @ W).max()
for d2, dm2 in []:
    pairs = [(0 * dm2 + y, (d2 - 1) * dm2 + y + 1) for y in range(dm2 - 1)]
    print(f'generalized pi_ext (d={d2}, d_M={dm2}): lambda_max = {lam_float(d2, dm2, pairs):.12f}  (d_M-1 = {dm2 - 1})')
