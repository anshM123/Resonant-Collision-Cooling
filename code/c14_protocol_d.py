"""Exact finite-N cooling protocols for an equally spaced d-level system with d-level machines.
usage: c14_protocol_d.py d beta lam N mechfile pattern
  mechfile: json list of {'e': machine spectrum, 'pi': permutation of d*d configurations}  (or 'swap')
  pattern : string of mechanism indices cycled over the steps, e.g. '0101' or '001'; 'S' denotes a swap step.
Prints Sigma_total (exact), N*Sigma, and the swap convex optimum for comparison.
"""
import sys, json
import numpy as np
import torch
import cvxpy as cp
torch.set_default_dtype(torch.float64)

d, beta, lam, N = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
mechfile, pattern = sys.argv[5], sys.argv[6]
E = torch.arange(d, dtype=torch.float64)
if len(sys.argv) > 7:
    E = torch.tensor([float(v) for v in sys.argv[7].split(',')], dtype=torch.float64)   # custom system spectrum
SWAP = [(c % d) * d + c // d for c in range(d * d)]
mechs = json.load(open(mechfile)) if mechfile != 'swap' else []
steps = []
for n in range(N):
    ch = pattern[n % len(pattern)]
    steps.append(('S', SWAP, None) if ch == 'S' else ('M', mechs[int(ch)]['pi'], mechs[int(ch)]['e']))

def gibbs(b, en=None):
    en = E if en is None else torch.tensor(en, dtype=torch.float64)
    v = torch.exp(-b * en); return v / v.sum()

def run(logits, p0):
    p, tot = p0, torch.zeros(())
    for n, (kind, pi, e) in enumerate(steps):
        q = torch.softmax(logits[n], 0)
        r = torch.outer(p, q).reshape(-1)
        r2 = torch.empty_like(r).index_copy(0, torch.tensor(pi), r)
        p2 = r2.reshape(d, d).sum(1)
        tot = tot + torch.sum(r2 * torch.log(r2 / r)) - torch.sum(p2 * torch.log(p2 / p))
        p = p2
    return tot, p

p0, pT = gibbs(beta), gibbs(beta * lam)
bs = np.linspace(beta, beta * lam, N + 1)
init = []
for n, (kind, pi, e) in enumerate(steps):
    if kind == 'S':
        init.append(torch.log(gibbs(float(bs[n + 1]))).numpy())
    else:
        if len(sys.argv) > 7:
            # solve this permutation's resonance equations at the planned point (works for any mechanism)
            pg = gibbs(float(bs[n])).numpy(); rows, rhs = [], []
            for c in range(d * d):
                c2 = pi[c]
                if c2 == c: continue
                x1, y1 = divmod(c, d); x2, y2 = divmod(c2, d)
                if y1 == y2: continue
                row = np.zeros(d); row[y1] += 1; row[y2] -= 1
                rows.append(row); rhs.append(np.log(pg[x2]) - np.log(pg[x1]))
            uu, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
            init.append(uu - uu.max())
        else:
            init.append(torch.log(gibbs(float(bs[n]), e)).numpy())
logits = torch.tensor(np.array(init), requires_grad=True)
scale = float(torch.sum(pT * (E - (pT * E).sum()) ** 2))        # keep the penalty meaningful at low T
for mu in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
    opt = torch.optim.LBFGS([logits], lr=1, max_iter=4000, tolerance_grad=1e-15, tolerance_change=1e-17,
                            history_size=60, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        tot, pN = run(logits, p0)
        loss = tot + mu * torch.sum((pN - pT) ** 2 / pT) * 1.0
        loss.backward(); return loss
    opt.step(closure)
with torch.no_grad():
    tot, pN = run(logits, p0)
err = float(torch.max(torch.abs(pN - pT) / pT))
# exact completion by one swap: cost D(p_N || p_T)
comp = float(torch.sum(pN * torch.log(pN / pT)))
# swap optimum (convex) with N+1 machines
P = cp.Variable((N + 2, d))
cons = [P[0] == p0.numpy(), P[N + 1] == pT.numpy(), P >= 1e-300, cp.sum(P, axis=1) == 1]
prob = cp.Problem(cp.Minimize(sum(cp.sum(cp.rel_entr(P[n], P[n + 1])) for n in range(N + 1))), cons)
try:
    prob.solve(solver=cp.CLARABEL); sw = prob.value; st = prob.status
except Exception as ex:
    sw, st = float('nan'), 'failed'
L = 2 * np.arccos(float(torch.sum(torch.sqrt(p0) * torch.sqrt(pT))))
print(f'd={d} beta={beta} lam={lam} N={N} pattern={pattern}: Sigma={tot.item():.10e} (+completion {comp:.2e}, '
      f'max rel err {err:.1e}) => {N+1} machines: {tot.item()+comp:.10e};  swap optimum with {N+1} machines: '
      f'{sw:.10e} [{st}];  ratio {(tot.item()+comp)/sw:.4f};  (N+1)*swap/(L*^2/2) = {(N+1)*sw/(L*L/2):.4f}')
np.save(f'logits_d{d}_b{beta}_l{lam}_N{N}_{pattern}' + ('_E' + sys.argv[7].replace(',', '_') if len(sys.argv) > 7 else '') + '.npy', logits.detach().numpy())
