"""Exact finite-N cooling protocols for a qutrit with qutrit machines (classical permutation unitaries).

Each step n: machine distribution q_n (free: thermal state of a freely chosen Hamiltonian), permutation pi_n of the
9 configurations (x, y).  Exact dynamics r = p x q -> r' = r o pi^{-1}; system p' = marginal.
Per-step dissipation Sigma_n = I(S:M_n) + D(q_n'||q_n) = D(r'||r) - D(p'||p)   (exact).
Total machine energy: beta*DeltaE_M = (S(p_0) - S(p_N)) + sum_n Sigma_n.   Target p_N = Gibbs(beta*lam) exactly.
usage: c7_protocol.py beta lam N pattern       pattern in {swap, alt, alt_swap}
"""
import sys, itertools
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

d = 3
E = torch.tensor([0.0, 1.0, 2.0])
SWAP = [3 * (c % 3) + c // 3 for c in range(9)]
PI576 = [5, 0, 2, 8, 3, 1, 6, 7, 4]
PI711 = [8, 5, 2, 7, 0, 1, 6, 3, 4]
PI679 = [8, 0, 2, 3, 4, 5, 6, 7, 1]
PI636 = [5, 3, 2, 8, 4, 0, 6, 7, 1]
PI713 = [8, 6, 2, 3, 4, 5, 1, 7, 0]


LETTERS = {'S': [3 * (c % 3) + c // 3 for c in range(9)], 'A': [8, 6, 2, 3, 4, 5, 1, 7, 0], 'B': [5, 3, 2, 8, 4, 0, 6, 7, 1],
           'C': [5, 0, 2, 8, 3, 1, 6, 7, 4], 'D': [8, 5, 2, 7, 0, 1, 6, 3, 4], 'F': [8, 0, 2, 3, 4, 5, 6, 7, 1]}
def seq_pattern(pattern, N):
    word = pattern.split(':', 1)[1]
    return [LETTERS[word[n % len(word)]] for n in range(N)]

def gibbs(b):
    v = torch.exp(-b * E); return v / v.sum()

def run(logits, pis, p0):
    p = p0
    total = torch.zeros(())
    for n, pi in enumerate(pis):
        q = torch.softmax(logits[n], 0)
        r = torch.outer(p, q).reshape(-1)
        r2 = torch.empty_like(r)
        r2 = r2.index_copy(0, torch.tensor(pi), r)       # r2[pi[c]] = r[c]
        p2 = r2.reshape(d, d).sum(1)
        sig = torch.sum(r2 * torch.log(r2 / r)) - torch.sum(p2 * torch.log(p2 / p))
        total = total + sig
        p = p2
    return total, p

def resonant_logits(p, pi):
    rows, rhs = [], []
    for c in range(9):
        c2 = pi[c]
        if c2 == c: continue
        x, y = divmod(c, 3); x2, y2 = divmod(c2, 3)
        row = np.zeros(3); row[y] += 1; row[y2] -= 1
        rows.append(row); rhs.append(np.log(p[x2]) - np.log(p[x]))
    u, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return u - u.mean()

def optimise(beta, lam, N, pattern, seed=0, verbose=False):
    if pattern.startswith('seq:'):
        pis = seq_pattern(pattern, N)
    elif pattern == 'swap':
        pis = [SWAP] * N
    elif pattern == 'alt':
        pis = [PI576 if n % 2 == 0 else PI711 for n in range(N)]
    elif pattern == 'alt_swap':
        pis = [PI576 if n % 2 == 0 else PI711 for n in range(N - 2)] + [SWAP, SWAP]
    elif pattern == 'alt2':
        pis = [PI576 if n % 2 == 0 else PI679 for n in range(N)]
    elif pattern == 'hot':
        pis = [PI636 if n % 2 == 0 else PI713 for n in range(N)]
    elif pattern == 'p713':
        pis = [PI713] * (N - 1) + [SWAP]
    elif pattern == 'p713b':
        pis = [PI713 if n % 4 else SWAP for n in range(N)]
    elif pattern == 'p713c':
        pis = [SWAP if n % 10 == 9 else PI713 for n in range(N)]
    elif pattern == 'alt3':
        pis = [[PI576, PI679, PI711][n % 3] for n in range(N)]
    else:
        raise ValueError(pattern)
    p0 = gibbs(beta); pT = gibbs(beta * lam)
    # initial guess: target states along the Gibbs curve, resonant machines for the planned point
    bs = np.linspace(beta, beta * lam, N + 1)
    init = []
    for n in range(N):
        pn = gibbs(float(bs[n])).numpy()
        if pis[n] == SWAP:
            init.append(np.log(gibbs(float(bs[n + 1])).numpy()))
        else:
            init.append(resonant_logits(pn, pis[n]))
    logits = torch.tensor(np.array(init), requires_grad=True)
    best = None
    for mu in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
        opt = torch.optim.LBFGS([logits], lr=1, max_iter=3000, tolerance_grad=1e-14, tolerance_change=1e-16,
                                history_size=50, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            tot, pN = run(logits, pis, p0)
            loss = tot + mu * torch.sum((pN - pT) ** 2)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            tot, pN = run(logits, pis, p0)
        if verbose:
            print(f'  mu={mu:.0e}  Sigma_total={tot.item():.8f}  final err={torch.max(torch.abs(pN - pT)).item():.2e}', flush=True)
    return tot.item(), torch.max(torch.abs(pN - pT)).item(), logits.detach().numpy()

if __name__ == '__main__':
    beta, lam, N = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    pattern = sys.argv[4]
    p0 = gibbs(beta).numpy(); pT = gibbs(beta * lam).numpy()
    Lstar = 2 * np.arccos(np.sqrt(p0) @ np.sqrt(pT))
    tot, err, lg = optimise(beta, lam, N, pattern, verbose=True)
    print(f'RESULT pattern={pattern} beta={beta} lam={lam} N={N}: Sigma_total={tot:.8f}  N*Sigma={N*tot:.6f}  '
          f'(L*^2/2 = {Lstar**2/2:.6f})  final err={err:.1e}')
    np.save(f"logits_{pattern.replace(':', '')}_b{beta}_l{lam}_N{N}.npy", lg)
