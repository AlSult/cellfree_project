# LABELS: what is a "good" resource allocation for one snapshot of the network?
#
# The paper says its network outputs 947 = 121 x 7 + 100 numbers: seven per AP and a softmax over the 100 users
# ("scheduling"), and that the labels are "optimal resource allocation decisions". It does not say how they were computed,
# and its split of the 7 numbers into magnitude / phase / power cannot describe the precoder of a 64-antenna AP.
# OUR READING (a choice, written down in run_labels.py):
#   slot s of AP m   = the AP's s-th strongest user (strongest first). The AP knows who these are from its own measurements.
#   p[m, s]          = the SHARE of the AP's power given to that user's beam (0 = beam off). Sum over s of p[m, s] <= 1.
#   s_k (scheduling) = the share of the whole network's transmit power that is spent on user k (sums to 1).
#                      It is NOT 'who is served': a user close to an AP needs little power and is still served.
#   output vector    = [ p[0, 0..6], p[1, 0..6], ..., p[120, 0..6], s_0 .. s_99 ]   (847 + 100 = 947 numbers)
# The beams are aimed with MRT as before, so the only decision is the power of every beam.
#
# The best allocation is searched with the exact gradient of the objective (derived in gradient()):
#   "sum": maximize the sum of the speeds of all users  (Eq. 20a of the paper, with equal priority weights)
#          minus a penalty for users below the minimum rate of Table I (1 Mbps)
#   "pf" : maximize the sum of log(speed): proportional fairness (a fairer variant, for comparison)
# The optimizer is Adam on the beam amplitudes, projected after every step onto the constraints (amplitudes >= 0 and
# sum of the powers of an AP <= its budget). The result is a LOCAL optimum (the problem is not convex).
# SCALE CONVENTION: the network is limited by interference, so multiplying the power of EVERY beam of EVERY AP by the same
# factor leaves the speeds (almost) unchanged. We therefore scale every allocation so that the busiest AP uses its whole
# budget (canonical_scale). This never lowers a speed (the noise becomes relatively smaller).

import numpy as np

import paper_config as cfg
import paper_serving as serving
from paper_rates import NOISE_W

L_MAX = 7                                                    # slots per AP (the paper's 7 numbers per AP)
BUDGET_W = 10 ** ((cfg.PTX_DBM - 30) / 10)                   # power of one AP: 46 dBm in Watts
RATE_MIN_MBPS = 1.0                                          # "Min Rate (Basic)" of Table I
KAPPA = 1000.0                                               # strength of the penalty for rates below the minimum
RATE_SCALE = cfg.B_HZ / 1e6 / np.log(2.0)                    # Mbps per nat


def candidate_users(beta, L=L_MAX):
    """(M, L): the user in every slot of every AP = the AP's L strongest users (every user ends up with at least one AP,
    by the same coverage fix as the cluster rule), strongest first."""
    mask = serving.cluster_mask(beta, L)
    cand = np.zeros((beta.shape[0], L), dtype=int)
    for m in range(beta.shape[0]):
        users = np.where(mask[m])[0]
        cand[m] = users[np.argsort(-beta[m, users])]
    return cand


class LabelProblem:
    """Everything about ONE snapshot that the allocation needs, computed once."""

    def __init__(self, H, beta, L=L_MAX):
        M, K, N = H.shape
        self.H, self.M, self.K, self.N, self.L = H, M, K, N, L
        self.cand = candidate_users(beta, L)                                       # (M, L)
        slot_channels = H[np.arange(M)[:, None], self.cand]                          # (M, L, N)
        self.U = slot_channels / np.linalg.norm(slot_channels, axis=2, keepdims=True)    # unit beam directions (MRT)
        # G[m, s, k] = what user k receives from AP m's beam in slot s, per unit amplitude
        self.G = np.matmul(self.U, H.conj().transpose(0, 2, 1))
        self.G2 = self.G.reshape(M * L, K)
        self.A = np.zeros((K, M * L))
        self.A[self.cand.ravel(), np.arange(M * L)] = 1.0                            # slot -> the user it serves

    def amplitudes(self, p):
        return np.sqrt(BUDGET_W * np.maximum(p, 0.0))

    def S(self, q):
        """S[j, k] = total signal user k receives from all beams meant for user j."""
        return self.A @ (q.reshape(-1, 1) * self.G2)

    def rates_from_S(self, S):
        a = np.abs(np.diag(S)) ** 2
        T = (np.abs(S) ** 2).sum(axis=0)
        b = T - a + NOISE_W
        return RATE_SCALE * np.log1p(a / b), a, T, b

    def rates(self, p):
        return self.rates_from_S(self.S(self.amplitudes(p)))[0]

    def gradient(self, q, S, a, T, b, weights):
        """d( sum_k weights_k R_k ) / d q[m, s], exact.
        With a_k = |S_kk|^2, T_k = sum_j |S_jk|^2, b_k = T_k - a_k + noise:   ln(1 + SINR_k) = ln(T_k + noise) - ln(b_k)."""
        c = 1.0 / (T + NOISE_W) - 1.0 / b                                            # (K,)
        dT = 2.0 * np.real(np.conj(S[self.cand]) * self.G)                           # (M, L, K): d|S_jk|^2 / dq_ms
        own = np.take_along_axis(dT, self.cand[:, :, None], axis=2)[:, :, 0]         # k = the slot's own user
        return RATE_SCALE * ((dT * (weights * c)).sum(axis=2) + weights[self.cand] / b[self.cand] * own)


def objective_and_weights(R, kind):
    """The objective value and the weights that turn its gradient into 'sum of weights x gradient of the rates'."""
    short = np.maximum(RATE_MIN_MBPS - R, 0.0)
    if kind == "sum":
        return R.sum() - KAPPA * (short ** 2).sum(), 1.0 + 2.0 * KAPPA * short
    if kind == "pf":
        return np.log(np.maximum(R, 1e-9)).sum(), 1.0 / np.maximum(R, 1e-9)
    raise ValueError(kind)


def initial_p(M, L, n_active=3, other=1e-2):
    """Start: equal power on the AP's strongest n_active users (our cluster rule), a little on the other slots."""
    p = np.full((M, L), other)
    p[:, :n_active] = 1.0
    return p / p.sum(axis=1, keepdims=True)


def project(q):
    """Back onto the constraints: amplitudes >= 0 and, for every AP, the sum of the squared amplitudes <= the budget."""
    q = np.maximum(q, 0.0)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q * np.minimum(1.0, np.sqrt(BUDGET_W) / np.maximum(norm, 1e-30))


def canonical_scale(p):
    """Scale an allocation so that the busiest AP uses its whole budget (see the scale convention above)."""
    return p / max(p.sum(axis=1).max(), 1e-30)


def optimize(problem, kind="sum", n_active=3, iters=400, lr=0.3, lr_end=0.01, p_start=None):
    """Adam on the beam amplitudes with projection. Returns the best allocation found, scaled canonically:
    p (M, L) = shares of every AP's power (sum <= 1), the objective value, and the history of the objective."""
    M, L = problem.M, problem.L
    p0 = initial_p(M, L, n_active) if p_start is None else p_start
    q = problem.amplitudes(p0)
    m1, m2 = np.zeros_like(q), np.zeros_like(q)
    best_F, best_q, history = -np.inf, None, []
    for it in range(iters):
        S = problem.S(q)
        R, a, T, b = problem.rates_from_S(S)
        F, w = objective_and_weights(R, kind)
        history.append(F)
        if F > best_F:
            best_F, best_q = F, q.copy()
        g = problem.gradient(q, S, a, T, b, w)
        step = lr_end + 0.5 * (lr - lr_end) * (1 + np.cos(np.pi * it / max(iters - 1, 1)))
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g ** 2
        q = project(q + step * (m1 / (1 - 0.9 ** (it + 1))) / (np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-12))
    p = canonical_scale(best_q ** 2 / BUDGET_W)
    F_final, _ = objective_and_weights(problem.rates(p), kind)
    return {"p": p, "objective": F_final, "history": np.array(history)}


def label_vector(problem, p):
    """The 947 numbers: p of every slot (121 x 7) followed by the scheduling shares of the 100 users."""
    share = np.zeros(problem.K)
    np.add.at(share, problem.cand.ravel(), p.ravel())
    return np.concatenate([p.ravel(), share / max(share.sum(), 1e-30)])


def beam_weights(problem, p):
    """The full precoding array W (M, K, N) of an allocation, for the independent check against the Step 5 library."""
    W = np.zeros((problem.M, problem.K, problem.N), dtype=complex)
    q = problem.amplitudes(p)
    for m in range(problem.M):
        for s in range(problem.L):
            W[m, problem.cand[m, s]] = q[m, s] * problem.U[m, s]
    return W


def equal_power(problem, n_active):
    """The baseline: equal split over the AP's n_active strongest users (the cluster rule with L = n_active)."""
    p = np.zeros((problem.M, problem.L))
    p[:, :n_active] = 1.0 / n_active
    return p