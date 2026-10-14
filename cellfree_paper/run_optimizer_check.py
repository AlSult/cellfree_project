# QUESTION: does the label optimizer get stuck on a small hill (local optimum)?
#   We start the optimizer from DIFFERENT places on the SAME snapshot and see where each run ends.
#   Plain Rayleigh channel (Section III-C), paper-size network, 7 slots per AP (our label format).
#
# Starts:
#   normal      : equal power to the 3 strongest users of every AP (what the label files use now)
#   five        : equal power to the 5 strongest users
#   all seven   : equal power to all 7 slots
#   random 1-3  : random power shares (seeded)
#   normal x3   : the normal start, but with 3 times more steps (is the normal run simply stopping too early?)
#
# How to read the result (printed at the end):
#   gap = how much worse a run is than the best run of that snapshot
#         'sum' : gap in the sum of the speeds, in %        'pf' : gap in the typical (geometric mean) speed, in %
#   Small gaps for all starts  -> the hills are about equally high, the labels are close to the best possible.
#   A start that is clearly better -> we should run several starts and keep the best one as the label.
#
# Nothing in the other files is changed: this script only imports them.
# Saves tables/optimizer_check.csv and img/optimizer_check.png

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_config as cfg
import paper_layout as layout
import paper_gains as gains
import paper_labels as lab
import paper_paths as paths
from paper_channel import full_channel

ap = argparse.ArgumentParser()
ap.add_argument("--snaps", type=int, default=3)
ap.add_argument("--iters", type=int, default=400)
args = ap.parse_args()

L = 7
ap_pos = layout.generate_ap_positions()
ALPHA, SIGMA = cfg.PAPER_LIKE["alpha"], cfg.PAPER_LIKE["shadow_sigma_db"]
STARTS = ["normal", "five", "all seven", "random 1", "random 2", "random 3", "normal x3 steps"]


def make_snapshot(i):
    rng = np.random.default_rng([cfg.SEED_DROPS, i])
    ue = layout.generate_user_drop(rng)
    beta = gains.beta_from_distance(gains.distance_matrix(ap_pos, ue), ALPHA, SIGMA, rng)
    return beta, full_channel(beta, rng)


def start_of(name, M, seed):
    if name in ("normal", "normal x3 steps"):
        return lab.initial_p(M, L, 3)
    if name == "five":
        return lab.initial_p(M, L, 5)
    if name == "all seven":
        return np.full((M, L), 1.0 / L)
    r = np.random.default_rng([seed, int(name.split()[1])]).random((M, L)) + 0.01
    return r / r.sum(axis=1, keepdims=True)


rows = []
curves = {}
t0 = time.time()
for i in range(args.snaps):
    beta, H = make_snapshot(i)
    prob = lab.LabelProblem(H, beta, L)
    for kind in ("sum", "pf"):
        res = {}
        for name in STARTS:
            iters = args.iters * 3 if name == "normal x3 steps" else args.iters
            res[name] = lab.optimize(prob, kind, iters=iters, p_start=start_of(name, prob.M, 100 + i))
        best_name = max(res, key=lambda n: res[n]["objective"])
        best = res[best_name]["objective"]
        p_ref = res["normal"]["p"].ravel()
        for name in STARTS:
            r = prob.rates(res[name]["p"])
            if kind == "sum":
                gap = (best - res[name]["objective"]) / abs(best) * 100
            else:
                gap = (np.exp((best - res[name]["objective"]) / cfg.N_UE) - 1) * 100
            same = float(np.corrcoef(p_ref, res[name]["p"].ravel())[0, 1])
            h = res[name]["history"]
            late = (h.max() - h[int(0.9 * len(h))]) / max(abs(h.max()), 1e-9) * 100     # still improving in the last 10% of steps?
            rows.append([i, kind, name, res[name]["objective"], r.sum(), np.percentile(r, 5), gap, same, late, name == best_name])
            if i == 0:
                curves[(kind, name)] = h
        print(f"snapshot {i + 1}/{args.snaps}, objective '{kind}': best start = '{best_name}'  ({time.time() - t0:.0f} s)", flush=True)

header = ["snapshot", "objective", "start", "objective_value", "sum_Mbps", "p5_Mbps",
          "gap_to_best_percent", "similarity_to_normal_allocation", "still_improving_last10pct_percent", "is_best"]
with open(paths.table("optimizer_check.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows([[f"{v:.5g}" if isinstance(v, (float, np.floating)) else v for v in r] for r in rows])


def col(kind, name, j):
    return [r[j] for r in rows if r[1] == kind and r[2] == name]


print(f"\nMean over {args.snaps} snapshots")
print(f"{'objective':9s} {'start':18s} {'gap to best %':>13s} {'sum Mbps':>9s} {'slowest 5%':>10s} {'similar to normal':>18s} {'still rising %':>14s}")
for kind in ("sum", "pf"):
    for name in STARTS:
        print(f"{kind:9s} {name:18s} {np.mean(col(kind, name, 6)):13.2f} {np.mean(col(kind, name, 4)):9.0f} "
              f"{np.mean(col(kind, name, 5)):10.0f} {np.mean(col(kind, name, 7)):18.3f} {np.mean(col(kind, name, 8)):14.3f}")

print("\nConclusions")
for kind in ("sum", "pf"):
    normal_gap = np.mean(col(kind, "normal", 6))
    worst_gap = max(np.mean(col(kind, n, 6)) for n in STARTS)
    long_gain = np.mean(col(kind, "normal", 6)) - np.mean(col(kind, "normal x3 steps", 6))
    print(f" [{kind}] normal start is {normal_gap:.2f}% below the best run; the worst start is {worst_gap:.2f}% below it")
    print(f" [{kind}] 3 times more steps improve the normal start by {long_gain:.2f} percentage points "
          f"({'steps are enough' if long_gain < 0.5 else 'more steps would help'})")
    print(f" [{kind}] normal start is within 1% of the best run:", "PASS" if normal_gap < 1.0 else "CHECK (use several starts)")

# ---- picture: how the score rises during the steps (snapshot 0) ----
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, kind in zip(axs, ("sum", "pf")):
    for name in STARTS:
        h = curves[(kind, name)]
        ax.plot(np.arange(len(h)) + 1, h, label=name, lw=1.4)
    lo = min(v[len(v) // 5:].min() for (k, _), v in curves.items() if k == kind)     # skip the very first steps for a readable scale
    hi = max(v.max() for (k, _), v in curves.items() if k == kind)
    ax.set_ylim(lo - 0.02 * abs(hi), hi + 0.02 * abs(hi))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("score")
    ax.set_title(f"objective '{kind}'", fontsize=10)
    ax.grid(alpha=0.3)
axs[0].legend(fontsize=8)
fig.suptitle("Do different starting points end on the same hill?  (snapshot 0, 7 slots, plain Rayleigh)", fontsize=10)
plt.tight_layout()
plt.savefig(paths.image("optimizer_check.png"), dpi=110)
print("\nSaved tables/optimizer_check.csv and img/optimizer_check.png")