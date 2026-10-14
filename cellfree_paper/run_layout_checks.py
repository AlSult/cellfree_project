# QUESTION: do our conclusions depend on ONE lucky placement of the 121 APs?
#   We repeat the same experiment on 6 different AP layouts and compare.
#   The users and their channels are the SAME in every layout (same seeds), only the APs move,
#   so any difference comes from the layout alone.
#
# Layouts:  original (seed 1, jitter 10 m) | seeds 2, 3, 4 (jitter 10 m) | perfect grid (jitter 0 m) | big jitter (20 m)
# Cases per layout (plain Rayleigh, exponent 3.8, 8 dB shadowing):
#   equal power, clusters L=3  |  equal power, everyone serves everyone  |  optimized 7 slots ('pf')  |  optimized 7 slots ('sum')
#   |  optimized 3 slots ('sum')  (3 slots = only the 3 strongest users may be served, to see what the serving rule is worth)
#
# Nothing in the other files is changed: this script only imports them.
# Saves tables/layout_check.csv and img/layout_check.png

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
import paper_serving as serving
import paper_labels as lab
import paper_paths as paths
from paper_channel import full_channel
from paper_precoding import mrt_precoders
from paper_rates import sinr_and_rate

ap = argparse.ArgumentParser()
ap.add_argument("--snaps", type=int, default=3)
ap.add_argument("--iters", type=int, default=300)
args = ap.parse_args()

ALPHA, SIGMA = cfg.PAPER_LIKE["alpha"], cfg.PAPER_LIKE["shadow_sigma_db"]
ORIGINAL_JITTER = cfg.AP_JITTER_M

LAYOUTS = [("original (seed 1, 10 m)", 1, 10.0),
           ("seed 2, 10 m", 2, 10.0),
           ("seed 3, 10 m", 3, 10.0),
           ("seed 4, 10 m", 4, 10.0),
           ("perfect grid (0 m)", 1, 0.0),
           ("big jitter (20 m)", 1, 20.0)]
CASES = ["equal, clusters L=3", "equal, everyone serves everyone", "optimized 7 slots (pf)", "optimized 7 slots (sum)",
         "optimized 3 slots (sum)"]


def ap_positions(seed, jitter):
    cfg.AP_JITTER_M = jitter                       # paper_layout reads this value when it is called
    try:
        return layout.generate_ap_positions(seed)
    finally:
        cfg.AP_JITTER_M = ORIGINAL_JITTER


def jain(r):
    return float(r.sum() ** 2 / (len(r) * (r ** 2).sum()))


def one_snapshot(ap_pos, i):
    rng = np.random.default_rng([cfg.SEED_DROPS, i])        # same users and same random numbers for every layout
    ue = layout.generate_user_drop(rng)
    beta = gains.beta_from_distance(gains.distance_matrix(ap_pos, ue), ALPHA, SIGMA, rng)
    H = full_channel(beta, rng)
    out = {}
    for name, mask in ((CASES[0], serving.cluster_mask(beta, 3)), (CASES[1], serving.everyone_mask(beta))):
        out[name] = sinr_and_rate(H, mrt_precoders(H, serving.equal_power_per_ap(mask)))[3]
    prob = lab.LabelProblem(H, beta, 7)
    for name, kind in ((CASES[2], "pf"), (CASES[3], "sum")):
        res = lab.optimize(prob, kind, iters=args.iters, p_start=lab.initial_p(prob.M, 7, 3))
        out[name] = prob.rates(res["p"])
        assert res["p"].sum(axis=1).max() <= 1 + 1e-9, "an AP uses more than its power budget"
    prob3 = lab.LabelProblem(H, beta, 3)
    res = lab.optimize(prob3, "sum", iters=args.iters, p_start=lab.initial_p(prob3.M, 3, 3))
    out[CASES[4]] = prob3.rates(res["p"])
    return out


rows = []
t0 = time.time()
for lname, seed, jit in LAYOUTS:
    pos = ap_positions(seed, jit)
    acc = {c: {"mean": [], "p5": [], "sum": [], "jain": []} for c in CASES}
    for i in range(args.snaps):
        res = one_snapshot(pos, i)
        for c in CASES:
            r = res[c]
            acc[c]["mean"].append(r.mean()); acc[c]["p5"].append(np.percentile(r, 5))
            acc[c]["sum"].append(r.sum()); acc[c]["jain"].append(jain(r))
    for c in CASES:
        rows.append([lname, c] + [float(np.mean(acc[c][k])) for k in ("mean", "p5", "sum", "jain")])
    print(f"layout '{lname}' done  ({time.time() - t0:.0f} s)", flush=True)

with open(paths.table("layout_check.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["layout", "case", "mean_Mbps", "p5_Mbps", "sum_Mbps", "jain"])
    w.writerows([[r[0], r[1]] + [f"{v:.5g}" for v in r[2:]] for r in rows])


def get(lname, case, col):                      # col: 2 mean, 3 p5, 4 sum, 5 jain
    return next(r[col] for r in rows if r[0] == lname and r[1] == case)


print(f"\nMean over {args.snaps} snapshots (speeds in Mbps)")
print(f"{'layout':26s} {'case':34s} {'mean':>7s} {'p5':>7s} {'sum':>9s} {'jain':>6s}")
for r in rows:
    print(f"{r[0]:26s} {r[1]:34s} {r[2]:7.0f} {r[3]:7.0f} {r[4]:9.0f} {r[5]:6.3f}")

# ---- the conclusions we want to test ----
print("\nConclusions tested on every layout")
ok_power, ok_pf = [], []
print(f"{'layout':26s} {'optimizing power':>17s} {'3 -> 7 slots':>13s}   (change of the SUM of speeds)")
print(f"{'':26s} {'(vs equal, L=3)':>17s} {'(opt 3 vs opt 7)':>13s}")
for lname, _, _ in LAYOUTS:
    base = get(lname, CASES[0], 4)
    g_opt = get(lname, CASES[3], 4) / base - 1               # optimizing the power, 7 slots, vs equal power
    g_srv = get(lname, CASES[3], 4) / get(lname, CASES[4], 4) - 1   # allowing 7 instead of 3 users per AP, power optimized
    print(f"{lname:26s} {g_opt * 100:+16.0f}% {g_srv * 100:+12.0f}%")
    ok_power.append(abs(g_opt) > abs(g_srv))
    ok_pf.append(get(lname, CASES[2], 3) > get(lname, CASES[3], 3))          # pf lifts the slowest 5% more than sum
print(" optimizing the power matters more than the serving rule (3 vs 7 slots), on every layout:", "PASS" if all(ok_power) else "FAIL on some layout")
print(" 'pf' gives the slowest 5% more speed than 'sum', on every layout   :", "PASS" if all(ok_pf) else "FAIL on some layout")

sums = [get(l, CASES[2], 4) for l, _, _ in LAYOUTS]
spread = (max(sums) - min(sums)) / np.mean(sums) * 100
print(f" sum of speeds ('pf' labels) differs between layouts by {spread:.0f}% (largest minus smallest, relative to the mean)")

# ---- picture ----
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
x = np.arange(len(LAYOUTS))
w = 0.16
for ax, col, title in ((axs[0], 4, "Sum of all speeds (Mbps)"), (axs[1], 3, "Slowest 5% of users (Mbps)")):
    for j, c in enumerate(CASES):
        ax.bar(x + (j - 2) * w, [get(l, c, col) for l, _, _ in LAYOUTS], w, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (", "\n(") for l, _, _ in LAYOUTS], fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
axs[0].legend(fontsize=7)
fig.suptitle(f"Do the results depend on one lucky AP layout?  ({args.snaps} snapshots per layout, same users in every layout)", fontsize=10)
plt.tight_layout()
plt.savefig(paths.image("layout_check.png"), dpi=110)
print("\nSaved tables/layout_check.csv and img/layout_check.png")