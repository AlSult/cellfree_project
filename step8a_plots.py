# Two figures of the middle network:
#   step8a_map.png      one drop on the map: who serves whom under two rules
#   step8a_results.png  six pictures with the results of the tables

import sys
import numpy as np
import matplotlib.pyplot as plt

import config
import mid_config as cfg
import mid_layout
import mid_serving
import mid_simulate as sim
from rate_calculation import NOISE_W

ap_pos = mid_layout.generate_ap_positions()
RULES = sim.standard_rules()
NAMES = list(RULES)
COLOURS = {"nearest AP only": "tab:gray", f"clusters (L={cfg.L_PER_AP})": "tab:blue", "everyone serves everyone": "tab:red"}

# =====================================================================
# FIGURE 1: one drop on the map
# =====================================================================
drop = sim.simulate_drop(ap_pos, np.random.default_rng(3), RULES)
ue = drop["ue"]
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
for a, rname, title in [(ax[0], NAMES[0], "Conventional rule: every user is served by its nearest AP only"),
                        (ax[1], NAMES[1], f"Cell-free with clusters: every AP serves its {cfg.L_PER_AP} strongest users")]:
    mask = drop["rules"][rname]["mask"]
    for m, k in zip(*np.where(mask)):
        a.plot([ap_pos[m, 0], ue[k, 0]], [ap_pos[m, 1], ue[k, 1]], "-", color="tab:blue" if "cluster" in rname else "gray",
               lw=1.2, alpha=0.6, zorder=1)
    served = mask.sum(axis=1)
    for m in range(cfg.N_AP):
        a.scatter(*ap_pos[m], marker="^", s=170, zorder=3, color="black" if served[m] > 0 else "white", edgecolor="black")
        a.text(ap_pos[m, 0] + 8, ap_pos[m, 1] - 18, str(served[m]), fontsize=9, color="black" if served[m] > 0 else "red")
    a.scatter(ue[:, 0], ue[:, 1], s=60, color="tab:orange", edgecolor="black", zorder=4)
    for k in range(cfg.N_UE):
        a.text(ue[k, 0] + 6, ue[k, 1] + 6, str(k), fontsize=8, zorder=5)
    a.set_xlim(0, cfg.AREA_M)
    a.set_ylim(0, cfg.AREA_M)
    a.set_aspect("equal")
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.set_title(f"{title}\n{int(mask.sum())} beams; {int((served == 0).sum())} APs are silent (white triangles)", fontsize=10)
    a.grid(True, alpha=0.2)
fig.text(0.5, 0.01, "triangle = AP (the number is how many users it serves), orange dot = user, line = a beam from an AP to a user",
         ha="center", fontsize=10)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig("img/step8a_map.png", dpi=120)

# =====================================================================
# FIGURE 2: results
# =====================================================================
SETTINGS = {"free space": cfg.FREE_SPACE, "paper-like": cfg.PAPER_LIKE}
res = {s: sim.simulate(ap_pos, RULES, **kw) for s, kw in SETTINGS.items()}
fig, ax = plt.subplots(2, 3, figsize=(18, 10.5))

# A, B: distribution of the speeds
for a, s, letter in [(ax[0, 0], "free space", "A"), (ax[0, 1], "paper-like", "B")]:
    for rname in NAMES:
        r = np.sort(res[s]["rules"][rname]["rate"].ravel())
        a.plot(r, np.arange(1, r.size + 1) / r.size * 100, color=COLOURS[rname], lw=2.5, label=rname)
    a.set_xlim(0, 1400)
    a.set_xlabel("speed of a user (Mbps)")
    a.set_ylabel("share of users at or below this speed (%)")
    a.set_title(f"{letter}. Speeds, {s} propagation\n(a curve to the right is faster; a steep curve is more even)")
    a.grid(True, alpha=0.3)
    a.legend(loc="lower right", fontsize=8)

# C: how many users per AP
L_VALUES = [1, 2, 3, 4, 6, 10, 20]
sweep = {s: {"mean": [], "p5": []} for s in SETTINGS}
for s, kw in SETTINGS.items():
    for L in L_VALUES:
        o = sim.simulate(ap_pos, {"c": (lambda beta, L=L: mid_serving.cluster_mask(beta, L))}, **kw)["rules"]["c"]["rate"]
        sweep[s]["mean"].append(o.mean())
        sweep[s]["p5"].append(np.percentile(o, 5))
for s, ls in [("free space", "-"), ("paper-like", "--")]:
    ax[0, 2].plot(L_VALUES, sweep[s]["mean"], ls, marker="o", color="tab:blue", label=f"average speed, {s}")
    ax[0, 2].plot(L_VALUES, sweep[s]["p5"], ls, marker="s", color="tab:orange", label=f"slowest 5% (p5), {s}")
ax[0, 2].axvline(cfg.L_PER_AP, color="gray", ls=":", label=f"our choice L = {cfg.L_PER_AP}")
ax[0, 2].set_xscale("log")
ax[0, 2].set_xticks(L_VALUES)
ax[0, 2].set_xticklabels([str(v) for v in L_VALUES])
ax[0, 2].set_xlabel("number of users each AP serves (L); L = 20 is everyone serves everyone")
ax[0, 2].set_ylabel("speed (Mbps)")
ax[0, 2].set_title("C. How many users should an AP serve?\nThe slowest users do best at L = 3 to 4")
ax[0, 2].grid(True, alpha=0.3)
ax[0, 2].legend(fontsize=8)

# D: speed against distance to the nearest AP
edges = np.arange(0, 90, 10)
centres = (edges[:-1] + edges[1:]) / 2
dist = res["free space"]["dist_nearest"].ravel()
for rname in NAMES:
    r = res["free space"]["rules"][rname]["rate"].ravel()
    means = [r[(dist >= lo) & (dist < hi)].mean() if np.any((dist >= lo) & (dist < hi)) else np.nan
             for lo, hi in zip(edges[:-1], edges[1:])]
    ax[1, 0].plot(centres, means, "o-", color=COLOURS[rname], lw=2.5, label=rname)
ax[1, 0].set_xlabel("distance from the user to its nearest AP (m)")
ax[1, 0].set_ylabel("average speed (Mbps)")
ax[1, 0].set_title("D. Speed against distance (free space)\nthe conventional rule falls with distance, clusters stay level")
ax[1, 0].grid(True, alpha=0.3)
ax[1, 0].legend(fontsize=8)

# E: load of the APs
upa = {rname: res["free space"]["rules"][rname]["users_per_ap"].ravel() for rname in NAMES[:2]}
bins = np.arange(0, 6)
width = 0.38
for i, rname in enumerate(NAMES[:2]):
    share = [100 * np.mean(upa[rname] == b) for b in bins]
    ax[1, 1].bar(bins + (i - 0.5) * width, share, width, color=COLOURS[rname], label=rname)
    for b, v in zip(bins, share):
        if v > 1:
            ax[1, 1].text(b + (i - 0.5) * width, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
ax[1, 1].set_xlabel("number of users an AP serves")
ax[1, 1].set_ylabel("share of the APs (%)")
ax[1, 1].set_title("E. How busy are the APs?\n(conventional: many idle; clusters: all serve 3)", fontsize=11)
ax[1, 1].legend(fontsize=8)

# F: what limits the speed
labels, real_v, nonoise_v, nointf_v = [], [], [], []
for s in SETTINGS:
    for rname in NAMES[:2]:
        o = res[s]["rules"][rname]
        labels.append(f"{s}\n{rname.split(' (')[0]}")
        real_v.append(o["rate"].mean())
        nonoise_v.append((config.B_HZ * np.log2(1 + o["signal"] / np.maximum(o["interference"], 1e-30)) / 1e6).mean())
        nointf_v.append((config.B_HZ * np.log2(1 + o["signal"] / NOISE_W) / 1e6).mean())
x = np.arange(len(labels))
ax[1, 2].bar(x - 0.27, real_v, 0.27, color="tab:blue", label="real")
ax[1, 2].bar(x, nonoise_v, 0.27, color="tab:green", label="if there were no noise")
ax[1, 2].bar(x + 0.27, nointf_v, 0.27, color="tab:gray", label="if there were no interference")
ax[1, 2].set_xticks(x)
ax[1, 2].set_xticklabels(labels, fontsize=8)
ax[1, 2].set_ylabel("average speed (Mbps)")
ax[1, 2].set_title("F. What limits the speed?\n(no noise: no change; no interference: huge change)", fontsize=11)
ax[1, 2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("img/step8a_results.png", dpi=110)
print("Saved step8a_map.png and step8a_results.png")
if "--no-show" not in sys.argv:
    plt.show()