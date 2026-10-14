# Two figures of the paper-size network:
#   img/map.png      one drop on the map: who serves whom under two rules
#   img/results.png  six pictures with the results of the tables

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the files of THIS folder always win
import numpy as np
import matplotlib.pyplot as plt

import paper_config as cfg
import paper_layout as layout
import paper_serving as serving
import paper_simulate as sim
import paper_paths as paths

ap_pos = layout.generate_ap_positions()
RULES = sim.standard_rules()
NAMES = list(RULES)
COLOURS = {"nearest AP only": "tab:gray", f"clusters (L={cfg.L_PER_AP})": "tab:blue", "everyone serves everyone": "tab:red"}

# =====================================================================
# FIGURE 1: one drop on the map
# =====================================================================
drop = sim.simulate_drop(ap_pos, np.random.default_rng(3), RULES, **cfg.PAPER_LIKE)
ue = drop["ue"]
fig, ax = plt.subplots(1, 2, figsize=(17, 8.6))
for a, rname, title in [(ax[0], NAMES[0], "Conventional rule: every user is served by its nearest AP only"),
                        (ax[1], NAMES[1], f"Cell-free with clusters: every AP serves its {cfg.L_PER_AP} strongest users")]:
    mask = drop["rules"][rname]["mask"]
    for m, k in zip(*np.where(mask)):
        a.plot([ap_pos[m, 0], ue[k, 0]], [ap_pos[m, 1], ue[k, 1]], "-", color="tab:blue" if "cluster" in rname else "gray",
               lw=0.9, alpha=0.55, zorder=1)
    served = mask.sum(axis=1)
    a.scatter(ap_pos[served > 0, 0], ap_pos[served > 0, 1], marker="^", s=55, color="black", zorder=3)
    a.scatter(ap_pos[served == 0, 0], ap_pos[served == 0, 1], marker="^", s=55, color="white", edgecolor="black", zorder=3)
    a.scatter(ue[:, 0], ue[:, 1], s=26, color="tab:orange", edgecolor="black", linewidth=0.6, zorder=4)
    a.set_xlim(0, cfg.AREA_M)
    a.set_ylim(0, cfg.AREA_M)
    a.set_aspect("equal")
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.set_title(f"{title}\n{int(mask.sum())} beams; {int((served == 0).sum())} of {cfg.N_AP} APs are silent (white triangles)", fontsize=10)
    a.grid(True, alpha=0.2)
fig.text(0.5, 0.01, "triangle = AP, orange dot = user, line = a beam from an AP to a user", ha="center", fontsize=10)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig(paths.image("map.png"), dpi=110)

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
    a.set_xlim(0, 1800)
    a.set_xlabel("speed of a user (Mbps)")
    a.set_ylabel("share of users at or below this speed (%)")
    a.set_title(f"{letter}. Speeds, {s} propagation\n(a curve to the right is faster; a steep curve is more even)")
    a.grid(True, alpha=0.3)
    a.legend(loc="lower right", fontsize=8)

# C: how many users per AP
L_VALUES = [1, 2, 3, 4, 5, 7, 10]
sweep = {s: {"mean": [], "p5": []} for s in SETTINGS}
for s, kw in SETTINGS.items():
    for L in L_VALUES:
        o = sim.simulate(ap_pos, {"c": (lambda beta, L=L: serving.cluster_mask(beta, L))}, **kw)["rules"]["c"]["rate"]
        sweep[s]["mean"].append(o.mean())
        sweep[s]["p5"].append(np.percentile(o, 5))
for s, ls in [("free space", "-"), ("paper-like", "--")]:
    ax[0, 2].plot(L_VALUES, sweep[s]["mean"], ls, marker="o", color="tab:blue", label=f"average speed, {s}")
    ax[0, 2].plot(L_VALUES, sweep[s]["p5"], ls, marker="s", color="tab:orange", label=f"slowest 5% (p5), {s}")
ax[0, 2].axvline(cfg.L_PER_AP, color="gray", ls=":", label=f"our choice L = {cfg.L_PER_AP}")
ax[0, 2].axvline(7, color="green", ls=":", label="L = 7 (the paper's output layout)")
ax[0, 2].set_xscale("log")
ax[0, 2].set_xticks(L_VALUES)
ax[0, 2].set_xticklabels([str(v) for v in L_VALUES])
ax[0, 2].set_xlabel("number of users each AP serves (L)")
ax[0, 2].set_ylabel("speed (Mbps)")
ax[0, 2].set_title("C. How many users should an AP serve?\n(slowest users: best at L = 4 with paper-like propagation)")
ax[0, 2].grid(True, alpha=0.3)
ax[0, 2].legend(fontsize=7)

# D: speed against distance to the nearest AP (paper-like)
edges = np.arange(0, 80, 10)
centres = (edges[:-1] + edges[1:]) / 2
dist = res["paper-like"]["dist_nearest"].ravel()
for rname in NAMES:
    r = res["paper-like"]["rules"][rname]["rate"].ravel()
    means = [r[(dist >= lo) & (dist < hi)].mean() if np.any((dist >= lo) & (dist < hi)) else np.nan
             for lo, hi in zip(edges[:-1], edges[1:])]
    ax[1, 0].plot(centres, means, "o-", color=COLOURS[rname], lw=2.5, label=rname)
ax[1, 0].set_xlabel("distance from the user to its nearest AP (m)")
ax[1, 0].set_ylabel("average speed (Mbps)")
ax[1, 0].set_title("D. Speed against distance (paper-like)\nconventional falls with distance, clusters stay level")
ax[1, 0].grid(True, alpha=0.3)
ax[1, 0].legend(fontsize=8)

# E: load of the APs
upa = {rname: res["paper-like"]["rules"][rname]["users_per_ap"].ravel() for rname in NAMES[:2]}
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

# F: 16 or 64 antennas
antennas = {n_ant: sim.simulate(ap_pos, RULES, n_ant=n_ant, **cfg.PAPER_LIKE) for n_ant in (16, 64)}
x = np.arange(len(NAMES))
for i, (n_ant, colour) in enumerate([(16, "lightsteelblue"), (64, "tab:blue")]):
    means = [antennas[n_ant]["rules"][n]["rate"].mean() for n in NAMES]
    p5 = [np.percentile(antennas[n_ant]["rules"][n]["rate"], 5) for n in NAMES]
    ax[1, 2].bar(x + (i - 0.5) * 0.36, means, 0.36, color=colour, label=f"average, {n_ant} antennas")
    ax[1, 2].scatter(x + (i - 0.5) * 0.36, p5, marker="_", s=900, color="tab:orange" if n_ant == 64 else "peachpuff",
                     linewidths=4, zorder=3, label=f"slowest 5%, {n_ant} antennas")
    for xi, v in zip(x, means):
        ax[1, 2].text(xi + (i - 0.5) * 0.36, v + 12, f"{v:.0f}", ha="center", fontsize=8)
ax[1, 2].set_xticks(x)
ax[1, 2].set_xticklabels([n.replace(" serves ", "\nserves ").replace(" only", "\nonly") for n in NAMES], fontsize=8)
ax[1, 2].set_ylabel("speed (Mbps)")
ax[1, 2].set_title("F. 16 or 64 antennas per AP? (paper-like)\n(bars: average, orange marks: slowest 5%)", fontsize=11)
ax[1, 2].legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(paths.image("results.png"), dpi=110)
print("Saved img/map.png and img/results.png")
if "--no-show" not in sys.argv:
    plt.show()