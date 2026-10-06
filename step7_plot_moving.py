# Two figures: (1) the walk and the speeds of all users, (2) what ONE user sees.

import numpy as np
import matplotlib.pyplot as plt

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from mobility_model import generate_trajectories
from path_loss_model import beta_from_positions
from channel_model import full_channel
from power_allocation import nearest_ap_mask, cooperative_mask, equal_power_allocation
from precoding import mrt_precoders
from rate_calculation import sinr_and_rate
from closed_form_rate import expected_rate

n_ap, n_ue, T = config.N_AP, config.N_UE, config.N_STEPS
time = np.arange(T) * config.DT_S
colours = plt.cm.tab10(np.arange(n_ue))

ap_pos, ue_start = generate_ap_positions(), generate_ue_positions()
pos, speed = generate_trajectories(ue_start)

dist_t, beta_t = np.zeros((T, n_ap, n_ue)), np.zeros((T, n_ap, n_ue))
for t in range(T):
    dist_t[t], _, beta_t[t] = beta_from_positions(ap_pos, pos[t])
masks_t = {"nearest": np.array([nearest_ap_mask(dist_t[t]) for t in range(T)]),
           "cellfree": np.array([cooperative_mask() for _ in range(T)])}
power_t = {n: np.array([equal_power_allocation(masks_t[n][t]) for t in range(T)]) for n in masks_t}
serving = np.argmax(masks_t["nearest"], axis=1)

trials = 300
rng_single, rng_ens = np.random.default_rng(config.SEED), np.random.default_rng(1000)
single = {n: np.zeros((T, n_ue)) for n in masks_t}
ens = {n: np.zeros((trials, T, n_ue)) for n in masks_t}
for t in range(T):
    Hs = full_channel(beta_t[t], rng_single)
    for n in masks_t:
        single[n][t] = sinr_and_rate(Hs, mrt_precoders(Hs, power_t[n][t]))[3]
    for r in range(trials):
        Hr = full_channel(beta_t[t], rng_ens)
        for n in masks_t:
            ens[n][r, t] = sinr_and_rate(Hr, mrt_precoders(Hr, power_t[n][t]))[3]
mean = {n: ens[n].mean(axis=0) for n in masks_t}
ref = {n: np.array([expected_rate(beta_t[t], power_t[n][t]) for t in range(T)]) for n in masks_t}

# =============================== figure 1 ===============================
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# A: the walk on the map
ax[0, 0].plot([0, config.AREA_M], [config.AREA_M, 0], "k--", lw=1, alpha=0.5)
ax[0, 0].text(128, 58, "equally far from\nboth APs", fontsize=8, rotation=-45, alpha=0.7)
for k in range(n_ue):
    ax[0, 0].plot(pos[:, k, 0], pos[:, k, 1], "-", color=colours[k], lw=2, alpha=0.8)
    ax[0, 0].plot(pos[0, k, 0], pos[0, k, 1], "o", color=colours[k], ms=9)
    ax[0, 0].plot(pos[-1, k, 0], pos[-1, k, 1], "s", color=colours[k], ms=9)
    ax[0, 0].annotate(f"U{k}", pos[0, k], textcoords="offset points", xytext=(9, -15), color=colours[k], fontweight="bold")
ax[0, 0].scatter(*ap_pos.T, marker="^", s=220, color="black", zorder=3)
for m in range(n_ap):
    ax[0, 0].annotate(f"AP{m}", ap_pos[m], textcoords="offset points", xytext=(-8, 12), fontweight="bold", ha="right")
ax[0, 0].set_xlim(0, config.AREA_M)
ax[0, 0].set_ylim(0, config.AREA_M)
ax[0, 0].set_aspect("equal")
ax[0, 0].set_xlabel("x (m)")
ax[0, 0].set_ylabel("y (m)")
ax[0, 0].set_title(f"A. The walk ({config.N_STEPS} s). Circle = start, square = end")
ax[0, 0].grid(True, alpha=0.3)

# B: distance to each AP
for k in range(n_ue):
    ax[0, 1].plot(time, dist_t[:, 0, k], "-", color=colours[k], lw=2, label=f"User{k}")
    ax[0, 1].plot(time, dist_t[:, 1, k], "--", color=colours[k], lw=2)
ax[0, 1].set_xlabel("time (s)")
ax[0, 1].set_ylabel("distance (m)")
ax[0, 1].set_title("B. Distance to AP0 (solid line) and to AP1 (dashed line)")
ax[0, 1].grid(True, alpha=0.3)
ax[0, 1].legend(ncol=2, fontsize=9)

# C and D: speeds over time (ensemble average)
ymax = 1.05 * max(mean["nearest"].max(), mean["cellfree"].max())
for a, n, title in [(ax[1, 0], "nearest", "C. Step 5 rule (nearest AP): average speed of every user"),
                    (ax[1, 1], "cellfree", "D. Step 6 rule (cell-free): average speed of every user")]:
    for k in range(n_ue):
        a.plot(time, mean[n][:, k], "-", color=colours[k], lw=2.5, label=f"User{k}")
    a.set_ylim(0, ymax)
    a.set_xlabel("time (s)")
    a.set_ylabel(f"speed, average of {trials} random channels (Mbps)")
    a.set_title(title, fontsize=10)
    a.grid(True, alpha=0.3)
    a.legend(ncol=2, fontsize=9, loc="upper right")
for k in range(n_ue):                                   # mark the handovers of the Step 5 rule
    for t in np.where(np.diff(serving[:, k]) != 0)[0] + 1:
        ax[1, 0].plot(time[t], mean["nearest"][t, k], "v", color="red", ms=12, zorder=5)
        ax[1, 0].annotate("AP changes", (time[t], mean["nearest"][t, k]), textcoords="offset points",
                          xytext=(8, 10), color="red", fontsize=9)

ax[1, 0].annotate("at 1 s User1 switches to AP1:\nUser2 loses a lot, User0 gains", xy=(1, 700), xytext=(6, 640),
                  arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)

plt.tight_layout()
plt.savefig("step7_plot_moving.png", dpi=130)

# =============================== figure 2 ===============================
USER = 2
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for a, n, title in [(ax2[0], "nearest", f"Step 5 rule: User{USER}  (the jump at 1 s: User1 switches to AP1)"),
                    (ax2[1], "cellfree", f"Step 6 rule: the same walk of User{USER}")]:
    a.plot(time, single[n][:, USER], "o-", color="lightgray", mec="gray", lw=1.5, label="what you would see: ONE random channel per second")
    a.plot(time, mean[n][:, USER], "-", color="tab:blue", lw=3, label=f"average of {trials} random channels")
    a.plot(time, ref[n][:, USER], "k--", lw=2, label="paper-and-pencil formula (no random numbers)")
    a.set_xlabel("time (s)")
    a.set_title(title)
    a.grid(True, alpha=0.3)
ax2[0].set_ylabel(f"speed of User{USER} (Mbps)")
ax2[0].legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig("step7_plot_user_view.png", dpi=130)
plt.show()
print("Saved step7_plot_moving.png and step7_plot_user_view.png")