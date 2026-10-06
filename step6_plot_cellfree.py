# Four pictures: who gains and who loses, signal and interference, the
# map of the cooperation gain, and the slowest user in every draw.

import numpy as np
import matplotlib.pyplot as plt

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions
from channel_model import full_channel
from power_allocation import nearest_ap_mask, cooperative_mask, equal_power_allocation
from precoding import mrt_precoders
from rate_calculation import sinr_and_rate

n_ap, n_ue = config.N_AP, config.N_UE
users = np.arange(n_ue)


def dbm(x):
    return 10 * np.log10(x) + 30


ap_pos, ue_pos = generate_ap_positions(), generate_ue_positions()
dist, _, beta = beta_from_positions(ap_pos, ue_pos)
rules = {"nearest": nearest_ap_mask(dist), "cellfree": cooperative_mask()}
power = {name: equal_power_allocation(mask) for name, mask in rules.items()}
budget = 10 ** ((config.PTX_DBM - 30) / 10)

# ---- many draws, the same random channels for both rules ----
trials = 5000
rng = np.random.default_rng(7)
rates = {n: np.zeros((trials, n_ue)) for n in rules}
sig = {n: np.zeros(n_ue) for n in rules}
itf = {n: np.zeros(n_ue) for n in rules}
for t in range(trials):
    Ht = full_channel(beta, rng)
    for name in rules:
        s, i, _, r = sinr_and_rate(Ht, mrt_precoders(Ht, power[name]))
        rates[name][t] = r
        sig[name] += s
        itf[name] += i
avg = {n: rates[n].mean(axis=0) for n in rules}

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# ---- A: average speed per user ----
w = 0.38
ax[0, 0].bar(users - w / 2, avg["nearest"], w, color="tab:gray", label="Step 5: nearest AP only")
ax[0, 0].bar(users + w / 2, avg["cellfree"], w, color="tab:blue", label="Step 6: cell-free")
for k in users:
    ax[0, 0].text(k - w / 2, avg["nearest"][k] + 15, f"{avg['nearest'][k]:.0f}", ha="center", fontsize=9)
    ax[0, 0].text(k + w / 2, avg["cellfree"][k] + 15, f"{avg['cellfree'][k]:.0f}", ha="center", fontsize=9)
    change = 100 * (avg["cellfree"][k] - avg["nearest"][k]) / avg["nearest"][k]
    ax[0, 0].text(k, -95, f"{change:+.0f}%", ha="center", fontsize=10,
                  color="tab:green" if change > 0 else "tab:red", fontweight="bold")
ax[0, 0].set_xticks(users)
ax[0, 0].set_xticklabels([f"User{k}" for k in users])
ax[0, 0].set_ylim(-130, 1000)
ax[0, 0].axhline(0, color="black", lw=0.8)
ax[0, 0].set_ylabel(f"average speed over {trials} draws (Mbps)")
ax[0, 0].set_title("A. Who gains and who loses with cell-free\n(the percentage under each user is the change)")
ax[0, 0].legend()

# ---- B: signal and interference, averaged ----
for name, filled, dx in [("nearest", False, -0.12), ("cellfree", True, 0.12)]:
    s_dbm, i_dbm = dbm(sig[name] / trials), dbm(itf[name] / trials)
    ax[0, 1].plot(users + dx, s_dbm, "o", ms=11, color="tab:green", mfc="tab:green" if filled else "white", mew=2)
    ax[0, 1].plot(users + dx, i_dbm, "s", ms=10, color="tab:red", mfc="tab:red" if filled else "white", mew=2)
ax[0, 1].plot([], [], "o", color="tab:green", mfc="white", mew=2, ms=10, label="wanted signal, Step 5")
ax[0, 1].plot([], [], "o", color="tab:green", ms=10, label="wanted signal, cell-free")
ax[0, 1].plot([], [], "s", color="tab:red", mfc="white", mew=2, ms=9, label="interference, Step 5")
ax[0, 1].plot([], [], "s", color="tab:red", ms=9, label="interference, cell-free")
ax[0, 1].set_xticks(users)
ax[0, 1].set_xticklabels([f"User{k}" for k in users])
ax[0, 1].set_ylabel("average power at the user (dBm)")
ax[0, 1].set_title("B. Cell-free adds a little signal, but much more interference\nfor the users that stand next to an AP")
ax[0, 1].grid(True, alpha=0.3)
ax[0, 1].legend(fontsize=8, loc="lower left")

# ---- C: map of the cooperation gain for ONE lonely user ----
grid = np.linspace(0, config.AREA_M, 201)
Xg, Yg = np.meshgrid(grid, grid)
d_g = np.maximum(np.stack([np.hypot(Xg - ap_pos[m, 0], Yg - ap_pos[m, 1]) for m in range(n_ap)]), 1.0)
beta_g = 10 ** (-(20 * np.log10(4 * np.pi * d_g * config.FC_HZ / config.C)) / 10)
amp_g = np.sqrt(budget / n_ue * config.N_ANTENNAS * beta_g)
gain_db = 10 * np.log10(amp_g.sum(axis=0) ** 2 / amp_g.max(axis=0) ** 2)
im = ax[1, 0].imshow(gain_db, extent=[0, config.AREA_M, 0, config.AREA_M], origin="lower", cmap="viridis")
cs = ax[1, 0].contour(Xg, Yg, gain_db, levels=[3, 5], colors="white", linewidths=0.8)
ax[1, 0].clabel(cs, fmt="%d dB", fontsize=8)
ax[1, 0].scatter(*ap_pos.T, marker="^", s=160, color="white", edgecolor="black", zorder=3, label="AP")
ax[1, 0].scatter(*ue_pos.T, marker="o", s=60, color="red", edgecolor="white", zorder=3, label="our users")
for k in users:
    ax[1, 0].annotate(f"U{k}", ue_pos[k], textcoords="offset points", xytext=(6, 5), color="white", fontsize=9)
ax[1, 0].set_xlabel("x (m)")
ax[1, 0].set_ylabel("y (m)")
ax[1, 0].set_title("C. If a user were ALONE: extra signal from the second AP\n(no other beams, so no interference)")
ax[1, 0].legend(loc="lower right", fontsize=8)
plt.colorbar(im, ax=ax[1, 0], label="gain of cooperation (dB)")

# ---- D: the slowest user in every draw ----
for name, colour, label in [("nearest", "tab:gray", "Step 5: nearest AP only"), ("cellfree", "tab:blue", "Step 6: cell-free")]:
    slowest = np.sort(rates[name].min(axis=1))
    ax[1, 1].plot(slowest, np.arange(1, trials + 1) / trials * 100, color=colour, lw=2.5,
                  label=f"{label} (average {slowest.mean():.0f} Mbps)")
ax[1, 1].set_xlabel("speed of the SLOWEST user in a draw (Mbps)")
ax[1, 1].set_ylabel("share of the draws with a slowest user at or below this speed (%)")
ax[1, 1].set_title("D. The slowest user of a draw is usually faster with cell-free\n(the blue curve is to the right for most of the draws)")
ax[1, 1].grid(True, alpha=0.3)
ax[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("img/step6_plot_cellfree.png", dpi=130)
plt.show()
print("Saved step6_plot_cellfree.png")