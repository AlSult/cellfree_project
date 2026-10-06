# Four pictures: signal vs interference vs noise, speeds, the Shannon curve,
# and who disturbs whom.

import numpy as np
import matplotlib.pyplot as plt

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions
from channel_model import full_channel
from power_allocation import nearest_ap_mask, equal_power_allocation
from precoding import mrt_precoders
from rate_calculation import sinr_and_rate, interference_table, noise_power_dbm, NOISE_W

n_ap, n_ue = config.N_AP, config.N_UE


def dbm(x):
    return 10 * np.log10(x) + 30


dist, _, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())
mask = nearest_ap_mask(dist)
p = equal_power_allocation(mask)

H = full_channel(beta, np.random.default_rng(config.SEED))      # the seed-0 draw
W = mrt_precoders(H, p)
signal, interference, sinr, rate = sinr_and_rate(H, W)
rate_noise_only = config.B_HZ * np.log2(1 + signal / NOISE_W) / 1e6

# average speed over many draws
trials = 5000
rng = np.random.default_rng(7)
rates = np.zeros((trials, n_ue))
for t in range(trials):
    Ht = full_channel(beta, rng)
    _, _, _, rates[t] = sinr_and_rate(Ht, mrt_precoders(Ht, p))
rate_avg = rates.mean(axis=0)

users = np.arange(n_ue)
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# ---- A: signal, interference, noise (dBm) ----
ax[0, 0].plot(users, dbm(signal), "o", ms=11, color="tab:green", label="wanted signal")
ax[0, 0].plot(users, dbm(interference), "s", ms=10, color="tab:red", label="interference (the other beams)")
ax[0, 0].axhline(noise_power_dbm(), color="gray", ls="--", lw=2, label=f"noise floor ({noise_power_dbm():.0f} dBm)")
for k in users:
    ax[0, 0].plot([k, k], [dbm(interference[k]), dbm(signal[k])], color="gray", lw=1, zorder=0)
    ax[0, 0].text(k + 0.07, (dbm(interference[k]) + dbm(signal[k])) / 2,
                  f"gap {10*np.log10(sinr[k]):.0f} dB", fontsize=8)
ax[0, 0].set_xticks(users)
ax[0, 0].set_xticklabels([f"User{k}" for k in users])
ax[0, 0].set_ylabel("power at the user (dBm)")
ax[0, 0].set_ylim(-95, -5)
ax[0, 0].set_title("A. What every user receives (this random draw)\nThe noise is far below the interference")
ax[0, 0].grid(True, alpha=0.3)
ax[0, 0].legend(fontsize=8, loc="center right")

# ---- B: speeds ----
w = 0.27
ax[0, 1].bar(users - w, rate_noise_only, w, color="lightgray", label="no interference (noise only)")
ax[0, 1].bar(users, rate, w, color="tab:blue", label="this draw")
ax[0, 1].bar(users + w, rate_avg, w, color="tab:orange", label=f"average of {trials} draws")
for k in users:
    ax[0, 1].text(k, rate[k] + 25, f"{rate[k]:.0f}", ha="center", fontsize=8)
    ax[0, 1].text(k + w, rate_avg[k] + 25, f"{rate_avg[k]:.0f}", ha="center", fontsize=8)
ax[0, 1].set_xticks(users)
ax[0, 1].set_xticklabels([f"User{k}" for k in users])
ax[0, 1].set_ylabel("speed (Mbps)")
ax[0, 1].set_title("B. Speed of every user\nInterference takes away most of the speed")
ax[0, 1].legend(fontsize=8)

# ---- C: Shannon curve ----
s_db = np.linspace(-10, 40, 300)
curve = config.B_HZ * np.log2(1 + 10 ** (s_db / 10)) / 1e6
ax[1, 0].plot(s_db, curve, color="black", lw=2)
ax[1, 0].plot(10 * np.log10(sinr), rate, "o", ms=11, color="tab:blue", label="our 4 users (this draw)")
for k in users:
    ax[1, 0].annotate(f"User{k}", (10 * np.log10(sinr[k]), rate[k]), textcoords="offset points", xytext=(-8, 10), fontsize=9)
step = config.B_HZ * np.log2(10) / 1e6
ax[1, 0].set_xlabel("SINR (dB)")
ax[1, 0].set_ylabel("speed (Mbps)")
ax[1, 0].set_title(f"C. Speed = bandwidth x log2(1 + SINR)\nAt high SINR, +10 dB gives only about {step:.0f} Mbps more")
ax[1, 0].grid(True, alpha=0.3)
ax[1, 0].legend(loc="upper left")

# ---- D: who disturbs whom ----
I = dbm(np.where(interference_table(H, W) > 0, interference_table(H, W), np.nan))
im = ax[1, 1].imshow(I, cmap="viridis_r")
for k in users:
    for j in users:
        label = "wanted" if j == k else f"{I[k, j]:.0f}"
        ax[1, 1].text(j, k, label, ha="center", va="center", fontsize=11,
                      color="gray" if j == k else ("black" if I[k, j] < -62 else "white"))
ax[1, 1].set_xticks(users)
ax[1, 1].set_yticks(users)
ax[1, 1].set_xticklabels([f"beam for\nUser{j}\n(from AP{np.argmax(mask[:, j])})" for j in users], fontsize=8)
ax[1, 1].set_yticklabels([f"User{k} hears" for k in users])
ax[1, 1].set_title("D. Interference (dBm) that each user hears from every other beam\n(this random draw)")
plt.colorbar(im, ax=ax[1, 1], label="dBm")

plt.tight_layout()
plt.savefig("img/step5_plot_rate.png", dpi=130)
plt.show()
print("Saved step5_plot_rate.png")