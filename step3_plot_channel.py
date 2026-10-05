# Four pictures that show how the random channel behaves.

import numpy as np
import matplotlib.pyplot as plt

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions
from channel_model import rayleigh_fading, full_channel

_, _, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())
rng = np.random.default_rng(config.SEED)

fig, ax = plt.subplots(2, 2, figsize=(12, 9))

# ---- A: a single antenna, many random draws ----
g1 = rayleigh_fading(20000, 1, 1, rng).ravel()
power1 = np.abs(g1) ** 2
ax[0, 0].hist(power1, bins=60, range=(0, 5), density=True, alpha=0.6, label="measured")
x = np.linspace(0, 5, 200)
ax[0, 0].plot(x, np.exp(-x), "k--", label="theory: e^(-x)")
ax[0, 0].set_xlabel("signal power of ONE antenna (1 = average)")
ax[0, 0].set_ylabel("how often")
ax[0, 0].set_title("A. One antenna: very random\n(often weak, sometimes strong)")
ax[0, 0].legend()

# ---- B: adding antennas makes the total strength steadier ----
for n, colour in zip([1, 4, 16, 64], ["tab:red", "tab:orange", "tab:green", "tab:blue"]):
    gg = rayleigh_fading(20000, 1, n, rng)
    total = np.sum(np.abs(gg) ** 2, axis=2).ravel() / n      # divide by n so the average is 1
    ax[0, 1].hist(total, bins=80, range=(0, 3), density=True, histtype="step", lw=2,
                  color=colour, label=f"{n} antennas")
ax[0, 1].set_xlabel("total strength / number of antennas (1 = average)")
ax[0, 1].set_ylabel("how often")
ax[0, 1].set_title("B. More antennas: steadier total strength")
ax[0, 1].legend()

# ---- C: channel hardening curve ----
ns = [1, 2, 4, 8, 16, 32, 64, 128, 256]
rels = []
for n in ns:
    s = np.sum(np.abs(rayleigh_fading(5000, 1, n, rng)) ** 2, axis=2).ravel()
    rels.append(s.std() / s.mean())
ax[1, 0].loglog(ns, rels, "o-", label="measured")
ax[1, 0].loglog(ns, 1 / np.sqrt(np.array(ns, float)), "k--", label="theory: 1/sqrt(N)")
ax[1, 0].axvline(config.N_ANTENNAS, color="red", ls=":", label=f"our network (N={config.N_ANTENNAS})")
ax[1, 0].set_xlabel("number of antennas N")
ax[1, 0].set_ylabel("relative randomness")
ax[1, 0].set_title("C. Randomness shrinks like 1/sqrt(N)")
ax[1, 0].grid(True, which="both", alpha=0.3)
ax[1, 0].legend()

# ---- D: one random draw versus the expected strength, for all 8 pairs ----
H = full_channel(beta, np.random.default_rng(config.SEED))
strength_db = 10 * np.log10(np.sum(np.abs(H) ** 2, axis=2)).ravel()
expected_db = 10 * np.log10(config.N_ANTENNAS * beta).ravel()
labels = [f"AP{m}-U{k}" for m in range(config.N_AP) for k in range(config.N_UE)]
xx = np.arange(len(labels))
ax[1, 1].scatter(xx, strength_db, s=90, color="tab:blue", zorder=3, label="this random draw")
ax[1, 1].scatter(xx, expected_db, s=200, marker="_", linewidths=3, color="black", zorder=4,
                 label="expected average (16 x beta)")
for i in range(len(labels)):
    ax[1, 1].plot([i, i], [strength_db[i], expected_db[i]], color="gray", lw=1, zorder=2)
ax[1, 1].set_xticks(xx)
ax[1, 1].set_xticklabels(labels, rotation=45)
ax[1, 1].grid(True, alpha=0.3)
ax[1, 1].set_ylabel("channel strength (dB)")
ax[1, 1].set_title("D. All 8 channels: one draw vs the expected average")
ax[1, 1].legend()

plt.tight_layout()
plt.savefig("step3_plot_channel.png", dpi=130)
plt.show()
print("Saved step3_plot_channel.png")