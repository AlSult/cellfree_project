# Four pictures that show what aiming the antennas does.

import numpy as np
import matplotlib.pyplot as plt

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions
from channel_model import full_channel
from precoding import mrt_precoders, no_aim_precoders, signal_seen_by_users

_, _, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())
n_ap, n_ue, n_ant = config.N_AP, config.N_UE, config.N_ANTENNAS
idx = np.arange(n_ue)
p = np.ones((n_ap, n_ue))
reference = p * beta

# the same random channel as in Step 3 (seed 0)
H = full_channel(beta, np.random.default_rng(config.SEED))
W_aim, W_off = mrt_precoders(H, p), no_aim_precoders(H, p)
A_aim, A_off = signal_seen_by_users(H, W_aim), signal_seen_by_users(H, W_off)
gain_aim = np.abs(A_aim[:, idx, idx]) ** 2 / reference
gain_off = np.abs(A_off[:, idx, idx]) ** 2 / reference

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# ---- A: the 16 arrows of AP0 -> User0, with and without aiming ----
# every antenna n contributes one arrow: conj(h_n) * w_n  (divided by sqrt(beta) to get nice numbers)
arrows_aim = np.conj(H[0, 0]) * W_aim[0, 0] / np.sqrt(beta[0, 0])
arrows_off = np.conj(H[0, 0]) * W_off[0, 0] / np.sqrt(beta[0, 0])
for arrows, colour, name in [(arrows_off, "tab:red", "NOT aimed"), (arrows_aim, "tab:green", "AIMED")]:
    chain = np.concatenate([[0], np.cumsum(arrows)])
    ax[0, 0].plot(chain.real, chain.imag, "-o", ms=4, lw=1.2, color=colour, alpha=0.8)
    ax[0, 0].annotate("", xy=(chain[-1].real, chain[-1].imag), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", color=colour, lw=3))
    ax[0, 0].plot([], [], color=colour, lw=3, label=f"{name}: total arrow length^2 = {abs(chain[-1])**2:.1f}")
ax[0, 0].scatter([0], [0], color="black", zorder=5)
ax[0, 0].set_aspect("equal")
ax[0, 0].margins(0.15)
ax[0, 0].grid(True, alpha=0.3)
ax[0, 0].set_ylim(-1.3, 0.9)
ax[0, 0].legend(loc="lower right", fontsize=8)
ax[0, 0].set_title("A. The 16 antennas of AP0 as 16 small arrows (user 0)\nAimed: all arrows point the same way, so the total is long")

# ---- B: 8 links, aimed vs not aimed (dB) ----
labels = [f"AP{m}-U{k}" for m in range(n_ap) for k in range(n_ue)]
xx = np.arange(len(labels))
ax[0, 1].plot(xx, 10 * np.log10(gain_aim.ravel()), "o", ms=10, color="tab:green", label="aimed")
ax[0, 1].plot(xx, 10 * np.log10(gain_off.ravel()), "s", ms=9, color="tab:red", label="not aimed")
ax[0, 1].axhline(10 * np.log10(n_ant), color="tab:green", ls="--", lw=1, label=f"expected when aimed: {n_ant}x = {10*np.log10(n_ant):.0f} dB")
ax[0, 1].axhline(0, color="tab:red", ls="--", lw=1, label="expected when not aimed: 1x = 0 dB")
ax[0, 1].set_xticks(xx)
ax[0, 1].set_xticklabels(labels, rotation=45)
ax[0, 1].set_ylabel("received power compared with ONE antenna (dB)")
ax[0, 1].set_title("B. All 8 links in this random draw")
ax[0, 1].grid(True, alpha=0.3)
ax[0, 1].set_ylim(-7, 22)
ax[0, 1].legend(fontsize=8, loc="upper center", ncol=2)

# ---- many random draws for panels C and D ----
trials = 5000
rng = np.random.default_rng(3)
ga = np.zeros(trials)
go = np.zeros(trials)
leak_sum = np.zeros((n_ue, n_ue))
for t in range(trials):
    Ht = full_channel(beta, rng)
    Aa = signal_seen_by_users(Ht, mrt_precoders(Ht, p))
    Ao = signal_seen_by_users(Ht, no_aim_precoders(Ht, p))
    ga[t] = np.abs(Aa[0, 0, 0]) ** 2 / reference[0, 0]
    go[t] = np.abs(Ao[0, 0, 0]) ** 2 / reference[0, 0]
    leak_sum += np.abs(Aa[0]) ** 2 / beta[0][:, None]
leak_avg = leak_sum / trials

# ---- C: distribution over many draws (AP0 -> User0) ----
ax[1, 0].hist(10 * np.log10(go), bins=100, range=(-25, 20), density=True, alpha=0.6, color="tab:red", label="not aimed")
ax[1, 0].hist(10 * np.log10(ga), bins=100, range=(-25, 20), density=True, alpha=0.6, color="tab:green", label="aimed")
ax[1, 0].axvline(0, color="black", ls="--", lw=1, label="one antenna (average)")
ax[1, 0].axvline(10 * np.log10(n_ant), color="tab:green", ls="--", lw=1)
ax[1, 0].set_xlabel("received power compared with ONE antenna (dB)")
ax[1, 0].set_ylabel("how often")
ax[1, 0].set_title(f"C. {trials} random draws (AP0 to User0)\nAiming shifts everything by about +12 dB and removes the bad moments")
ax[1, 0].legend()

# ---- D: who hears what (AP0 only), average over many draws ----
im = ax[1, 1].imshow(leak_avg, cmap="viridis")
for k in range(n_ue):
    for j in range(n_ue):
        ax[1, 1].text(j, k, f"{leak_avg[k, j]:.1f}", ha="center", va="center",
                      color="white" if leak_avg[k, j] < 8 else "black", fontsize=12)
ax[1, 1].set_xticks(range(n_ue))
ax[1, 1].set_yticks(range(n_ue))
ax[1, 1].set_xticklabels([f"beam for\nUser{j}" for j in range(n_ue)])
ax[1, 1].set_yticklabels([f"User{k} hears" for k in range(n_ue)])
ax[1, 1].set_title("D. What each user hears from AP0's four beams\n(average, compared with one antenna)")
plt.colorbar(im, ax=ax[1, 1])

plt.tight_layout()
plt.savefig("step4_plot_precoding.png", dpi=130)
plt.show()
print("Saved step4_plot_precoding.png")