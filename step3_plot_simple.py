# A simpler way to see the random flicker: 60 random draws, shown one by one,
# for a different number of antennas.


import math
import numpy as np
import matplotlib.pyplot as plt

from channel_model import rayleigh_fading

rng = np.random.default_rng(5)
antenna_counts = [1, 4, 16, 64]
n_draws = 60
weak_level = 0.2          # "very weak" = below 20% of the average strength


def chance_very_weak(n, level):
    """Exact probability that the total strength of n antennas is below
    `level` x its average (a textbook formula for adding n random powers)."""
    x = n * level
    return 1 - math.exp(-x) * sum(x ** k / math.factorial(k) for k in range(n))


# ---------------- numbers ----------------
print("How often is the link 'nearly dead' (below 20% of its average strength)?")
print(f"{'antennas':>9} {'measured':>10} {'theory':>10}   {'weakest of 60':>14} {'strongest of 60':>16}")
all_ok = True
strengths_to_plot = {}
for n in antenna_counts:
    # many draws, only to measure the "nearly dead" frequency accurately
    big = np.sum(np.abs(rayleigh_fading(200000, 1, n, rng)) ** 2, axis=2).ravel() / n
    measured = np.mean(big < weak_level)
    theory = chance_very_weak(n, weak_level)
    # just 60 draws, to draw
    few = np.sum(np.abs(rayleigh_fading(n_draws, 1, n, rng)) ** 2, axis=2).ravel() / n
    strengths_to_plot[n] = few
    ok = abs(measured - theory) < 0.003
    all_ok &= ok
    print(f"{n:9d} {measured*100:9.2f}% {theory*100:9.2f}%   {few.min():14.2f} {few.max():16.2f}   {'OK' if ok else 'OFF'}")
print(f"\n[check] measured frequency matches the theory formula -> {'PASS' if all_ok else 'FAIL'}")

# ---------------- picture ----------------
fig, ax = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
for a, n in zip(ax, antenna_counts):
    s = strengths_to_plot[n]
    draws = np.arange(1, n_draws + 1)
    weak = s < weak_level
    a.scatter(draws[~weak], s[~weak], s=25, color="tab:blue", label="normal")
    a.scatter(draws[weak], s[weak], s=45, color="red", label="nearly dead (<20%)")
    a.axhline(1.0, color="black", ls="--", lw=1.5, label="average")
    a.axhline(weak_level, color="red", ls=":", lw=1)
    a.set_ylim(0, 4.6)
    a.set_xlabel("draw number (each draw = one new random channel)")
    a.set_title(f"{n} antenna" + ("s" if n > 1 else ""))
    a.grid(True, alpha=0.3)
ax[0].set_ylabel("signal strength  (1 = average)")
ax[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Step 3, simple view: the same random experiment, with more and more antennas")
plt.tight_layout()
plt.savefig("step3_plot_simple.png", dpi=130)
plt.show()
print("Saved step3_plot_simple.png")