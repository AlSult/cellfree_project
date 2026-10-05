# Draws the path-loss curve and marks our 8 AP-user pairs on it.

import numpy as np
import matplotlib.pyplot as plt

from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions, free_space_path_loss_db

dist, pl_db, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())

d_curve = np.logspace(0, 3, 200)    # 1 m to 1000 m
loss_curve = [free_space_path_loss_db(d) for d in d_curve]

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogx(d_curve, loss_curve, color="gray", lw=2, label="free-space loss")
colors = ["tab:blue", "tab:orange"]
for m in range(dist.shape[0]):
    ax.scatter(dist[m], pl_db[m], s=80, color=colors[m], zorder=3, label=f"AP{m} to the 4 users")
    for k in range(dist.shape[1]):
        ax.annotate(f"U{k}", (dist[m, k], pl_db[m, k]), textcoords="offset points", xytext=(5, -12), fontsize=8)
ax.set_xlabel("distance (metres, log scale)")
ax.set_ylabel("path loss (dB)")
ax.set_title("Step 2: the farther away, the weaker the signal")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("step2_path_loss.png", dpi=130)
plt.show()
print("Saved step2_path_loss.png")