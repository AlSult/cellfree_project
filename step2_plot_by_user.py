# A clearer way to look at the Step 2 results: one view per AP, plus a
# bar chart that compares the two APs user by user.

import numpy as np
import matplotlib.pyplot as plt

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions, free_space_path_loss_db

dist, pl_db, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())

# ---- print the ranking of users for each AP (closest first) ----
print("Users ranked from closest to farthest:")
rankings = []
for m in range(config.N_AP):
    order = np.argsort(dist[m], kind="stable")
    rankings.append(list(order))
    text = "  ->  ".join(f"User{k} ({dist[m, k]:.1f} m, {pl_db[m, k]:.1f} dB)" for k in order)
    print(f"AP{m}:  {text}")

# ---- CHECK: the ranking you worked out by looking at the map ----
expected = {0: [0, 1, 2, 3],     # AP0: User0, then User1, then User2, then User3
            1: [2, 1, 0, 3]}     # AP1: User2, then User1, then User0, then User3
for m in range(config.N_AP):
    ok = rankings[m] == expected[m]
    print(f"[check] AP{m} ranking matches the map -> {'PASS' if ok else 'FAIL'}")

# ---- the picture ----
d_curve = np.logspace(0, 3, 200)
loss_curve = [free_space_path_loss_db(d) for d in d_curve]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

# Panel 1: bars, users on the x-axis, one bar per AP
x = np.arange(config.N_UE)
ax[0].bar(x - 0.2, pl_db[0], 0.4, color="tab:blue", label="AP0")
ax[0].bar(x + 0.2, pl_db[1], 0.4, color="tab:orange", label="AP1")
ax[0].set_xticks(x)
ax[0].set_xticklabels([f"User{k}" for k in range(config.N_UE)])
ax[0].set_ylim(0, 100)
ax[0].set_ylabel("path loss (dB)   (higher bar = weaker signal)")
ax[0].set_title("Same users, two APs")
ax[0].legend(loc="upper center", ncol=2)
for k in range(config.N_UE):
    ax[0].text(k - 0.2, pl_db[0, k] + 0.5, f"{pl_db[0, k]:.0f}", ha="center", fontsize=8)
    ax[0].text(k + 0.2, pl_db[1, k] + 0.5, f"{pl_db[1, k]:.0f}", ha="center", fontsize=8)

# Panels 2 and 3: the loss curve, one panel per AP
for m, colour in enumerate(["tab:blue", "tab:orange"]):
    a = ax[m + 1]
    a.semilogx(d_curve, loss_curve, color="lightgray", lw=2)
    a.scatter(dist[m], pl_db[m], s=90, color=colour, zorder=3)
    for k in range(config.N_UE):
        a.annotate(f"User{k}", (dist[m, k], pl_db[m, k]), textcoords="offset points",
                   xytext=(6, -14 if k % 2 == 0 else 8), fontsize=9)
    a.set_xlabel("distance (metres, log scale)")
    a.set_ylabel("path loss (dB)")
    a.set_title(f"Seen from AP{m} only")
    a.set_xlim(8, 300)
    a.set_ylim(60, 92)
    a.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("step2_plot_by_user.png", dpi=130)
plt.show()
print("Saved step2_plot_by_user.png")