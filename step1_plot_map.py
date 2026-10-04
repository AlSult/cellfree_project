# Draws the map with the APs and users.

import matplotlib.pyplot as plt

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions

ap_pos = generate_ap_positions()
ue_pos = generate_ue_positions()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(ap_pos[:, 0], ap_pos[:, 1], marker="^", s=250, color="black", label="AP", zorder=3)
ax.scatter(ue_pos[:, 0], ue_pos[:, 1], marker="o", s=100, color="tab:red", label="User", zorder=3)

for m in range(config.N_AP):
    ax.annotate(f"AP{m}", ap_pos[m], textcoords="offset points", xytext=(8, 8))
for k in range(config.N_UE):
    ax.annotate(f"User{k}", ue_pos[k], textcoords="offset points", xytext=(8, 8))

ax.set_xlim(0, config.AREA_M)
ax.set_ylim(0, config.AREA_M)
ax.set_xlabel("x (metres)")
ax.set_ylabel("y (metres)")
ax.set_title("Step 1: the map")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("step1_map.png", dpi=130)
plt.show()
print("Saved step1_map.png")