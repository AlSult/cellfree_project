# Builds the AP-to-user distance table and checks that it is correct.


import numpy as np

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions

ap_pos = generate_ap_positions()   # shape (2, 2)
ue_pos = generate_ue_positions()   # shape (4, 2)

# ---- Method A: the slow, easy-to-read way (two loops, Pythagoras) ----
dist_loop = np.zeros((config.N_AP, config.N_UE))
for m in range(config.N_AP):            # m = which AP
    for k in range(config.N_UE):        # k = which user
        dx = ap_pos[m, 0] - ue_pos[k, 0]
        dy = ap_pos[m, 1] - ue_pos[k, 1]
        dist_loop[m, k] = np.sqrt(dx ** 2 + dy ** 2)

# ---- Method B: a different way, one line of numpy ----
dist_numpy = np.linalg.norm(ap_pos[:, None, :] - ue_pos[None, :, :], axis=2)

print("Distance table in metres (rows = APs, columns = users):")
print(dist_loop.round(1))

# Save the table so you can open it in Excel
#np.savetxt("step1_distance_table.csv", dist_loop, delimiter=",", fmt="%.3f",
#           header="user0,user1,user2,user3")
#print("\nSaved step1_distance_table.csv")

# ---- CHECKS ----
print("\nCHECKS")

# 1. A number we can calculate by hand: AP0 (50,50) to User0 (60,60)
by_hand = np.sqrt(10 ** 2 + 10 ** 2)
ok1 = abs(dist_loop[0, 0] - by_hand) < 1e-9
print(f"[1] AP0 to User0: code {dist_loop[0, 0]:.3f} m, by hand {by_hand:.3f} m -> {'PASS' if ok1 else 'FAIL'}")

# 2. Everyone must be inside the map
everything = np.vstack([ap_pos, ue_pos])
ok2 = bool(np.all((everything >= 0) & (everything <= config.AREA_M)))
print(f"[2] everyone inside the {config.AREA_M:.0f} x {config.AREA_M:.0f} m map -> {'PASS' if ok2 else 'FAIL'}")

# 3. The table has the right shape
ok3 = dist_loop.shape == (config.N_AP, config.N_UE)
print(f"[3] table shape {dist_loop.shape}, expected ({config.N_AP}, {config.N_UE}) -> {'PASS' if ok3 else 'FAIL'}")

# 4. Two different methods must give the same answer
ok4 = bool(np.allclose(dist_loop, dist_numpy))
print(f"[4] loop method and numpy method agree -> {'PASS' if ok4 else 'FAIL'}")

# 5. User1 is exactly between the APs, so its two distances must be equal
ok5 = abs(dist_loop[0, 1] - dist_loop[1, 1]) < 1e-9
print(f"[5] User1 is equally far from both APs -> {'PASS' if ok5 else 'FAIL'}")