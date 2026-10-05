# Builds the path-loss tables for our 2 APs x 4 users and checks them.

import math
import numpy as np

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions, free_space_path_loss_db

ap_pos = generate_ap_positions()
ue_pos = generate_ue_positions()
dist, pl_db, beta = beta_from_positions(ap_pos, ue_pos)

np.set_printoptions(linewidth=150)
print("Distance (m), rows = APs, columns = users:")
print(dist.round(1))
print("\nPath loss (dB):")
print(pl_db.round(1))
print("\nFraction of signal power that survives (beta):")
print(beta)

# Save both tables so you can open them in Excel
#np.savetxt("step2_path_loss_db.csv", pl_db, delimiter=",", fmt="%.3f", header="user0,user1,user2,user3")
#np.savetxt("step2_beta.csv", beta, delimiter=",", fmt="%.6e", header="user0,user1,user2,user3")
#print("\nSaved step2_path_loss_db.csv and step2_beta.csv")

# ---------------- CHECKS ----------------
print("\nCHECKS")

# 1. Compare with the engineers' rule-of-thumb formula, which is written
#    in different units: loss = 32.44 + 20*log10(f in MHz) + 20*log10(d in km)
d_test = 100.0
rule_of_thumb = 32.44 + 20 * math.log10(config.FC_HZ / 1e6) + 20 * math.log10(d_test / 1000)
ours = free_space_path_loss_db(d_test)
ok1 = abs(ours - rule_of_thumb) < 0.01
print(f"[1] loss at 100 m: ours {ours:.2f} dB, rule of thumb {rule_of_thumb:.2f} dB -> {'PASS' if ok1 else 'FAIL'}")

# 2. Doubling the distance must add exactly 20*log10(2) = 6.02 dB
added = free_space_path_loss_db(200) - free_space_path_loss_db(100)
ok2 = abs(added - 20 * math.log10(2)) < 1e-9
print(f"[2] doubling the distance adds {added:.2f} dB (expected 6.02 dB) -> {'PASS' if ok2 else 'FAIL'}")

# 3. User1 is equally far from both APs, so the loss must be equal
ok3 = abs(pl_db[0, 1] - pl_db[1, 1]) < 1e-9
print(f"[3] User1 has the same loss to both APs -> {'PASS' if ok3 else 'FAIL'}")

# 4. For each AP, a user that is farther away must have MORE loss
ok4 = True
for m in range(config.N_AP):
    order_by_distance = np.argsort(dist[m])
    ok4 &= bool(np.all(np.diff(pl_db[m, order_by_distance]) >= 0))
print(f"[4] farther away always means more loss -> {'PASS' if ok4 else 'FAIL'}")

# 5. Converting beta back to dB must give the path loss again
back_to_db = -10 * np.log10(beta)
ok5 = bool(np.allclose(back_to_db, pl_db))
print(f"[5] beta converted back to dB equals the path loss -> {'PASS' if ok5 else 'FAIL'}")

# 6. Over a wide range of distances the loss must grow by 20 dB every time
#    the distance grows 10 times (the straight-line rule on a log axis)
d_range = np.logspace(0, 3, 50)
loss_range = np.array([free_space_path_loss_db(d) for d in d_range])
slope = np.polyfit(np.log10(d_range), loss_range, 1)[0]
ok6 = abs(slope - 20) < 0.01
print(f"[6] loss grows {slope:.2f} dB per 10x distance (expected 20) -> {'PASS' if ok6 else 'FAIL'}")