# All the numbers you may want to change live here.
# ---- Step 1: the map ----
AREA_M = 200.0   # the map is a square: 200 m x 200 m
N_AP = 2         # number of APs (towers)
N_UE = 4         # number of users (phones)

# ---- Step 2: how signals weaken with distance ----
FC_HZ = 3.5e9    # carrier frequency: 3.5 GHz (from Table I of the paper)
C = 3e8          # speed of light in metres per second

# ---- Step 3: many antennas and random fluctuation ----
N_ANTENNAS = 16  # antennas on every AP
SEED = 0         # same seed = same "random" numbers every run