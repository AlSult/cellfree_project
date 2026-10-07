# All the numbers of the MIDDLE network. The radio numbers (frequency, power, bandwidth,
# noise, antennas) are taken from config.py, so the tiny and the middle network always
# use exactly the same radio. Only the SIZE and the serving rule are new.

from config import FC_HZ, C, PTX_DBM, B_HZ, NF_DB, N0_DBM_HZ, N_ANTENNAS     # same radio as before

# ---- size and layout ----
GRID_SIDE = 5                       # APs stand on a 5 x 5 grid
AP_SPACING_M = 100.0                # distance between neighbouring APs (the paper: about 91 m)
AP_JITTER_M = 10.0                  # every AP is moved randomly by up to 10 m (no perfect grid)
N_AP = GRID_SIDE ** 2               # 25 APs
N_UE = 20                           # 20 users (the paper: 100 users for 121 APs, so about 0.83 per AP)
AREA_M = GRID_SIDE * AP_SPACING_M   # 500 m x 500 m

# ---- serving rule ----
L_PER_AP = 3                        # every AP serves its L strongest users (our design choice)

# ---- simulation ----
N_DROPS = 300                       # a "drop" = users placed at random + one random channel
SEED_LAYOUT = 1                     # seed for the AP positions
SEED_DROPS = 5                      # seed for the drops

# ---- the two propagation settings we compare ----
FREE_SPACE = dict(alpha=2.0, shadow_sigma_db=0.0)     # what the tiny network used
PAPER_LIKE = dict(alpha=3.8, shadow_sigma_db=8.0)     # path loss exponent 3.8 and 8 dB shadowing (Table I of the paper)