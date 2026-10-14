# paper_config.py  (PAPER-SIZE network)
# All the numbers of the network we want to build: the size of the paper (Table I), in one place.
# The physics files (paper_channel.py, paper_precoding.py, paper_rates.py) read their numbers from here.

# ---- the radio (Table I of the paper) ----
FC_HZ = 3.5e9          # carrier frequency: 3.5 GHz
C = 3e8                # speed of light in metres per second
PTX_DBM = 46.0         # power budget of one AP: 46 dBm
B_HZ = 100e6           # bandwidth: 100 MHz
NF_DB = 9.0            # noise figure of the receiver: 9 dB
N0_DBM_HZ = -174.0     # thermal noise per Hz: -174 dBm/Hz

# ---- size and layout (Table I of the paper) ----
AREA_M = 1000.0                     # the map is 1 km x 1 km
GRID_SIDE = 11                      # APs stand on an 11 x 11 grid
N_AP = GRID_SIDE ** 2               # 121 APs
N_UE = 100                          # 100 users
N_ANTENNAS = 64                     # 64 antennas on every AP
AP_SPACING_M = AREA_M / GRID_SIDE   # 90.9 m between neighbouring APs
AP_JITTER_M = 10.0                  # every AP is moved randomly by up to 10 m (no perfect grid)

# ---- serving rule ----
L_PER_AP = 3                        # every AP serves its L strongest users (our design choice, tested in the L table)

# ---- simulation ----
N_DROPS = 100                       # a "drop" = users placed at random + one random channel
SEED_LAYOUT = 1                     # seed for the AP positions
SEED_DROPS = 5                      # seed for the drops

# ---- the two propagation settings we compare ----
FREE_SPACE = dict(alpha=2.0, shadow_sigma_db=0.0)     # free space: exponent 2, no shadowing
PAPER_LIKE = dict(alpha=3.8, shadow_sigma_db=8.0)     # exponent 3.8 and 8 dB shadowing (Table I): the default for the dataset

# ---- Random Waypoint mobility (Table I: Random Waypoint, 3-60 km/h) ----
RWP_V_MIN_MS = 3 / 3.6              # 3 km/h (Table I)
RWP_V_MAX_MS = 60 / 3.6             # 60 km/h (Table I)
RWP_PAUSE_S = 0.0                   # NOT in the paper: our assumption (zero pause = classic Random Waypoint)
RWP_BURN_IN_S = 3000.0              # only needed if the pause is not zero
STEP_S = 1.0                        # NOT in the paper: time between two snapshots (1 s)
EPISODE_STEPS = 40                  # NOT in the paper: length of an episode (40 snapshots = 40 s)
SHADOW_CORR_DIST_EQ2_M = 50.0       # NOT in the paper: shadowing memory (metres walked)

# =====================================================================
# Dataset (Step 10)
# =====================================================================
TRAFFIC_DEMAND = 1.0                # NOT modelled in the paper-size network: every user has the SAME demand (a simplification;
                                    # the paper's input has a demand per user). Stored as a column of 1.0 in the dataset.