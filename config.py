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

# ---- Step 5: noise, interference and speed ----
PTX_DBM = 46.0       # power budget of one AP: 46 dBm (Table I of the paper)
B_HZ = 100e6         # bandwidth: 100 MHz (Table I of the paper)
NF_DB = 9.0          # noise figure of the receiver (Table I of the paper)
N0_DBM_HZ = -174.0   # thermal noise per Hz (Table I of the paper)

# ---- Step 7: users walking around ----
N_STEPS = 40         # number of time steps (snapshots)
DT_S = 1.0           # seconds between two snapshots
SPEED_MIN_MS = 1.0   # slowest walking speed (m/s)
SPEED_MAX_MS = 4.0   # fastest walking speed (m/s)