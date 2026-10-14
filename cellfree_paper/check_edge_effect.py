# Compares speeds of users near the border (< 150 m) with users in the middle. Equal power, clusters L=3.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import paper_config as cfg
import paper_layout as layout
import paper_gains as gains
import paper_serving as serving
from paper_channel import full_channel
from paper_precoding import mrt_precoders
from paper_rates import sinr_and_rate

ap_pos = layout.generate_ap_positions()
A, S = cfg.PAPER_LIKE["alpha"], cfg.PAPER_LIKE["shadow_sigma_db"]
edge_r, mid_r = [], []
for i in range(10):
    rng = np.random.default_rng([cfg.SEED_DROPS, i])
    ue = layout.generate_user_drop(rng)
    beta = gains.beta_from_distance(gains.distance_matrix(ap_pos, ue), A, S, rng)
    H = full_channel(beta, rng)
    rate = sinr_and_rate(H, mrt_precoders(H, serving.equal_power_per_ap(serving.cluster_mask(beta, 3))))[3]
    d_border = np.minimum(ue.min(axis=1), (cfg.AREA_M - ue).min(axis=1))
    edge_r += list(rate[d_border < 150]); mid_r += list(rate[d_border >= 150])
print(f"edge users (<150 m from border): n={len(edge_r)}, mean speed {np.mean(edge_r):.0f} Mbps, slowest 5%: {np.percentile(edge_r, 5):.0f} Mbps")
print(f"middle users                   : n={len(mid_r)}, mean speed {np.mean(mid_r):.0f} Mbps, slowest 5%: {np.percentile(mid_r, 5):.0f} Mbps")
print(f"edge / middle (mean) = {np.mean(edge_r)/np.mean(mid_r):.2f}")