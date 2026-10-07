# A real stream cannot carry more than about 8 bits per second per Hz (the best 5G modulation,
# 256-QAM, has 8 bits per symbol, before coding). With 100 MHz that is about 800 Mbps per stream.
# Shannon's formula has no such limit. How much do our averages change if we cap the speed?

import numpy as np

import mid_config as cfg
import mid_layout
import mid_simulate as sim

CAP_MBPS = 8 * cfg.B_HZ / 1e6          # 8 bit/s/Hz x 100 MHz = 800 Mbps
ap_pos = mid_layout.generate_ap_positions()
rules = sim.standard_rules()

print(f"cap = {CAP_MBPS:.0f} Mbps per user (one stream)\n")
print(f"{'propagation':14s} {'rule':27s} {'average':>8} {'capped average':>15} {'users above the cap':>20}")
capped = {}
for name, kw in (("free space", cfg.FREE_SPACE), ("paper-like", cfg.PAPER_LIKE)):
    res = sim.simulate(ap_pos, rules, **kw)
    for rule, o in res["rules"].items():
        rate = o["rate"]
        capped[(name, rule)] = np.minimum(rate, CAP_MBPS).mean()
        print(f"{name:14s} {rule:27s} {rate.mean():8.0f} {capped[(name, rule)]:15.0f} {100 * np.mean(rate > CAP_MBPS):19.1f}%")

near, clus = capped[("paper-like", "nearest AP only")], capped[("paper-like", "clusters (L=3)")]
print(f"\n[check] paper-like propagation: the conventional rule is still ahead of the clusters after the cap "
      f"({near:.0f} against {clus:.0f} Mbps) -> {'YES' if near > clus else 'NO'}")