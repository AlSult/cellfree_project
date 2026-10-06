# Audit of the tiny dataset: 38 of 38 checks passed

margin = measured / limit. Below 1 passes; the smaller, the safer.

| code | group | check | result | measured | limit | margin | picture |
|---|---|---|---|---|---|---|---|
| A1 | Structure | all expected arrays are in the file | PASS |  |  |  |  |
| A2 | Structure | every array has the shape the documentation promises | PASS |  |  |  |  |
| A3 | Structure | no NaN or infinite value anywhere | PASS |  |  |  |  |
| A4 | Structure | data types are right (H complex, masks True/False, the rest real numbers) | PASS |  |  |  |  |
| A5 | Structure | the info file documents units and the random seeds | PASS |  |  |  |  |
| A6 | Structure | no two episodes are copies of each other | PASS |  |  |  |  |
| B1 | Plausibility | everybody is inside the map | PASS |  |  |  |  |
| B2 | Plausibility | distances are positive and not larger than the diagonal of the map | PASS |  |  |  |  |
| B3 | Plausibility | path loss is between 40 and 100 dB | PASS |  |  |  | panel 1 |
| B4 | Plausibility | speeds are never negative and never above the 'no interference' limit | PASS |  |  |  | panel 2 |
| B5 | Plausibility | no AP ever uses more than its power budget | PASS |  |  |  |  |
| B6 | Plausibility | noise floor is -85 dBm (-174 + 80 + 9) | PASS |  |  |  |  |
| C1 | Independent recomputation | distances recomputed from the positions | PASS | 0 | 1e-08 | 1e-08 |  |
| C2 | Independent recomputation | path loss recomputed with the rule of thumb 32.44 + 20log10(f/MHz) + 20log10(d/km) | PASS | 0.00177 | 0.01 | 0.177 | panel 1 |
| C3 | Independent recomputation | the nearest-AP rule: every user has exactly one AP and it is the closest one | PASS |  |  |  |  |
| C4n | Independent recomputation | power of every beam is budget / 4 where a beam exists (nearest rule) | PASS |  |  |  |  |
| C4c | Independent recomputation | power of every beam is budget / 4 where a beam exists (cellfree rule) | PASS |  |  |  |  |
| C5 | Independent recomputation | signal, interference, SINR and speed recomputed from H for all 800 snapshots | PASS | 4.47e-13 | 1e-08 | 4.47e-05 | panel 3 |
| C6 | Independent recomputation | speed = bandwidth x log2(1 + SINR) | PASS | 0 | 1e-05 | 1e-11 |  |
| D1 | Consistency | nobody moves faster than its own speed in any second | PASS | 1.93e-14 | 1e-09 | 1.93e-05 | panel 4 |
| D2 | Consistency | almost all steps are exactly speed x time (the others are bounces) | PASS | 0.0154 | 0.1 | 0.154 | panel 4 |
| D3 | Consistency | the distance to an AP can never change by more than the user walked | PASS | 0 | 1e-09 | 1e-07 | panel 4 |
| D4 | Consistency | all episodes start where the documentation says | PASS |  |  |  |  |
| D5 | Consistency | the cell-free wanted signal is never weaker than the nearest-AP one (same channel) | PASS | 0 | 1e-12 | 0.0001 | panel 9 |
| D6 | Consistency | cell-free: every AP uses exactly its whole budget | PASS |  |  |  |  |
| E1 | Random part | random part: average 0, average power 1, real and imaginary variance 0.5 | PASS | 0.104 | 1 | 0.104 |  |
| E2 | Random part | antenna powers follow the exponential law of Rayleigh fading (Kolmogorov-Smirnov test) | PASS | 0.00339 | 0.0072 | 0.47 | panel 5 |
| E3 | Random part | different antennas are independent | PASS | 0.0241 | 0.08 | 0.302 | panel 6 |
| E4 | Random part | one second and the next one are independent (a new draw every second, as documented) | PASS | 0.00205 | 0.025 | 0.082 | panel 6 |
| E5 | Random part | different AP-user pairs are independent | PASS | 0.00795 | 0.06 | 0.133 | panel 6 |
| E6 | Random part | average channel strength is 16 x beta (array gain) | PASS | 0.000672 | 0.02 | 0.0336 | panel 7 |
| E7 | Random part | channel hardening: relative randomness of the strength is 1/sqrt(16) = 0.250 | PASS | 0.00634 | 0.05 | 0.127 | panel 7 |
| F1 | Stress tests | halving every power changes the speeds by less than 1% (noise is negligible) | PASS | 1.66e-05 | 0.01 | 0.00166 | panel 8 |
| F2 | Stress tests | renumbering the users just renumbers the results | PASS |  |  |  |  |
| F3 | Stress tests | renumbering the APs changes nothing | PASS |  |  |  |  |
| F4 | Stress tests | switching off the beams of one user: its speed is 0 and everybody else hears less interference | PASS |  |  |  |  |
| G1 | Reproducibility | same seeds -> every array is identical to the delivered file | PASS |  |  |  |  |
| G2 | Reproducibility | other seeds -> different data | PASS |  |  |  |  |

## Differences from Table I of the paper

| item | tiny network | paper | verdict |
|---|---|---|---|
| carrier | 3.5 | 3.5 | same |
| bandwidth (MHz) | 100.0 | 100 | same |
| power (dBm) | 46.0 | 46 | same |
| noise figure (dB) | 9.0 | 9 | same |
| noise density (dBm/Hz) | -174.0 | -174 | same |
| antennas per AP | 16 | 64 | DIFFERENT |
| area | 200 m x 200 m | 1 km x 1 km | DIFFERENT |
| APs | 2 | 121 (20 federated) | DIFFERENT |
| users | 4 | 100 | DIFFERENT |
| path loss exponent | 2.0 | 3.8 | DIFFERENT |
| shadow fading | none | 8 dB | DIFFERENT |
| channel model | free space + Rayleigh | 3GPP TR 38.901 | DIFFERENT |
| mobility | straight walk, bounces | Random Waypoint | DIFFERENT |
| speeds | 4 to 14 km/h | 3 to 60 km/h | DIFFERENT |
| all episodes start from the same positions | True | 100 runs with different topologies | NOTE |
