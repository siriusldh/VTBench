# Image-clustered bootstrap CI (frozen)

- Resampling unit: **image**; B = 10000; seed = **20260730** (numpy PCG64)
- Independent (B, 20) draw per dimension (5 dimensions, drawn in fixed order); rho and agreement share it within a dimension
- CI: 95% percentile (2.5/97.5)
- rho: BT win rates, identical to `metric_bt.py`
- Agreement: `compute_agreement.run()` strict plurality; tied top votes -> tie
- Item accounting: 21 unique method pairs × 20 images = **420 items per dimension**; 5 dimensions = **2100 items in total** (1609 exact agreements, no item excluded, 20 votes per item)
- These 2100 items are **not** independent bootstrap units: resampling is over 100 image clusters (20 per dimension), each carrying its full set of 21 pairs and 20 questionnaires
- Reproduce from the repository root: `python reproducibility/bootstrap_ci.py`

| Dimension | rho | rho 95% CI | exact agreement | pair agreement | agreement 95% CI |
|---|---|---|---|---|---|
| Aesthetic | 0.8929 | [0.6429, 0.9643] | 371/420 | 0.8833 | [0.8333, 0.9310] |
| Background | 0.8818 | [0.8214, 0.9643] | 349/420 | 0.8310 | [0.8119, 0.8500] |
| Cross | 0.9643 | [0.7027, 0.9910] | 283/420 | 0.6738 | [0.6333, 0.7143] |
| Hand | 0.8571 | [0.7143, 0.9643] | 307/420 | 0.7310 | [0.6929, 0.7667] |
| Text | 0.8829 | [0.5000, 0.9286] | 299/420 | 0.7119 | [0.6690, 0.7548] |
| **Overall** | — | — | **1609/2100** | **0.7662** | **[0.7486, 0.7833]** |

Full settings (seed, B, CI method): `bootstrap_ci.json`.
