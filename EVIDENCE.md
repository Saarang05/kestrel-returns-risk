# Evidence that it works, and how often it does not

## Set-up
- **Data used:** `train.csv` de-duplicated (11,155 rows -> 10,504 orders; 651 partner-feed re-imports were exact copies), Apr 2025 to Jun 2026, return rate 11.4%.
- **Fixes:** 700 October-2025 orders had `order_value_inr` stored x100 (paise) and were rescaled (test file has none).
- **Features (dispatch-time only):** product family/price tier, channel, payment mode, discount, qty, promised days, gift, customer prior orders/returns, Shield, no-address flag.
- **Excluded on purpose:** `pickup_scheduled_at`, `last_service_event_type` (written after a return is approved or after delivery), `source`, `signup_date`, `delivery_note`.
- **Model:** logistic regression (C=0.3). Tried gradient boosting, extra interactions, SKU and pincode-prefix features: none beat it out of time (AUC 0.770 vs 0.778), so the simpler model won.

## Validation: rolling-origin, trained only on the past
| Train up to | Validate on | n | Return rate | AUC | PR-AUC | Accuracy @0.5 | "Never flag" accuracy |
|---|---|---|---|---|---|---|---|
| Sep 2025 | Oct-Dec 2025 | 2,082 | 10.8% | 0.769 | 0.346 | 89.5% | 89.2% |
| Dec 2025 | Jan-Mar 2026 | 2,126 | 11.6% | 0.775 | 0.395 | 89.4% | 88.4% |
| Mar 2026 | Apr-Jun 2026 | 2,126 | 11.5% | 0.789 | 0.417 | 89.6% | 88.5% |
| **Pooled** | 6,334 orders | | 11.3% | **0.778** | **0.384** | **89.5%** | **88.7%** |

Bootstrap at the test file's size (2,096 rows): AUC 95% range 0.74 to 0.81; PR-AUC 0.33 to 0.45.
Probabilities are well calibrated (`evidence/calibration.csv`, `evidence.png`): the top decile predicts 41% and sees 41%.

## What the 95% target looks like
- Doing nothing ("never flag") is already **88.7% accurate**. Our best accuracy is **89.5%**. 95% is not reachable from what is known at dispatch.
- Flagging the top 15% as "will be returned" gives only 84.1% accuracy, *worse* than doing nothing, yet it is the setting that makes money. Accuracy rewards the wrong thing here, so we report AUC, PR-AUC and rupees.
- Using the pickup and service-event columns gives AUC 0.997 and 99% accuracy (`metrics.json`, `leaky_demo`). That is the trap: those fields appear only after a return is approved, and the test file does not have them.

## Lift and rupees (pooled out-of-time, 700 orders a month)
| Call top | Calls/month | Share actually returned | Returns caught | Net/month, Rs 1,150 | Net/month, Rs 600 |
|---|---|---|---|---|---|
| 5% | 35 | 52.5% | 23% | 5,825 | 2,286 |
| 10% | 70 | 41.1% | 36% | 8,423 | 2,888 |
| **15%** | **105** | **34.6%** | **46%** | **9,911** | **2,911** |
| 20% | 140 | 30.1% | 53% | 10,658 | 2,548 |
| 30% | 210 | 24.9% | 66% | 11,592 | 1,529 |

Net = share returned x 35% prevented (spring pilot) x return cost, minus Rs 45 per call. Calls still break even if they prevent only **11%** of returns at the top-15% setting.
Holding: only 12% of held orders cancel; at an *assumed* Rs 1,000 margin per lost good order, holding loses money from the top 10% upward (`hold_net_per_flag_rs_margin1000`). About 45% of flagged orders are Shield members.

## Where it gets it wrong (`evidence/segment_errors.csv`, top-15% flag)
- **Misses 54% of returns overall (62% of returns by non-Shield customers).** Returns with no warning sign (prepaid, first-time customer, cheap fan or mixer) look like normal orders. About 70% of returns on prepaid UPI and on mixers/cooktops are missed.
- **2 in 3 flagged orders are fine.** Precision is about 35%, so any hold or call lands on many good customers.
- **Shield members:** 30% are flagged (vs 11% of others) because they return 2x as often. Precision is similar (37%), but they are the customers Meenal wants protected.
- **Cold start:** customers with no order history (29% of orders) are ranked less well: AUC 0.75 vs 0.78 for repeat customers.

## Not verified
- Whether calls prevent 35% of returns (one spring pilot, design unknown). Needs a randomised test.
- Whether `customer_prior_*` are strictly as-of-order in all rows (they look it, but pre-April-2025 history cannot be rebuilt).
- Whether July-September 2026 returns behave like earlier months. Score distribution on test is similar (16% above the call threshold vs 15% expected), which suggests no big shift.
