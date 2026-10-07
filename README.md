# Kestrel Home: returns-risk model and service (Variant A)

Scores each order at dispatch with the chance it will be returned, explains why in plain words, and recommends **SHIP** or **CALL BEFORE DISPATCH**. It does **not** recommend holding orders (see `MEMO.md`).

No paid API, no network calls, no keys. Plain scikit-learn logistic regression. Cost per prediction: Rs 0.

## Run it (clean machine, Python 3.10+)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/service.py                                   # then open http://127.0.0.1:8000
```
The trained model is committed (`model/model.joblib`, coefficients only, no customer rows), so the service starts without the data files.

Endpoint: `POST /score` with one order as JSON (fields as in the data pack README). Required: `sku, sales_channel, payment_mode, discount_pct, qty, order_value_inr, promised_delivery_days, customer_prior_orders, customer_prior_returns`. Optional: `shield_member` (Y/N), `customer_id`, `is_gift`, `delivery_pincode`.
```bash
curl -s -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{"sku":"KH-RV-02","sales_channel":"marketplace","payment_mode":"cod","shield_member":"N","discount_pct":25,"qty":1,"order_value_inr":16499.25,"promised_delivery_days":8,"customer_prior_orders":3,"customer_prior_returns":2}'
```
Returns `return_probability`, `risk_band`, `action`, `advice`, `reasons[]` (sentences), `warnings[]`. Bad input gets a 400/422 with a readable message, never a stack trace. `pickup_scheduled_at` and `last_service_event_type` are accepted but **ignored with a warning**.

## Reproduce the numbers (needs the data pack in `data/`, which is not in the repo)
```bash
mkdir data   # put train.csv test_unlabelled.csv customers.csv products.csv sample_submission.csv here
python src/train.py              # validation, rupee tables -> evidence/, predictions.csv, model/model.joblib
python evidence/make_charts.py   # evidence/evidence.png
python -m unittest discover tests
```

## Layout
| Path | What |
|---|---|
| `predictions.csv` | score per test order (probability of return) |
| `src/features.py` | cleaning (dedup, x100 fix) and features, shared by training and service |
| `src/train.py` | out-of-time validation, rupee analysis, final fit, predictions |
| `src/service.py`, `templates/index.html` | endpoint and screen |
| `EVIDENCE.md`, `evidence/` | how we know it works and how often it does not |
| `MEMO.md` | one-page memo to Ritu |

## Data handling
Kestrel policy s10: data must not be published or shared beyond the engagement team. `data/` and per-order validation outputs are git-ignored. **Keep this repo private** (or share only with Banao); `predictions.csv` lists real order ids.

## Known limits
See `EVIDENCE.md`. Short version: AUC about 0.78, about 1 in 3 flagged orders really is returned, and a 95% accuracy target is not reachable from dispatch-time data.
