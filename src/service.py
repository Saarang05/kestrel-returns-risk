"""Kestrel returns-risk service.  python src/service.py  ->  http://127.0.0.1:8000
No paid API, no network calls. Needs only flask, pandas, numpy, scikit-learn, joblib.
POST /score  takes one order as JSON, returns score + plain-language reasons + recommended action.
"""
import os, sys, json
import numpy as np, pandas as pd, joblib
from flask import Flask, request, jsonify, render_template
sys.path.insert(0, os.path.dirname(__file__))
from train import prep, COLS   # same feature code as training

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = joblib.load(os.path.join(ROOT, "model", "model.joblib"))
MODEL, REF, THR = ART["model"], ART["ref_rates"], ART["threshold"]
CATALOG = pd.DataFrame.from_dict(ART["catalog"], orient="index").reset_index().rename(columns={"index": "sku"})
FEATS = list(MODEL[0].get_feature_names_out())
COEF = MODEL[-1].coef_[0]
# customers.csv is confidential (policy s10) and not shipped; if present locally we use it to look up Shield status.
CUST = None
_cpath = os.path.join(os.environ.get("DATA_DIR", os.path.join(ROOT, "data")), "customers.csv")
if os.path.exists(_cpath):
    CUST = pd.read_csv(_cpath)

REQUIRED = ["sku", "sales_channel", "payment_mode", "discount_pct", "qty", "order_value_inr",
            "promised_delivery_days", "customer_prior_orders", "customer_prior_returns"]
DEFAULTS = {"is_gift": "N", "delivery_pincode": "440001", "order_placed_at": "2026-10-01 12:00",
            "customer_id": "UNKNOWN", "delivery_note": ""}
NOT_AT_DISPATCH = ["pickup_scheduled_at", "last_service_event_type", "returned", "source"]
GROUP = {"family": "product", "price_tier": "product", "log_price": "product", "qty": "product",
         "payment_mode": "payment", "cod_n": "payment", "sales_channel": "channel",
         "prior_orders_c": "history", "prior_ret_c": "history", "prior_return_rate": "history",
         "shield_n": "shield", "discount_pct": "discount", "promised_delivery_days": "promise",
         "is_gift_n": "gift", "no_address": "address"}

def pct(x): return f"{round(100 * x)}%"

def explain(rec_row, groups):
    """Turn grouped logit contributions into sentences a Kestrel employee can read."""
    r = rec_row.iloc[0]; ov = REF["overall"]; out = []
    for g, c in groups:
        if g == "payment":
            pm = {"cod": "Cash on delivery", "emi": "EMI", "prepaid_card": "Prepaid by card", "prepaid_upi": "Prepaid by UPI"}[r.payment_mode]
            t = f"{pm}: {pct(REF['payment_mode'][r.payment_mode])} of these orders came back (average {pct(ov)})"
        elif g == "product":
            t = f"{r.family}: {pct(REF['family'][r.family])} of these orders came back (average {pct(ov)})"
        elif g == "history":
            t = (f"Customer has {int(r.customer_prior_returns)} earlier return(s) in {int(r.customer_prior_orders)} order(s)"
                 if c > 0 else f"Customer has a clean history ({int(r.customer_prior_returns)} returns in {int(r.customer_prior_orders)} orders)")
        elif g == "shield":
            t = ("Kestrel Shield member: returns are free for them, so they come back more often (%s vs %s). "
                 "Do not hold this order; a confirmation call is fine." % (pct(REF['shield_n']['1']), pct(REF['shield_n']['0']))) if r.shield_n else "Not a Shield member"
        elif g == "discount":
            t = f"Discount of {int(r.discount_pct)}%: deeper discounts are returned more" if c > 0 else f"Discount of {int(r.discount_pct)}% is modest"
        elif g == "promise":
            t = f"Promised delivery in {int(r.promised_delivery_days)} days: longer promises are returned more" if c > 0 else f"Short delivery promise ({int(r.promised_delivery_days)} days)"
        elif g == "gift":
            t = "Order is a gift" if r.is_gift_n else "Not a gift"
        elif g == "channel":
            t = f"Sold via {r.sales_channel.replace('_', ' ')}: {pct(REF['sales_channel'][r.sales_channel])} return rate"
        elif g == "address":
            t = "No delivery address captured (walk-in default pincode)" if r.no_address else "Address captured"
        else:
            t = g
        out.append({"reason": t, "direction": "raises risk" if c > 0 else "lowers risk", "weight": round(float(c), 3)})
    return out

def score_record(rec):
    problems = [k for k in REQUIRED if k not in rec or rec[k] in (None, "")]
    if problems:
        return {"error": "Missing required field(s)", "missing": problems, "required": REQUIRED}, 400
    warnings = []
    for k in NOT_AT_DISPATCH:
        if k in rec and rec[k] not in (None, ""):
            warnings.append(f"'{k}' was ignored: it is not known at dispatch (it is only written after a return is approved).")
    row = {**DEFAULTS, **{k: v for k, v in rec.items() if k not in NOT_AT_DISPATCH}}
    if row["sku"] not in set(CATALOG.sku):
        return {"error": f"Unknown sku '{row['sku']}'", "known_skus": sorted(CATALOG.sku)}, 400
    try:
        for k in ["discount_pct", "qty", "order_value_inr", "promised_delivery_days", "customer_prior_orders", "customer_prior_returns"]:
            row[k] = float(row[k])
    except (TypeError, ValueError):
        return {"error": "Numeric fields must be numbers"}, 400
    if row["sales_channel"] not in REF["sales_channel"] or row["payment_mode"] not in REF["payment_mode"]:
        return {"error": "sales_channel must be one of %s and payment_mode one of %s" % (list(REF["sales_channel"]), list(REF["payment_mode"]))}, 400
    # Shield status: explicit field > local customers.csv > assume not Shield (and say so)
    shield = row.pop("shield_member", None); state = row.pop("state", "UNK")
    if shield is None and CUST is not None and row["customer_id"] in set(CUST.customer_id):
        c = CUST[CUST.customer_id == row["customer_id"]].iloc[0]; shield = c.shield_member; state = c.state
    if shield is None:
        shield = "N"; warnings.append("Shield status unknown (send shield_member or customer_id with customers.csv available): assumed 'N'.")
    cu = pd.DataFrame([{"customer_id": row["customer_id"], "state": state, "shield_member": shield}])
    df = pd.DataFrame([row])
    exp = float(CATALOG.set_index("sku").loc[row["sku"], "list_price_inr"]) * row["qty"] * (1 - row["discount_pct"] / 100)
    if abs(row["order_value_inr"] / exp - 100) < 1:
        warnings.append("order_value_inr looks 100x too large (paise, as in the Oct-2025 gateway issue): rescaled.")
    elif not 0.9 < row["order_value_inr"] / exp < 1.1:
        warnings.append(f"order_value_inr (Rs {row['order_value_inr']:.0f}) does not match list price x qty x discount (Rs {exp:.0f}); check the payment record.")
    d = prep(df, cu, CATALOG)
    p = float(MODEL.predict_proba(d[COLS])[:, 1][0])
    # contributions = coef * (x - training mean), grouped by business concept
    x = MODEL[0].transform(d[COLS]); x = np.asarray(x.todense() if hasattr(x, "todense") else x).ravel()
    contrib = COEF * (x - ART["x_mean"]); g = {}
    for name, c in zip(FEATS, contrib):
        base = name.split("__", 1)[1]
        key = next((k for k in sorted(GROUP, key=len, reverse=True) if base == k or base.startswith(k + "_")), base)
        g[GROUP.get(key, key)] = g.get(GROUP.get(key, key), 0.0) + float(c)
    top = sorted(g.items(), key=lambda kv: -abs(kv[1]))[:4]
    is_shield = shield == "Y"
    if p >= THR:
        action = "CALL_BEFORE_DISPATCH"
        advice = "Phone the customer to confirm model, address and intent before dispatch." + (" Shield member: call only, do NOT hold the order." if is_shield else " Do not hold longer than a day: about 12% of held orders are cancelled.")
    else:
        action, advice = "SHIP", "Dispatch as normal."
    band = "high" if p >= THR else ("medium" if p >= 0.10 else "low")
    return {"order_id": rec.get("order_id"), "return_probability": round(p, 4), "risk_band": band, "action": action, "advice": advice,
            "expected_return_cost_rs": round(p * ART["return_cost"]), "call_threshold": round(THR, 3),
            "reasons": explain(d, top), "warnings": warnings,
            "model_note": f"Logistic regression trained on {ART['train_rows']} orders up to {ART['train_end'][:10]}. Probabilities are calibrated on past data; ~1 in 3 flagged orders is actually returned."}, 200

app = Flask(__name__, template_folder=os.path.join(ROOT, "templates"))

@app.get("/")
def index(): return render_template("index.html")

@app.get("/health")
def health(): return {"status": "ok", "paid_api_required": False, "customers_lookup": CUST is not None}

@app.post("/score")
def score():
    rec = request.get_json(silent=True)
    if not isinstance(rec, dict):
        return jsonify({"error": "Send one order as a JSON object"}), 400
    try:
        body, code = score_record(rec)
    except Exception as e:            # fail politely, never a stack trace to the screen
        return jsonify({"error": "Could not score this record", "detail": str(e)}), 422
    return jsonify(body), code

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
