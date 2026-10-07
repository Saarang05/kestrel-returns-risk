"""Train the returns-risk model, validate out-of-time, score the test file, write evidence.
Run from the repo root:  python src/train.py
"""
import sys, json, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from features import build, load_ref
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, brier_score_loss

CAT = ["family", "sales_channel", "payment_mode", "price_tier"]
NUM = ["discount_pct", "qty", "log_price", "promised_delivery_days", "is_gift_n", "prior_orders_c",
       "prior_ret_c", "prior_return_rate", "shield_n", "no_address", "cod_n"]
COLS = CAT + NUM
RETURN_COST, CALL_COST, CALL_SAVE, HOLD_CANCEL = 1150, 45, 0.35, 0.12

def prep(df, cu, pr):
    d = build(df, cu, pr)
    d["log_price"] = np.log(d.list_price_inr)
    d["prior_orders_c"] = d.customer_prior_orders.clip(upper=6)
    d["prior_ret_c"] = d.customer_prior_returns.clip(upper=4)
    return d

def make_model():
    ct = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT),
                            ("n", StandardScaler(), NUM)])
    return make_pipeline(ct, LogisticRegression(C=0.3, max_iter=3000))

if __name__ == "__main__":
    cu, pr = load_ref()
    raw = pd.read_csv("data/train.csv", parse_dates=["order_placed_at"])
    n_raw = len(raw)
    tr = raw.drop_duplicates("order_id").sort_values("order_placed_at").reset_index(drop=True)
    d = prep(tr, cu, pr); y = d.returned.values
    out = {"rows_raw": n_raw, "rows_dedup": len(tr), "dupes_removed": n_raw - len(tr),
           "value_x100_rows_fixed": int(d.order_value_fixed.sum()), "base_rate": float(y.mean())}

    # ---- rolling-origin (out-of-time) validation: 3 folds, each trained only on the past
    folds = [("2025-10-01", "2026-01-01"), ("2026-01-01", "2026-04-01"), ("2026-04-01", "2026-07-01")]
    oot = []
    out["folds"] = []
    for s, e in folds:
        itr = (d.order_placed_at < s).values; iva = ((d.order_placed_at >= s) & (d.order_placed_at < e)).values
        m = make_model().fit(d.loc[itr, COLS], y[itr]); p = m.predict_proba(d.loc[iva, COLS])[:, 1]
        out["folds"].append({"train_end": s, "val_end": e, "n_val": int(iva.sum()), "base": float(y[iva].mean()),
                             "auc": roc_auc_score(y[iva], p), "pr_auc": average_precision_score(y[iva], p),
                             "acc_at_0.5": accuracy_score(y[iva], p > .5), "always_no_acc": 1 - float(y[iva].mean())})
        t = d.loc[iva, ["order_id", "shield_n", "payment_mode", "family", "order_value_inr"]].copy(); t["y"] = y[iva]; t["p"] = p; t["fold"] = s
        oot.append(t)
    oot = pd.concat(oot)
    out["oot_n"] = len(oot); out["oot_base"] = float(oot.y.mean())
    out["oot_auc"] = roc_auc_score(oot.y, oot.p); out["oot_pr_auc"] = average_precision_score(oot.y, oot.p)
    out["oot_brier"] = brier_score_loss(oot.y, oot.p)
    out["oot_acc_at_0.5"] = accuracy_score(oot.y, oot.p > .5); out["oot_always_no_acc"] = 1 - out["oot_base"]
    oot.to_csv("evidence/oot_predictions.csv", index=False)

    # ---- lift / economics on pooled out-of-time predictions
    rows = []
    oot = oot.sort_values("p", ascending=False).reset_index(drop=True)
    for k in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        top = oot.head(int(len(oot) * k)); prec = top.y.mean(); rec = top.y.sum() / oot.y.sum()
        call_net = prec * CALL_SAVE * RETURN_COST - CALL_COST                       # per flagged order
        call_net_600 = prec * CALL_SAVE * 600 - CALL_COST
        # hold: 12% cancel. Cancelled returner saves return cost; cancelled non-returner loses margin M (assumed Rs 1000)
        hold_net = HOLD_CANCEL * (prec * RETURN_COST - (1 - prec) * 1000)
        rows.append({"flag_top_pct": int(k * 100), "orders_flagged_per_700": round(700 * k), "precision": prec, "recall": rec,
                     "lift": prec / oot.y.mean(), "shield_share_of_flags": top.shield_n.mean(),
                     "call_net_per_flag_rs": call_net, "call_net_if_return_cost_600": call_net_600,
                     "hold_net_per_flag_rs_margin1000": hold_net,
                     "call_net_per_month_rs": call_net * 700 * k})
    lift = pd.DataFrame(rows); lift.to_csv("evidence/lift_and_rupees.csv", index=False)
    # calibration deciles
    oot["bin"] = pd.qcut(oot.p, 10, labels=False, duplicates="drop")
    cal = oot.groupby("bin").agg(mean_pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index(); cal.to_csv("evidence/calibration.csv", index=False)
    # segment error table (where it is wrong): flag top 15%
    thr = oot.p.quantile(0.85); oot["flag"] = oot.p >= thr
    seg = []
    for col in ["payment_mode", "family", "shield_n"]:
        for v, g in oot.groupby(col):
            seg.append({"segment": f"{col}={v}", "n": len(g), "return_rate": g.y.mean(), "flag_rate": g.flag.mean(),
                        "precision_of_flags": g[g.flag].y.mean() if g.flag.any() else np.nan,
                        "missed_returns_share": ((~g.flag) & (g.y == 1)).sum() / max(g.y.sum(), 1)})
    pd.DataFrame(seg).to_csv("evidence/segment_errors.csv", index=False)
    out["threshold_top15pct_score"] = float(thr)

    # ---- leakage demonstration (not used in the product)
    from sklearn.ensemble import HistGradientBoostingClassifier
    leak = d.copy()
    Xl = pd.DataFrame({"pk": leak.pickup_scheduled_at.notna().astype(int)})
    for e_ in ["REVERSE_PICKUP", "TECH_VISIT", "DEMO_DONE", "INSTALL_DONE"]: Xl[e_] = (leak.last_service_event_type == e_).astype(int)
    itr = (d.order_placed_at < "2026-04-01").values; iva = ~itr
    ml = HistGradientBoostingClassifier(max_iter=100).fit(Xl[itr], y[itr]); pl = ml.predict_proba(Xl[iva])[:, 1]
    out["leaky_demo"] = {"auc": roc_auc_score(y[iva], pl), "acc": accuracy_score(y[iva], pl > .5),
                         "note": "pickup/service columns are written after a return is approved; absent in test file. NOT used."}

    # ---- final fit on all history, score test
    m = make_model().fit(d[COLS], y)
    te = pd.read_csv("data/test_unlabelled.csv", parse_dates=["order_placed_at"])
    dt = prep(te, cu, pr)
    te_scores = m.predict_proba(dt[COLS])[:, 1]
    sub = pd.read_csv("data/sample_submission.csv")
    assert list(sub.order_id) == list(te.order_id), "order mismatch"
    sub["score"] = np.round(te_scores, 6)
    sub.to_csv("predictions.csv", index=False)
    out["test_rows"] = len(te); out["test_mean_score"] = float(te_scores.mean())
    out["test_pct_scored_over_thr"] = float((te_scores >= thr).mean())
    out["test_shield_share_over_thr"] = float(dt.loc[te_scores >= thr, "shield_n"].mean())

    # reference rates used by the service to explain reasons in plain words
    ref = {"overall": float(y.mean())}
    for col in ["payment_mode", "family", "sales_channel", "shield_n", "is_gift_n", "customer_prior_returns", "long_promise", "hi_discount", "no_address"]:
        ref[col] = {str(k): float(v) for k, v in d.groupby(col).returned.mean().items()}
    Xt = m[0].transform(d[COLS]); x_mean = np.asarray(Xt.mean(axis=0)).ravel()
    catalog = pr.set_index("sku")[["family", "list_price_inr"]].to_dict("index")   # public catalogue, no customer data
    joblib.dump({"model": m, "cols": COLS, "cat": CAT, "num": NUM, "ref_rates": ref, "threshold": float(thr),
                 "x_mean": x_mean, "catalog": catalog, "return_cost": RETURN_COST, "call_cost": CALL_COST,
                 "train_rows": len(d), "train_end": str(d.order_placed_at.max())}, "model/model.joblib")
    json.dump(out, open("evidence/metrics.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float)); print(lift.round(3).to_string()); print(cal.round(3).to_string()); print(pd.DataFrame(seg).round(3).to_string())
