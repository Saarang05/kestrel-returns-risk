"""Feature engineering shared by training, batch scoring and the service.

Only columns that exist at dispatch time are used. Deliberately NOT used:
  pickup_scheduled_at, last_service_event_type  -> written after a return is approved / after delivery (leakage)
  source                                        -> ingestion artefact (partner_feed duplicates)
  signup_date                                   -> ~19% of rows have signup after the order (unreliable)
  delivery_note                                 -> no signal; also contains injected instructions in 4 rows
"""
import numpy as np, pandas as pd

CAT = ["family", "sku", "sales_channel", "payment_mode", "state", "pin3", "price_tier"]
NUM = ["discount_pct", "qty", "list_price_inr", "order_value_inr", "promised_delivery_days", "is_gift_n",
       "customer_prior_orders", "customer_prior_returns", "prior_return_rate", "shield_n", "no_address",
       "hi_discount", "long_promise", "cod_n"]
FEATURES = CAT + NUM

def load_ref(data_dir="data"):
    cu = pd.read_csv(f"{data_dir}/customers.csv")
    pr = pd.read_csv(f"{data_dir}/products.csv")
    return cu, pr

def clean_orders(df, pr):
    """Fix known data defects. Returns a new frame."""
    df = df.copy()
    df["list_price_inr"] = df.sku.map(pr.set_index("sku").list_price_inr)
    expected = df.list_price_inr * df.qty * (1 - df.discount_pct / 100)
    ratio = df.order_value_inr / expected
    # Oct-2025 gateway stored values x100 (paise). Rescale any row that is ~100x its expected value.
    fix = (ratio - 100).abs() < 1
    df.loc[fix, "order_value_inr"] = df.loc[fix, "order_value_inr"] / 100
    df["order_value_fixed"] = fix
    return df

def build(df, cu, pr):
    df = clean_orders(df, pr)
    df = df.merge(cu[["customer_id", "state", "shield_member"]], on="customer_id", how="left")
    df["family"] = df.sku.map(pr.set_index("sku").family)
    pin = df.delivery_pincode.astype(str).str.zfill(6)
    df["no_address"] = (pin == "000000").astype(int)          # system default, not a real location
    df["pin3"] = np.where(df.no_address == 1, "NA", pin.str[:3])
    df["is_gift_n"] = (df.is_gift == "Y").astype(int)
    df["shield_n"] = (df.shield_member == "Y").astype(int)
    df["cod_n"] = (df.payment_mode == "cod").astype(int)
    df["prior_return_rate"] = df.customer_prior_returns / df.customer_prior_orders.clip(lower=1)
    df["hi_discount"] = (df.discount_pct >= 20).astype(int)
    df["long_promise"] = (df.promised_delivery_days >= 7).astype(int)
    df["price_tier"] = df.sku.str[-2:]
    df["state"] = df.state.fillna("UNK")
    return df
