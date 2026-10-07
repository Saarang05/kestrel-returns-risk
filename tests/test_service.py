"""Run: python -m unittest discover tests   (from repo root)"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from service import app, score_record

OK = {"sku": "KH-RV-02", "sales_channel": "marketplace", "payment_mode": "cod", "shield_member": "N", "discount_pct": 25, "qty": 1,
      "order_value_inr": 16499.25, "promised_delivery_days": 8, "customer_prior_orders": 3, "customer_prior_returns": 2}

class T(unittest.TestCase):
    def setUp(self): self.c = app.test_client()
    def test_health_needs_no_key(self): self.assertFalse(self.c.get("/health").get_json()["paid_api_required"])
    def test_score_shape(self):
        j = self.c.post("/score", json=OK).get_json()
        for k in ["return_probability", "risk_band", "action", "reasons", "warnings"]: self.assertIn(k, j)
        self.assertTrue(0 <= j["return_probability"] <= 1); self.assertTrue(j["reasons"])
    def test_leaky_fields_ignored(self):
        a = self.c.post("/score", json=OK).get_json()["return_probability"]
        b = self.c.post("/score", json={**OK, "pickup_scheduled_at": "2026-07-03 10:00", "last_service_event_type": "REVERSE_PICKUP"}).get_json()
        self.assertEqual(a, b["return_probability"]); self.assertEqual(len(b["warnings"]), 2)
    def test_missing_fields_polite(self):
        r = self.c.post("/score", json={"sku": "KH-RV-02"}); self.assertEqual(r.status_code, 400); self.assertIn("missing", r.get_json())
    def test_garbage_polite(self): self.assertEqual(self.c.post("/score", data="nope").status_code, 400)
    def test_unknown_sku(self): self.assertEqual(self.c.post("/score", json={**OK, "sku": "X"}).status_code, 400)
    def test_x100_value_rescaled(self):
        j = self.c.post("/score", json={**OK, "order_value_inr": OK["order_value_inr"] * 100}).get_json()
        self.assertTrue(any("100x" in w for w in j["warnings"]))
    def test_shield_never_held(self):
        j = self.c.post("/score", json={**OK, "shield_member": "Y"}).get_json(); self.assertIn("do NOT hold", j["advice"])
if __name__ == "__main__": unittest.main()
