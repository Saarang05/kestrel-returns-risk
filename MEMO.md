---
title: "Returns model: what to do next week"
---

**To:** Ritu Deshpande, Head of D2C Operations  **Cc:** Farhan Sheikh, Meenal Joshi, Tanmay Kulkarni

## The decision
**Do not hold orders. Phone the riskiest one in seven before dispatch, and ship everything else.** Each month that is about 105 of your 700 orders. The model ranks every order by its chance of coming back and tells the call team why (cash on delivery, earlier returns, long delivery promise, product type).

## The number
**95% accuracy is not possible, and it is the wrong yardstick.** Shipping everything and never flagging anything is already 88.7% "accurate", because only 11 in 100 orders come back. The best this model can do is 89.5%. A version that does hit 99% exists, but it uses the pickup-booking fields, which are filled in *after* a customer's return is approved, so it cannot be used at dispatch.

What the model does deliver: of the orders it flags, **1 in 3 really is returned**, against 1 in 9 across all orders. That is three times better than picking at random. It still misses more than half of all returns, mainly prepaid first-time buyers with no warning signs. It was tested only on orders it had never seen, month by month.

## The rupees
| | Per month (700 orders) |
|---|---|
| Returns today: about 79 x Rs 1,150 (Finance figure) | about Rs 91,000 |
| Call the top 105 orders: 105 x Rs 45 | Rs 4,700 |
| Returns avoided (1 in 3 flagged come back; spring pilot prevented 35%): about 13 x Rs 1,150 | Rs 14,600 |
| **Net saving** | **about Rs 10,000 (about Rs 1.2 lakh a year)** |

- This is real but small. Calls still pay as long as they prevent just 11% of returns, well below what the pilot saw.
- At your Rs 600 per return the saving drops to about Rs 3,000 a month. Please settle on one cost; Finance uses Rs 1,150.
- **Holding does not pay.** Only 12% of held orders cancel, and about 45% of the orders we would hold are Shield members, your best repeat buyers. At an assumed Rs 1,000 margin per lost sale, holding loses money. Shield members return twice as often (19% vs 9%) because it is free, so call them, never hold them.

## What to do next week
1. **Start calling, as a test.** For two weeks, call a random half of the flagged orders and not the other half. That is the only way to confirm the pilot's 35%. Give Finance the result.
2. **Reset the board message.** Not "95%", but "we find returns three times better than chance and save about Rs 10,000 a month, with the real prize elsewhere".
3. **Look at the bigger levers.** Cash on delivery orders come back at 19% against about 8% for prepaid. Robot vacuums are at 19% and water purifiers 15%. Orders promised 7+ days out and discounts of 20%+ also return more.
4. **Ask Tanmay** to confirm the October 2025 payment-gateway values (they were stored 100 times too large; we corrected them) and to stop post-return fields appearing in the dispatch export.
