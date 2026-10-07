# 3-minute screen recording: talk track (no slides)
Record your screen with the repo open. Aim for 2:45. Say these in your own words.

**0:00 - 0:25  What I tried.** Show `data/` and `src/train.py`. "I ran a quick test with every column in the file, including the service and pickup fields. It scored 99% accuracy." Show the leaky demo line in `evidence/metrics.json` (`leaky_demo`: AUC 0.997). "That is too good. The pickup and service-event columns are written after a return is approved, and the test file doesn't have them."

**0:25 - 1:05  What I changed.** Show `src/features.py` top docstring. "I removed those columns, de-duplicated 651 partner-feed rows, and fixed 700 October values that were 100x too big." Open `EVIDENCE.md`: "I validated out of time, three quarters, never random. AUC is 0.78. Accuracy is 89.5%, but doing nothing is 88.7%, so 95% isn't reachable and accuracy is the wrong yardstick."

**1:05 - 1:40  What I threw away.** "Gradient boosting lost to logistic regression out of time, so it's gone, with the extra features. I dropped delivery notes: no signal, and four contained instructions aimed at AI tools, which I ignored. And I dropped holding orders." Show the chart `evidence/evidence.png` and the hold-vs-call numbers.

**1:40 - 2:35  The service.** `python src/service.py`, open the screen. Click the high-risk example: point at the 90% score, the action, the reasons. Click the Shield example: point at "do not hold". Paste an order value 100x too big or add a `pickup_scheduled_at` via curl and show the warning. "No API key, costs Rs 0 per prediction."

**2:35 - 2:55  The answer for Ritu.** Show `MEMO.pdf`. "Call about 105 orders a month; net about Rs 10,000 a month; the real test is a two-week randomised call trial."
