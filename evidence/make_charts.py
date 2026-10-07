import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
lift=pd.read_csv("evidence/lift_and_rupees.csv"); cal=pd.read_csv("evidence/calibration.csv")
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
ax[0].plot(cal.mean_pred*100,cal.actual*100,"o-",label="model"); ax[0].plot([0,45],[0,45],"--",c="grey",label="perfect"); ax[0].set(title="Calibration (out-of-time, deciles)",xlabel="predicted return %",ylabel="actual return %"); ax[0].legend()
ax[1].bar(lift.flag_top_pct.astype(str),lift.precision*100); ax[1].axhline(11.3,c="r",ls="--",label="base rate 11.3%"); ax[1].set(title="Share of flagged orders actually returned",xlabel="flag top X% of orders",ylabel="% returned"); ax[1].legend()
ax[2].plot(lift.flag_top_pct,lift.call_net_per_month_rs/1000,"o-",label="calls (Rs1,150/return)"); ax[2].plot(lift.flag_top_pct,[r*700*f/100/1000 for r,f in zip(lift.call_net_if_return_cost_600,lift.flag_top_pct)],"s--",label="calls if return costs Rs600")
ax[2].axhline(0,c="grey"); ax[2].set(title="Net saving per month, 700 orders (Rs thousand)",xlabel="call top X% of orders",ylabel="Rs '000"); ax[2].legend()
plt.tight_layout(); plt.savefig("evidence/evidence.png",dpi=130)
