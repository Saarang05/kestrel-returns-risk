import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"src")
from features import *
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
cu,pr=load_ref()
tr=pd.read_csv("data/train.csv",parse_dates=["order_placed_at"]).drop_duplicates("order_id").sort_values("order_placed_at").reset_index(drop=True)
d=build(tr,cu,pr); y=d.returned.values
d["log_val"]=np.log1p(d.order_value_inr); d["log_price"]=np.log(d.list_price_inr)
d["shield_cod"]=d.shield_n*d.cod_n; d["shield_x_prior"]=d.shield_n*d.customer_prior_returns
d["disc_sq"]=d.discount_pct**2/100; d["prior_orders_c"]=d.customer_prior_orders.clip(upper=6); d["prior_ret_c"]=d.customer_prior_returns.clip(upper=4)
base_num=["discount_pct","qty","log_price","promised_delivery_days","is_gift_n","prior_orders_c","prior_ret_c","prior_return_rate","shield_n","no_address","cod_n"]
sets={
 "LR base":(["family","sales_channel","payment_mode","state","price_tier"],base_num,0.3),
 "LR base C=0.1":(["family","sales_channel","payment_mode","state","price_tier"],base_num,0.1),
 "LR +sku":(["sku","sales_channel","payment_mode","state"],base_num,0.3),
 "LR +pin3":(["family","sales_channel","payment_mode","state","price_tier","pin3"],base_num,0.3),
 "LR +interact":(["family","sales_channel","payment_mode","state","price_tier"],base_num+["shield_cod","shield_x_prior","disc_sq"],0.3),
 "LR no-state":(["family","sales_channel","payment_mode","price_tier"],base_num,0.3),
}
folds=[("2025-10-01","2026-01-01"),("2026-01-01","2026-04-01"),("2026-04-01","2026-07-01")]
def lr(cat,num,C):
    ct=ColumnTransformer([("c",OneHotEncoder(handle_unknown="ignore",min_frequency=20),cat),("n",StandardScaler(),num)])
    return make_pipeline(ct,LogisticRegression(C=C,max_iter=3000))
res={}
for name,(cat,num,C) in sets.items():
    a=[];b=[]
    for s,e in folds:
        itr=(d.order_placed_at<s).values; iva=((d.order_placed_at>=s)&(d.order_placed_at<e)).values
        m=lr(cat,num,C).fit(d.loc[itr,cat+num],y[itr]); p=m.predict_proba(d.loc[iva,cat+num])[:,1]
        a.append(roc_auc_score(y[iva],p)); b.append(average_precision_score(y[iva],p))
    print(f"{name:16s} AUC {np.mean(a):.4f} {np.round(a,3)} PR {np.mean(b):.4f} {np.round(b,3)}")
# HGB regularised
Xcols=["family","sales_channel","payment_mode","state","price_tier"]+base_num
X=d[Xcols].copy()
for c in ["family","sales_channel","payment_mode","state","price_tier"]: X[c]=X[c].astype("category")
for lr_,it,leaf in [(0.03,150,6),(0.03,300,8),(0.05,100,4)]:
    a=[];b=[]
    for s,e in folds:
        itr=(d.order_placed_at<s).values; iva=((d.order_placed_at>=s)&(d.order_placed_at<e)).values
        m=HistGradientBoostingClassifier(max_iter=it,learning_rate=lr_,max_leaf_nodes=leaf,min_samples_leaf=60,l2_regularization=5,categorical_features="from_dtype",random_state=0).fit(X[itr],y[itr]); p=m.predict_proba(X[iva])[:,1]
        a.append(roc_auc_score(y[iva],p)); b.append(average_precision_score(y[iva],p))
    print(f"HGB {lr_} {it} {leaf}  AUC {np.mean(a):.4f} PR {np.mean(b):.4f}")
