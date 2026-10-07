import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"src")
from features import *
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, log_loss, brier_score_loss
cu,pr=load_ref()
tr=pd.read_csv("data/train.csv",parse_dates=["order_placed_at"])
tr=tr.drop_duplicates("order_id").sort_values("order_placed_at").reset_index(drop=True)
raw=tr.copy()
d=build(tr,cu,pr)
y=d.returned.values
cut=pd.Timestamp("2026-04-01"); 
itr=(d.order_placed_at<cut).values; iva=~itr
print("train",itr.sum(),"val",iva.sum(),"val base",y[iva].mean())
def ev(name,p):
    print(f"{name:34s} AUC {roc_auc_score(y[iva],p):.4f} PR-AUC {average_precision_score(y[iva],p):.4f} acc@0.5 {accuracy_score(y[iva],p>0.5):.4f} brier {brier_score_loss(y[iva],p):.4f}")
print("always-0 accuracy", 1-y[iva].mean())
# 1 leaky model for demonstration
leak=d.copy(); leak["pk"]=leak.pickup_scheduled_at.notna().astype(int)
for ev_ in ["NONE","DEMO_DONE","INSTALL_DONE","REVERSE_PICKUP","TECH_VISIT"]: leak["ev_"+ev_]=(leak.last_service_event_type==ev_).astype(int)
Xl=leak[NUM+["pk"]+["ev_"+e for e in ["NONE","DEMO_DONE","INSTALL_DONE","REVERSE_PICKUP","TECH_VISIT"]]]
m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05).fit(Xl[itr],y[itr]); ev("LEAKY (pickup+service events)",m.predict_proba(Xl[iva])[:,1])
# Legit
X=d[FEATURES].copy()
for c in CAT: X[c]=X[c].astype("category")
m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.03,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.0,categorical_features="from_dtype",random_state=0).fit(X[itr],y[itr]); p_hgb=m.predict_proba(X[iva])[:,1]; ev("HGB legit",p_hgb)
ct=ColumnTransformer([("c",OneHotEncoder(handle_unknown="ignore",min_frequency=20),CAT),("n",StandardScaler(),NUM)])
lr=make_pipeline(ct,LogisticRegression(C=0.3,max_iter=2000)).fit(d.loc[itr,FEATURES],y[itr]); p_lr=lr.predict_proba(d.loc[iva,FEATURES])[:,1]; ev("LogReg legit",p_lr)
ev("blend",0.5*p_hgb+0.5*p_lr)
# simple heuristics
ev("prior_returns only",d.loc[iva,"customer_prior_returns"].values/10+0.0)
ev("cod only",d.loc[iva,"cod_n"].values*1.0)
# No-fix of order value
d2=d.copy(); 
# without scaling fix (raw values)
d3=build(raw,cu,pr)  # fixed already; emulate unfixed
d3["order_value_inr"]=raw.order_value_inr.values
X3=d3[FEATURES].copy()
for c in CAT: X3[c]=X3[c].astype("category")
m3=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.03,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.0,categorical_features="from_dtype",random_state=0).fit(X3[itr],y[itr]); ev("HGB w/o x100 fix",m3.predict_proba(X3[iva])[:,1])
# dedup effect: random split with dupes leaks
