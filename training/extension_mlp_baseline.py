#!/usr/bin/env python3
"""Extension-only PyTorch MLP (256->64) with validation early stopping."""

from pathlib import Path
import copy, json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT=Path("./Extension_Kd-trunc.xlsx"); OUT=Path("Extension_MLP_results")
SEED=42; WT="IETITIYNYKKAADHFSMSM"; AA="ACDEFGHIKLMNPQRSTVWY"
R=1.987204258e-3; T=298.15; BATCH=64; EPOCHS=300; PATIENCE=25

def prepare():
    d=pd.read_excel(INPUT); s=d["Sequence"].astype("string").str.strip().str.upper(); k=pd.to_numeric(d["Kd"],errors="coerce")
    ok=s.str.fullmatch(f"[{AA}]{{20}}",na=False)&k.gt(0)&np.isfinite(k)
    d=pd.DataFrame({"Sequence":s[ok],"Kd_pM":k[ok]}).groupby("Sequence",as_index=False).Kd_pM.median()
    d["Delta_G"]=R*T*np.log(d.Kd_pM*1e-12); return d.sample(frac=1,random_state=SEED).reset_index(drop=True)
def encode(ss):
    ix={a:i for i,a in enumerate(AA)}; x=np.zeros((len(ss),20,21),np.float32)
    for n,s in enumerate(ss):
        for p,a in enumerate(s): x[n,p,ix[a]]=1; x[n,p,20]=a!=WT[p]
    return x.reshape(len(ss),-1)
class MLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(420,256),nn.ReLU(),nn.Dropout(.30),nn.Linear(256,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(1)
def scores(y,p):
    mse=mean_squared_error(y,p); return {"R2":r2_score(y,p),"MSE":mse,"RMSE":float(np.sqrt(mse)),"MAE":mean_absolute_error(y,p),"Pearson":float(np.corrcoef(y,p)[0,1]),"Spearman":float(pd.Series(y).corr(pd.Series(p),method="spearman"))}
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(int(__import__('os').environ.get('CFC_NUM_THREADS','1'))); OUT.mkdir(exist_ok=True)
    d=prepare(); n=len(d); a=int(.70*n); b=a+int(.15*n); d["Split"]=np.where(np.arange(n)<a,"train",np.where(np.arange(n)<b,"validation","test")); d.to_csv(OUT/"prepared_data_and_split.csv",index=False)
    x=encode(d.Sequence); y=d.Delta_G.to_numpy(np.float32); mu=float(y[:a].mean()); sd=float(y[:a].std()); z=(y-mu)/sd
    tx=torch.from_numpy(x); ty=torch.from_numpy(z); gen=torch.Generator().manual_seed(SEED)
    loader=DataLoader(TensorDataset(tx[:a],ty[:a]),batch_size=BATCH,shuffle=True,generator=gen,num_workers=0)
    model=MLP(); opt=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=3e-4); loss=nn.MSELoss(); best=np.inf; wait=0; hist=[]; state=None
    for epoch in range(1,EPOCHS+1):
        model.train(); vals=[]
        for xb,yb in loader: opt.zero_grad(); l=loss(model(xb),yb); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); vals.append(l.item())
        model.eval()
        with torch.no_grad(): vl=loss(model(tx[a:b]),ty[a:b]).item()
        hist.append({"Epoch":epoch,"Train_MSE_scaled":float(np.mean(vals)),"Validation_MSE_scaled":vl})
        if vl < best-1e-6: best=vl; state=copy.deepcopy(model.state_dict()); wait=0
        else: wait+=1
        if wait>=PATIENCE: break
    model.load_state_dict(state); torch.save({"state_dict":state,"target_mean":mu,"target_std":sd,"seed":SEED},OUT/"mlp_model.pt"); pd.DataFrame(hist).to_csv(OUT/"training_history.csv",index=False)
    rec=[]; met=[]; model.eval()
    with torch.no_grad():
        for name,sl in [("validation",slice(a,b)),("test",slice(b,n))]:
            p=model(tx[sl]).numpy()*sd+mu; yy=y[sl]; met.append({"Split":name,**scores(yy,p)}); q=d.iloc[sl][["Sequence","Kd_pM","Delta_G"]].copy(); q.insert(0,"Split",name); q.rename(columns={"Delta_G":"Actual_Delta_G"},inplace=True); q["Predicted_Delta_G"]=p; q["Residual"]=p-yy; q["Abs_Error"]=abs(p-yy); rec.append(q)
    pd.concat(rec).to_csv(OUT/"holdout_predictions.csv",index=False); pd.DataFrame(met).to_csv(OUT/"metrics.csv",index=False); (OUT/"run_config.json").write_text(json.dumps({"seed":SEED,"target_mean":mu,"target_std":sd,"best_epoch":int(np.argmin([h['Validation_MSE_scaled'] for h in hist])+1)},indent=2)); print(pd.DataFrame(met).to_string(index=False)); print(f"Saved to {OUT}")
if __name__=="__main__": main()
