#!/usr/bin/env python3
"""Extension-only PyTorch CNN: Conv1d 64/64, global max pool, Dense32."""

from pathlib import Path
import copy, json, random, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT=Path(os.environ.get("EXTENSION_INPUT", "./Extension_Kd-trunc.xlsx")); OUT=Path(os.environ.get("EXTENSION_CNN_OUT", "Extension_CNN_results"))
SEED=42; WT="IETITIYNYKKAADHFSMSM"; AA="ACDEFGHIKLMNPQRSTVWY"; R=1.987204258e-3; T=298.15
BATCH=64; EPOCHS=300; PATIENCE=25
def prepare():
    d=pd.read_excel(INPUT); s=d.Sequence.astype("string").str.strip().str.upper(); k=pd.to_numeric(d.Kd,errors="coerce"); ok=s.str.fullmatch(f"[{AA}]{{20}}",na=False)&k.gt(0)&np.isfinite(k); d=pd.DataFrame({"Sequence":s[ok],"Kd_pM":k[ok]}).groupby("Sequence",as_index=False).Kd_pM.median(); d["Delta_G"]=R*T*np.log(d.Kd_pM*1e-12); return d.sample(frac=1,random_state=SEED).reset_index(drop=True)
def encode(ss):
    ix={a:i for i,a in enumerate(AA)}; x=np.zeros((len(ss),20,21),np.float32)
    for n,s in enumerate(ss):
        for p,a in enumerate(s): x[n,p,ix[a]]=1; x[n,p,20]=a!=WT[p]
    return x
class CNN(nn.Module):
    def __init__(self):
        super().__init__(); self.conv=nn.Sequential(nn.Conv1d(21,64,3,padding=1),nn.ReLU(),nn.Conv1d(64,64,3,padding=1),nn.ReLU()); self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.20),nn.Linear(32,1))
    def forward(self,x): x=self.conv(x.transpose(1,2)); return self.head(torch.amax(x,dim=2)).squeeze(1)
def scores(y,p):
    mse=mean_squared_error(y,p); return {"R2":r2_score(y,p),"MSE":mse,"RMSE":float(np.sqrt(mse)),"MAE":mean_absolute_error(y,p),"Pearson":float(np.corrcoef(y,p)[0,1]),"Spearman":float(pd.Series(y).corr(pd.Series(p),method="spearman"))}
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(int(os.environ.get("CFC_NUM_THREADS","1"))); OUT.mkdir(exist_ok=True)
    d=prepare(); n=len(d); a=int(.70*n); b=a+int(.15*n); d["Split"]=np.where(np.arange(n)<a,"train",np.where(np.arange(n)<b,"validation","test")); d.to_csv(OUT/"prepared_data_and_split.csv",index=False)
    x=encode(d.Sequence); y=d.Delta_G.to_numpy(np.float32); mu=float(y[:a].mean()); sd=float(y[:a].std()); z=(y-mu)/sd; tx=torch.from_numpy(x); ty=torch.from_numpy(z); gen=torch.Generator().manual_seed(SEED); loader=DataLoader(TensorDataset(tx[:a],ty[:a]),batch_size=BATCH,shuffle=True,generator=gen,num_workers=0)
    model=CNN(); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4); criterion=nn.MSELoss(); best=np.inf; wait=0; state=None; hist=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); ls=[]
        for xb,yb in loader: opt.zero_grad(); l=criterion(model(xb),yb); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); ls.append(l.item())
        model.eval()
        with torch.no_grad():
            train_eval=criterion(model(tx[:a]),ty[:a]).item()
            vl=criterion(model(tx[a:b]),ty[a:b]).item()
            test_eval=criterion(model(tx[b:]),ty[b:]).item()
        hist.append({"Epoch":epoch,"Train_MSE_scaled":train_eval,"Validation_MSE_scaled":vl,"Test_MSE_scaled_diagnostic_only":test_eval})
        if vl<best-1e-6: best=vl; state=copy.deepcopy(model.state_dict()); wait=0
        else: wait+=1
        if wait>=PATIENCE: break
    model.load_state_dict(state); torch.save({"state_dict":state,"target_mean":mu,"target_std":sd,"seed":SEED},OUT/"cnn_model.pt"); history=pd.DataFrame(hist); history.to_csv(OUT/"training_history.csv",index=False)
    best_epoch=int(history["Validation_MSE_scaled"].idxmin()+1)
    plt.figure(figsize=(8,5.5)); plt.plot(history.Epoch,history.Train_MSE_scaled,label="Training MSE",linewidth=2); plt.plot(history.Epoch,history.Validation_MSE_scaled,label="Validation MSE",linewidth=2); plt.plot(history.Epoch,history.Test_MSE_scaled_diagnostic_only,label="Test MSE (diagnostic only)",linewidth=2); plt.axvline(best_epoch,color="black",linestyle="--",alpha=.65,label=f"Best validation epoch = {best_epoch}"); plt.xlabel("Epoch"); plt.ylabel("Scaled Delta_G MSE"); plt.title("Extension CNN: train, validation, and test loss"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"cnn_train_validation_test_loss_curve.png",dpi=250); plt.close()
    rec=[]; met=[]; model.eval()
    with torch.no_grad():
        for name,sl in [("validation",slice(a,b)),("test",slice(b,n))]:
            p=model(tx[sl]).numpy()*sd+mu; yy=y[sl]; met.append({"Split":name,**scores(yy,p)}); q=d.iloc[sl][["Sequence","Kd_pM","Delta_G"]].copy(); q.insert(0,"Split",name); q.rename(columns={"Delta_G":"Actual_Delta_G"},inplace=True); q["Predicted_Delta_G"]=p; q["Residual"]=p-yy; q["Abs_Error"]=abs(p-yy); rec.append(q)
    pd.concat(rec).to_csv(OUT/"holdout_predictions.csv",index=False); pd.DataFrame(met).to_csv(OUT/"metrics.csv",index=False); (OUT/"run_config.json").write_text(json.dumps({"seed":SEED,"target_mean":mu,"target_std":sd,"best_epoch":best_epoch},indent=2)); print(pd.DataFrame(met).to_string(index=False)); print(f"Saved to {OUT}")
if __name__=="__main__": main()
