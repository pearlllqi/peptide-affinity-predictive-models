#!/usr/bin/env python3
"""Run LSTM and histogram gradient-boosted trees with collaborator CfC logic."""

from pathlib import Path
import argparse
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
SEEDS = [42, 43, 44]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def encode(sequences):
    x = np.zeros((len(sequences), 20, 20), np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq): x[i, j, AA_INDEX[aa]] = 1.0
    return x


def prepare(df, seed):
    data = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    x = encode(data.Sequence.to_numpy())
    y = data.Delta_G.to_numpy(np.float32)
    ntr, nv = int(.70 * len(data)), int(.15 * len(data))
    sp = {"train": slice(0,ntr), "validation": slice(ntr,ntr+nv), "test": slice(ntr+nv,len(data))}
    mean, std = float(y[sp["train"]].mean()), float(y[sp["train"]].std())
    ys = ((y - mean) / (std + 1e-12)).astype(np.float32)
    return data, x, y, ys, sp, mean, std


def metrics(y, p):
    return {"MAE": mean_absolute_error(y,p), "MSE": mean_squared_error(y,p),
            "RMSE": np.sqrt(mean_squared_error(y,p)), "R2": r2_score(y,p),
            "Pearson": pearsonr(y,p).statistic, "Spearman": spearmanr(y,p).statistic}


class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=64, batch_first=True)
        self.drop1 = nn.Dropout(.2)
        self.dense = nn.Linear(64,16)
        self.drop2 = nn.Dropout(.2)
        self.out = nn.Linear(16,1)
    def forward(self,x):
        _,(h,_) = self.lstm(x)
        z = self.drop1(h[-1]); z = torch.tanh(self.dense(z)); z = self.drop2(z)
        return self.out(z).squeeze(-1)


@torch.no_grad()
def predict_lstm(model, x, batch=512):
    model.eval(); out=[]
    for i in range(0,len(x),batch): out.append(model(torch.from_numpy(x[i:i+batch])).cpu().numpy())
    return np.concatenate(out)


def train_lstm(xtr,ytr,xv,yv,seed,epochs,batch):
    set_seed(seed); model=LSTMRegressor()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    loss_fn=nn.HuberLoss(delta=.5)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=batch,shuffle=True,generator=generator,num_workers=0)
    history=[]
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True); pred=model(xb); loss=loss_fn(pred,yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(loss.item())
        vp=predict_lstm(model,xv); vl=float(loss_fn(torch.from_numpy(vp),torch.from_numpy(yv)).item())
        history.append({"Epoch":epoch,"loss":float(np.mean(losses)),"val_loss":vl})
        if epoch==1 or epoch%10==0 or epoch==epochs: print(f"  LSTM epoch {epoch:3d}: train={np.mean(losses):.5f} val={vl:.5f}",flush=True)
    return model,pd.DataFrame(history)


def gbt_grid():
    for lr in (.03,.05):
        for leaves in (15,31):
            for iterations in (200,500):
                yield {"learning_rate":lr,"max_leaf_nodes":leaves,"max_iter":iterations}


def save_predictions(path,data,sl,y,p,dataset,model,seed):
    out=pd.DataFrame({"Sequence":data.Sequence.iloc[sl].to_numpy(),"Actual_Delta_G":y,"Predicted_Delta_G":p})
    out["Residual"]=out.Predicted_Delta_G-out.Actual_Delta_G; out["Abs_Error"]=out.Residual.abs()
    out["Dataset"]=dataset; out["Model"]=model; out["Seed"]=seed; out.to_csv(path,index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/htsk.csv'); ap.add_argument('--out',default='outputs/current_logic_lstm_gbt'); ap.add_argument('--epochs',type=int,default=70); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--datasets',nargs='+',default=['primary','full']); ap.add_argument('--models',nargs='+',default=['lstm','gbt']); args=ap.parse_args()
    torch.set_num_threads(1); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.data); raw.Sequence=raw.Sequence.astype(str).str.strip().str.upper()
    sets={'primary_-16_to_-12':raw.loc[raw.Delta_G.between(-16,-12)].copy(),'full':raw}
    selected=['primary_-16_to_-12' if x=='primary' else x for x in args.datasets]
    results=[]; tuning=[]
    for dataset in selected:
        for seed in SEEDS:
            print(f"\n=== {dataset} seed {seed} ===",flush=True)
            data,x,y,ys,sp,mean,std=prepare(sets[dataset],seed); tr,va,te=sp['train'],sp['validation'],sp['test']
            if 'lstm' in args.models:
                start=time.time(); model,hist=train_lstm(x[tr],ys[tr],x[va],ys[va],seed,args.epochs,args.batch_size)
                hist['Dataset']=dataset; hist['Model']='LSTM'; hist['Seed']=seed; hist.to_csv(out/f'{dataset}_lstm_seed{seed}_history.csv',index=False)
                pred=predict_lstm(model,x[te])*std+mean; met=metrics(y[te],pred); met.update({'Dataset':dataset,'Model':'LSTM','Seed':seed})
                results.append(met); torch.save({'state_dict':model.state_dict(),'target_mean':mean,'target_std':std},out/f'{dataset}_lstm_seed{seed}.pt'); save_predictions(out/f'{dataset}_lstm_seed{seed}_test_predictions.csv',data,te,y[te],pred,dataset,'LSTM',seed)
                print('  LSTM',met,f"({time.time()-start:.1f}s)",flush=True)
            if 'gbt' in args.models:
                best=None
                xtr=x[tr].reshape(len(x[tr]),-1); xv=x[va].reshape(len(x[va]),-1); xt=x[te].reshape(len(x[te]),-1)
                for params in gbt_grid():
                    model=HistGradientBoostingRegressor(loss='squared_error',l2_regularization=1e-4,random_state=seed,early_stopping=False,**params)
                    model.fit(xtr,ys[tr]); pv=model.predict(xv)*std+mean; vm=metrics(y[va],pv)
                    row={'Dataset':dataset,'Seed':seed,**params,**{f'Validation_{k}':v for k,v in vm.items()}}; tuning.append(row)
                    if best is None or vm['MSE']<best[0]: best=(vm['MSE'],params,model)
                _,params,model=best; pred=model.predict(xt)*std+mean; met=metrics(y[te],pred); met.update({'Dataset':dataset,'Model':'GradientBoostedTrees','Seed':seed,**params})
                results.append(met); joblib.dump({'model':model,'target_mean':mean,'target_std':std,'params':params},out/f'{dataset}_gbt_seed{seed}.joblib'); save_predictions(out/f'{dataset}_gbt_seed{seed}_test_predictions.csv',data,te,y[te],pred,dataset,'GradientBoostedTrees',seed)
                print('  GBT',met,flush=True)
            pd.DataFrame(results).to_csv(out/'per_seed_metrics_partial.csv',index=False); pd.DataFrame(tuning).to_csv(out/'gbt_validation_tuning_partial.csv',index=False)
    per=pd.DataFrame(results); per.to_csv(out/'per_seed_metrics.csv',index=False); pd.DataFrame(tuning).to_csv(out/'gbt_validation_tuning.csv',index=False)
    cols=['MAE','MSE','RMSE','R2','Pearson','Spearman']; summary=per.groupby(['Dataset','Model'])[cols].agg(['mean','std']); summary.columns=[f'{a}_{b}' for a,b in summary.columns]; summary.reset_index().to_csv(out/'three_seed_summary.csv',index=False)
    with (out/'run_config.json').open('w') as f: json.dump(vars(args),f,indent=2)
    print('\nFINAL\n',summary,flush=True)

if __name__=='__main__': main()
