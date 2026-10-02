#!/usr/bin/env python3
from pathlib import Path
import random, os
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ncps.torch import CfC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/htsk.csv'
REF=ROOT/'data/ddim_15_sequences.csv'
SAVED=ROOT/'outputs/current_logic_lstm_gbt'
OUT=ROOT/'outputs/mixed_train_ddim15'
OUT.mkdir(parents=True,exist_ok=True)
AA='ACDEFGHIKLMNPQRSTVWY'; IDX={a:i for i,a in enumerate(AA)}; SEEDS=(42,43,44)

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def encode(seqs):
    x=np.zeros((len(seqs),20,20),np.float32)
    for i,s in enumerate(seqs):
        for j,a in enumerate(s): x[i,j,IDX[a]]=1
    return x

class CurrentCfC(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.LayerNorm(20); self.d1=nn.Linear(20,32)
        self.c1=CfC(input_size=32,units=32,return_sequences=True,batch_first=True)
        self.dp1=nn.Dropout(.2); self.d2=nn.Linear(32,16)
        self.c2=CfC(input_size=16,units=16,return_sequences=False,batch_first=True)
        self.dp2=nn.Dropout(.2); self.out=nn.Linear(16,1)
    def forward(self,x):
        x=torch.tanh(self.d1(self.norm(x))); x,_=self.c1(x); x=self.dp1(x)
        x=torch.tanh(self.d2(x)); x,_=self.c2(x); return self.out(self.dp2(x)).squeeze(-1)

class CNN(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.LayerNorm(20)
        self.conv=nn.Sequential(nn.Conv1d(20,64,3,padding=1),nn.ReLU(),nn.Conv1d(64,64,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
    def forward(self,x): return self.head(torch.amax(self.conv(self.norm(x).transpose(1,2)),dim=2)).squeeze(-1)

class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__(); self.lstm=nn.LSTM(20,64,batch_first=True); self.dp1=nn.Dropout(.2)
        self.dense=nn.Linear(64,16); self.dp2=nn.Dropout(.2); self.out=nn.Linear(16,1)
    def forward(self,x):
        _,(h,_)=self.lstm(x); return self.out(self.dp2(torch.tanh(self.dense(self.dp1(h[-1]))))).squeeze(-1)

@torch.no_grad()
def pred(model,x,b=1024):
    model.eval(); return np.concatenate([model(torch.from_numpy(x[i:i+b])).numpy() for i in range(0,len(x),b)])

def train(cls,x,y,seed):
    seed_all(seed); m=cls(); opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4); loss=nn.HuberLoss(delta=.5)
    g=torch.Generator().manual_seed(seed); loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),64,shuffle=True,generator=g,num_workers=0)
    for ep in range(1,71):
        m.train()
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True); z=loss(m(xb),yb); z.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if ep==1 or ep%10==0: print(cls.__name__,seed,ep,flush=True)
    return m

def metrics(y,p):
    mse=mean_squared_error(y,p); return dict(R2=r2_score(y,p),RMSE=np.sqrt(mse),MSE=mse,MAE=mean_absolute_error(y,p),Pearson=pearsonr(y,p).statistic,Spearman=spearmanr(y,p).statistic)

torch.set_num_threads(int(os.environ.get('MODEL_NUM_THREADS','2')))
raw=pd.read_csv(DATA); raw.Sequence=raw.Sequence.astype(str).str.strip().str.upper()
raw=raw.loc[raw.Sequence.str.fullmatch(f'[{AA}]{{20}}',na=False)&pd.to_numeric(raw.Delta_G,errors='coerce').between(-16,-12)].copy()
ref=pd.read_csv(REF)
ref['Experimental_Delta_G_Calculated']=1.987204258e-3*298.15*np.log(ref.Experimental_Kd_pM.to_numpy(float)*1e-12)
xd=encode(ref.Model_Input_20aa.to_numpy()); actual=ref.Experimental_Delta_G_Calculated.to_numpy(float)
rows=[]
for seed in SEEDS:
    d=raw.sample(frac=1,random_state=seed).reset_index(drop=True); n=int(.70*len(d)); x=encode(d.Sequence.to_numpy()); y=d.Delta_G.to_numpy(np.float32)
    mu,sd=float(y[:n].mean()),float(y[:n].std()); ys=((y[:n]-mu)/(sd+1e-12)).astype(np.float32)
    for name,cls in [('CfC',CurrentCfC),('CNN',CNN)]:
        m=train(cls,x[:n],ys,seed); p=pred(m,xd)*sd+mu
        torch.save({'state_dict':m.state_dict(),'target_mean':mu,'target_std':sd},OUT/f'{name}_seed{seed}.pt')
        for i,v in enumerate(p): rows.append({'Clone':ref.Name[i],'Model':name,'Seed':seed,'Predicted_Delta_G':v})
    a=torch.load(SAVED/f'primary_-16_to_-12_lstm_seed{seed}.pt',map_location='cpu',weights_only=False)
    m=LSTMRegressor(); m.load_state_dict(a['state_dict']); p=pred(m,xd)*a['target_std']+a['target_mean']
    for i,v in enumerate(p): rows.append({'Clone':ref.Name[i],'Model':'LSTM','Seed':seed,'Predicted_Delta_G':v})
    a=joblib.load(SAVED/f'primary_-16_to_-12_gbt_seed{seed}.joblib'); p=a['model'].predict(xd.reshape(len(xd),-1))*a['target_std']+a['target_mean']
    for i,v in enumerate(p): rows.append({'Clone':ref.Name[i],'Model':'GBT','Seed':seed,'Predicted_Delta_G':v})

per=pd.DataFrame(rows); per.to_csv(OUT/'DDIM15_mixed_predictions_per_seed.csv',index=False)
summary=per.groupby(['Clone','Model']).Predicted_Delta_G.agg(['mean','std']).reset_index().rename(columns={'mean':'Predicted_Delta_G_Mean','std':'Predicted_Delta_G_SD'})
summary=summary.merge(ref[['Name','Experimental_Kd_pM','Experimental_Delta_G_Calculated']],left_on='Clone',right_on='Name').drop(columns='Name')
summary['Abs_Error']=(summary.Predicted_Delta_G_Mean-summary.Experimental_Delta_G_Calculated).abs()
summary['Predicted_Rank']=summary.groupby('Model').Predicted_Delta_G_Mean.rank(method='min')
summary.to_csv(OUT/'DDIM15_mixed_predictions_3seed_summary.csv',index=False)
mr=[]
for model,g in summary.groupby('Model'):
    mr.append({'Model':model,**metrics(g.Experimental_Delta_G_Calculated,g.Predicted_Delta_G_Mean)})
pd.DataFrame(mr).to_csv(OUT/'DDIM15_mixed_clone_metrics.csv',index=False)
tops=[]
for model,g in summary.groupby('Model'):
    q=g.sort_values('Predicted_Delta_G_Mean').head(5); tops.append({'Model':model,**{f'Rank_{i+1}':v for i,v in enumerate(q.Clone)},'Kd_le_10pM':int((q.Experimental_Kd_pM<=10).sum())})
pd.DataFrame(tops).to_csv(OUT/'DDIM15_mixed_top5.csv',index=False)
print(pd.DataFrame(mr).to_string(index=False)); print(pd.DataFrame(tops).to_string(index=False))
