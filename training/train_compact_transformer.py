#!/usr/bin/env python3
from pathlib import Path
import copy, json, os, random, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('COMPACT_TRANSFORMER_DATA',str(ROOT/'data/htsk.csv')))
DDIM=ROOT/'data/ddim_15_sequences.csv'
OUT=Path(os.environ.get('COMPACT_TRANSFORMER_OUT',str(ROOT/'outputs/combined_compact_transformer')))
OUT.mkdir(parents=True,exist_ok=True)
AA='ACDEFGHIKLMNPQRSTVWY'; TOK={a:i for i,a in enumerate(AA)}; SEEDS=(42,43,44)

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def tokens(seqs): return np.asarray([[TOK[a] for a in s] for s in seqs],dtype=np.int64)
def metrics(y,p):
    mse=mean_squared_error(y,p)
    return {'MAE':mean_absolute_error(y,p),'MSE':mse,'RMSE':np.sqrt(mse),'R2':r2_score(y,p),'Pearson':pearsonr(y,p).statistic,'Spearman':spearmanr(y,p).statistic}

class CompactTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.aa=nn.Embedding(20,64); self.pos=nn.Parameter(torch.zeros(1,20,64)); nn.init.normal_(self.pos,std=.02)
        layer=nn.TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=128,dropout=.15,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2,norm=nn.LayerNorm(64))
        self.head=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Dropout(.15),nn.Linear(32,1))
    def forward(self,x): return self.head(self.encoder(self.aa(x)+self.pos).mean(dim=1)).squeeze(-1)

@torch.no_grad()
def predict(m,x,b=1024):
    m.eval(); return np.concatenate([m(torch.from_numpy(x[i:i+b])).cpu().numpy() for i in range(0,len(x),b)])

def train(xtr,ytr,xv,yv,seed):
    set_seed(seed); m=CompactTransformer(); opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4); loss=nn.HuberLoss(delta=.5)
    gen=torch.Generator().manual_seed(seed); dl=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=64,shuffle=True,generator=gen,num_workers=0)
    hist=[]; best=np.inf; state=None; best_epoch=None
    for ep in range(1,71):
        m.train()
        for xb,yb in dl:
            opt.zero_grad(set_to_none=True); z=loss(m(xb),yb); z.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        tr=float(loss(torch.from_numpy(predict(m,xtr)),torch.from_numpy(ytr)).item()); va=float(loss(torch.from_numpy(predict(m,xv)),torch.from_numpy(yv)).item())
        hist.append({'Epoch':ep,'Train_Huber_scaled':tr,'Validation_Huber_scaled':va})
        if va<best: best=va; state=copy.deepcopy(m.state_dict()); best_epoch=ep
        if ep==1 or ep%10==0: print(f'seed={seed} epoch={ep} train={tr:.5f} val={va:.5f}',flush=True)
    # Keep fixed-70 result for the primary fair comparison; also save the validation-best checkpoint.
    return m,state,best_epoch,pd.DataFrame(hist)

torch.set_num_threads(int(os.environ.get('MODEL_NUM_THREADS','2')))
d=pd.read_csv(DATA); d.Sequence=d.Sequence.astype(str).str.strip().str.upper(); d.Delta_G=pd.to_numeric(d.Delta_G,errors='coerce')
d=d.loc[d.Sequence.str.fullmatch(f'[{AA}]{{20}}',na=False)&d.Delta_G.between(-16,-12)].reset_index(drop=True)
ref=pd.read_csv(DDIM); ref['Experimental_Delta_G_Calculated']=1.987204258e-3*298.15*np.log(ref.Experimental_Kd_pM.to_numpy(float)*1e-12); xd=tokens(ref.Model_Input_20aa)
allmet=[]; allpred=[]; allhist=[]; ddimrows=[]
for seed in SEEDS:
    q=d.sample(frac=1,random_state=seed).reset_index(drop=True); x=tokens(q.Sequence); y=q.Delta_G.to_numpy(np.float32); n=int(.70*len(q)); nv=int(.15*len(q)); ve=n+nv
    mu,sd=float(y[:n].mean()),float(y[:n].std()); ys=((y-mu)/(sd+1e-12)).astype(np.float32)
    start=time.time(); model,best_state,best_epoch,h=train(x[:n],ys[:n],x[n:ve],ys[n:ve],seed); h['Seed']=seed; allhist.append(h)
    p=predict(model,x[ve:])*sd+mu; mm={'Seed':seed,'Checkpoint':'Epoch70','Best_validation_epoch':best_epoch,**metrics(y[ve:],p)}; allmet.append(mm)
    for i,v in enumerate(p): allpred.append({'Seed':seed,'Sequence':q.Sequence.iloc[ve+i],'Actual_Delta_G':y[ve+i],'Predicted_Delta_G':v})
    pdim=predict(model,xd)*sd+mu
    for i,v in enumerate(pdim): ddimrows.append({'Seed':seed,'Clone':ref.Name.iloc[i],'Experimental_Kd_pM':ref.Experimental_Kd_pM.iloc[i],'Experimental_Delta_G':ref.Experimental_Delta_G_Calculated.iloc[i],'Predicted_Delta_G':v})
    torch.save({'state_dict':model.state_dict(),'best_state_dict':best_state,'target_mean':mu,'target_std':sd,'best_validation_epoch':best_epoch},OUT/f'Transformer_seed{seed}.pt')
    print(mm,'seconds',time.time()-start,flush=True)
    pd.DataFrame(allmet).to_csv(OUT/'holdout_metrics_per_seed_partial.csv',index=False); pd.DataFrame(ddimrows).to_csv(OUT/'DDIM15_predictions_per_seed_partial.csv',index=False)

met=pd.DataFrame(allmet); met.to_csv(OUT/'holdout_metrics_per_seed.csv',index=False)
cols=['MAE','MSE','RMSE','R2','Pearson','Spearman']; summ=met[cols].agg(['mean','std']).T.reset_index().rename(columns={'index':'Metric'}); summ.to_csv(OUT/'holdout_metrics_3seed_summary.csv',index=False)
pd.concat(allhist,ignore_index=True).to_csv(OUT/'training_history.csv',index=False); pd.DataFrame(allpred).to_csv(OUT/'holdout_predictions.csv',index=False)
dr=pd.DataFrame(ddimrows); dr.to_csv(OUT/'DDIM15_predictions_per_seed.csv',index=False)
ds=dr.groupby('Clone').agg(Experimental_Kd_pM=('Experimental_Kd_pM','first'),Experimental_Delta_G=('Experimental_Delta_G','first'),Predicted_Delta_G_Mean=('Predicted_Delta_G','mean'),Predicted_Delta_G_SD=('Predicted_Delta_G','std')).reset_index(); ds['Abs_Error']=(ds.Predicted_Delta_G_Mean-ds.Experimental_Delta_G).abs(); ds['Predicted_Rank']=ds.Predicted_Delta_G_Mean.rank(method='min'); ds.to_csv(OUT/'DDIM15_predictions_3seed_summary.csv',index=False)
dm=metrics(ds.Experimental_Delta_G,ds.Predicted_Delta_G_Mean); pd.DataFrame([dm]).to_csv(OUT/'DDIM15_metrics.csv',index=False)
top=ds.sort_values('Predicted_Delta_G_Mean').head(5); top.to_csv(OUT/'DDIM15_top5.csv',index=False)
(OUT/'run_config.json').write_text(json.dumps({'input':str(DATA),'rows':len(d),'range':[-16,-12],'seeds':SEEDS,'split':[.70,.15,.15],'epochs':70,'model':{'d_model':64,'heads':4,'layers':2,'ffn':128,'dropout':.15}},indent=2))
print('\nHOLDOUT\n',summ.to_string(index=False)); print('\nDDIM15\n',dm); print('\nTOP5\n',top[['Clone','Predicted_Delta_G_Mean','Experimental_Kd_pM']].to_string(index=False))
