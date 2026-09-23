#!/usr/bin/env python3
"""Create common-test-set violin inputs for all mixed-trained models."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ncps.torch import CfC
from torch import nn

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/mixed_test_violin_graphpad'; OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data/htsk.csv'; AA='ACDEFGHIKLMNPQRSTVWY'; IDX={a:i for i,a in enumerate(AA)}; SEEDS=(42,43,44)
def onehot(seqs):
    x=np.zeros((len(seqs),20,20),np.float32)
    for i,s in enumerate(seqs):
        for j,a in enumerate(s): x[i,j,IDX[a]]=1
    return x
def tokens(seqs): return np.asarray([[IDX[a] for a in s] for s in seqs],np.int64)

class CurrentCfC(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.LayerNorm(20); self.d1=nn.Linear(20,32); self.c1=CfC(input_size=32,units=32,return_sequences=True,batch_first=True); self.dp1=nn.Dropout(.2); self.d2=nn.Linear(32,16); self.c2=CfC(input_size=16,units=16,return_sequences=False,batch_first=True); self.dp2=nn.Dropout(.2); self.out=nn.Linear(16,1)
    def forward(self,x):
        x=torch.tanh(self.d1(self.norm(x))); x,_=self.c1(x); x=self.dp1(x); x=torch.tanh(self.d2(x)); x,_=self.c2(x); return self.out(self.dp2(x)).squeeze(-1)
class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__(); self.lstm=nn.LSTM(20,64,batch_first=True); self.dropout1=nn.Dropout(.2); self.dense=nn.Linear(64,16); self.dropout2=nn.Dropout(.2); self.output=nn.Linear(16,1)
    def forward(self,x):
        _,(h,_)=self.lstm(x); return self.output(self.dropout2(torch.tanh(self.dense(self.dropout1(h[-1]))))).squeeze(-1)
class CompactTransformer(nn.Module):
    def __init__(self):
        super().__init__(); self.aa=nn.Embedding(20,64); self.pos=nn.Parameter(torch.zeros(1,20,64))
        layer=nn.TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=128,dropout=.15,activation='gelu',batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,num_layers=2,norm=nn.LayerNorm(64)); self.head=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Dropout(.15),nn.Linear(32,1))
    def forward(self,x): return self.head(self.encoder(self.aa(x)+self.pos).mean(dim=1)).squeeze(-1)
@torch.no_grad()
def predict(m,x,b=1024):
    m.eval(); return np.concatenate([m(torch.from_numpy(x[i:i+b])).cpu().numpy() for i in range(0,len(x),b)])

def read_existing(model,seed):
    if model=='CNN': p=ROOT/f'outputs/current_logic_neural_rerun/primary_-16_to_-12_cnn_seed{seed}_test_predictions.csv'
    elif model=='GBT': p=ROOT/f'outputs/current_logic_lstm_gbt/primary_-16_to_-12_gbt_seed{seed}_test_predictions.csv'
    elif model=='LSTM70': p=ROOT/f'outputs/current_logic_lstm_gbt/primary_-16_to_-12_lstm_seed{seed}_test_predictions.csv'
    d=pd.read_csv(p); return d.set_index('Sequence').Predicted_Delta_G

def main():
    torch.set_num_threads(2); raw=pd.read_csv(DATA); raw.Sequence=raw.Sequence.astype(str).str.strip().str.upper(); raw.Delta_G=pd.to_numeric(raw.Delta_G,errors='coerce'); raw=raw.loc[raw.Sequence.str.fullmatch(f'[{AA}]{{20}}',na=False)&raw.Delta_G.between(-16,-12)].reset_index(drop=True)
    transformer70=pd.read_csv(ROOT/'outputs/combined_compact_transformer/holdout_predictions.csv')
    all70=[]; allupdated=[]
    for seed in SEEDS:
        d=raw.sample(frac=1,random_state=seed).reset_index(drop=True); n=int(.70*len(d)); ve=n+int(.15*len(d)); test=d.iloc[ve:].copy(); seq=test.Sequence.to_numpy(); x=onehot(seq); xt=tokens(seq); mean=float(d.Delta_G.iloc[:n].mean())
        base=pd.DataFrame({'Seed':seed,'Sequence':seq,'Experimental_Delta_G':test.Delta_G.to_numpy(),'Mean_Predictor':mean})
        base['CNN']=base.Sequence.map(read_existing('CNN',seed)); base['GBT']=base.Sequence.map(read_existing('GBT',seed)); base['LSTM']=base.Sequence.map(read_existing('LSTM70',seed))
        tr=transformer70[transformer70.Seed.eq(seed)].set_index('Sequence').Predicted_Delta_G; base['Compact_Transformer']=base.Sequence.map(tr)
        saved=torch.load(ROOT/f'outputs/mixed_train_ddim15/CfC_seed{seed}.pt',map_location='cpu',weights_only=False); m=CurrentCfC(); m.load_state_dict(saved['state_dict']); base['CfC']=predict(m,x)*saved['target_std']+saved['target_mean']; all70.append(base)
        updated=base.copy()
        saved=torch.load(ROOT/f'outputs/mixed_lstm_transformer_200epochs/LSTM_seed{seed}_200epochs.pt',map_location='cpu',weights_only=False); m=LSTMRegressor(); m.load_state_dict(saved['best_state_dict']); updated['LSTM']=predict(m,x)*saved['target_std']+saved['target_mean']
        saved=torch.load(ROOT/f'outputs/mixed_lstm_transformer_200epochs/CompactTransformer_seed{seed}_200epochs.pt',map_location='cpu',weights_only=False); m=CompactTransformer(); m.load_state_dict(saved['best_state_dict']); updated['Compact_Transformer']=predict(m,xt)*saved['target_std']+saved['target_mean']; allupdated.append(updated)
    columns=['Experimental_Delta_G','Mean_Predictor','CNN','GBT','Compact_Transformer','LSTM','CfC']
    for label,frames in [('Fair70epoch',all70),('Updated200epoch',allupdated)]:
        pooled=pd.concat(frames,ignore_index=True); seed42=frames[0]
        seed42[['Sequence']+columns].to_csv(OUT/f'{label}_Seed42_common_test_wide_GraphPad.tsv',sep='\t',index=False,float_format='%.8f')
        pooled[['Seed','Sequence']+columns].to_csv(OUT/f'{label}_3seed_pooled_test_wide.tsv',sep='\t',index=False,float_format='%.8f')
        long=pooled.melt(id_vars=['Seed','Sequence'],value_vars=columns,var_name='Model',value_name='Delta_G')
        long.to_csv(OUT/f'{label}_3seed_pooled_test_long.tsv',sep='\t',index=False,float_format='%.8f')
        pd.DataFrame({'Model':columns,'N':[seed42[c].notna().sum() for c in columns],'Mean':[seed42[c].mean() for c in columns],'SD':[seed42[c].std() for c in columns],'Median':[seed42[c].median() for c in columns],'Min':[seed42[c].min() for c in columns],'Max':[seed42[c].max() for c in columns]}).to_csv(OUT/f'{label}_Seed42_distribution_summary.csv',index=False)
    print('Rows per seed:',len(all70[0])); print('Missing values fair:',pd.concat(all70)[columns].isna().sum().to_dict()); print('Missing values updated:',pd.concat(allupdated)[columns].isna().sum().to_dict())

if __name__=='__main__': main()
