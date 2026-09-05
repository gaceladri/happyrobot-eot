#!/usr/bin/env python
"""Score a reference on the frozen development split without accessing the benchmark."""
import argparse,json
from pathlib import Path
import torch,numpy as np
from torch.utils.data import DataLoader
from eot.model import load_checkpoint
from eot.data import read_samples,grouped_split,MinedDataset,collate,atomic_write_json,atomic_write_jsonl
from eot.train import evaluate
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=read_samples(Path('data/optimization/mixed.jsonl'));rows=[r for r in rows if r.get('time_to_onset') is None or r['time_to_onset']>=0]
_,dev=grouped_split(rows,dev_frac=.15,seed=17);m=load_checkpoint(a.checkpoint)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=m.to(d)
loader=DataLoader(MinedDataset(dev,seed=18,normalize_audio=m.cfg.normalize_audio),batch_size=64,num_workers=0,collate_fn=collate)
metrics,scores=evaluate(m,loader,d)
records=[{**r,**s,'path':None} for r,s in zip(dev,scores)]
point=[r for r in records if abs(r['cut_time']-r['pause_start']-.2)<.001]
y=np.array([r['label'] for r in point]);p=np.array([r['p_eot'] for r in point]);thr=np.quantile(p[y==0],.95)
metrics.update(dev_200ms_tpr_at_5fpr=float((p[y==1]>thr).mean()),checkpoint=a.checkpoint)
atomic_write_json(a.out,metrics);atomic_write_jsonl(a.out.with_suffix('.scores.jsonl'),records);print(json.dumps(metrics))
