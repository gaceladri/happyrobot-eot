#!/usr/bin/env python
"""Check frozen training lineage, source/clip disjointness, and exact audio-window collisions."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from datasets import load_dataset,Audio
from eot.data import read_samples,grouped_split
from eot.audio import last_window,resample,to_mono
from research import ROOT,STATE,atomic,verify,sha
sys.path.insert(0,str(ROOT/'third_party/eot-bench'))
from eot_harness.io import iter_transformed_row

p=argparse.ArgumentParser();p.add_argument('--extra-samples',type=Path);p.add_argument('--out',type=Path,default=STATE/'leakage_audit.json');args=p.parse_args()
verify()
rows=read_samples(ROOT/'data/optimization/mixed.jsonl')
rows=[r for r in rows if r.get('time_to_onset') is None or r['time_to_onset']>=0]
tr,dv,diagnostics=grouped_split(rows,dev_frac=.15,seed=17,return_diagnostics=True)
if args.extra_samples:
 extra=read_samples(args.extra_samples)
 assert {r['source'] for r in extra}<={r['source'] for r in tr}
 assert not ({r['id'] for r in extra}&{r['id'] for r in rows})
 tr=tr+extra
assert not ({r['source'] for r in tr}&{r['source'] for r in dv})
assert not ({r['clip_id'] for r in tr}&{r['clip_id'] for r in dv})
assert all('eot-bench' not in r['path'] and '/krisp/' not in r['path'] and '/heldout/' not in r['path'] for r in rows)
def fingerprint(x,sr):
 x=last_window(resample(to_mono(x),sr))
 return hashlib.sha256(np.asarray(x,dtype='<f4').tobytes()).hexdigest()
train_hash={};dev_hash={}
for group,dest in [(tr,train_hash),(dv,dev_hash)]:
 for r in group:
  x,sr=sf.read(r['path'],dtype='float32');dest.setdefault(fingerprint(x,sr),[]).append(r['id'])
collisions=[(train_hash[k],dev_hash[k]) for k in train_hash.keys()&dev_hash.keys()]
ds=load_dataset('livekit/eot-bench-data',name='en',split='validation',revision='ca9d98a9686b920a2d8c9eb984224ba9be74e4dd').cast_column('audio',Audio(decode=False))
bench_collisions=[];points=0
for row in ds:
 for point in iter_transformed_row(row,max_audio_sec=8):
  audio=point['audio'];h=fingerprint(audio['array'],audio['sampling_rate']);points+=1
  if h in train_hash:bench_collisions.append(dict(turn=str(row['id']),timestamp=point['timestamp'],train_ids=train_hash[h]))
result=dict(extra_samples_sha256=sha(args.extra_samples) if args.extra_samples else None,train_n=len(tr),dev_n=len(dv),source_split=diagnostics,benchmark_points=points,
 train_dev_exact_audio_collisions=collisions,train_benchmark_exact_audio_collisions=bench_collisions,
 passed=not collisions and not bench_collisions,
 limitations='Exact decoded last-window fingerprints and recorded provenance only; not semantic/near-duplicate detection or proof about unknown public pretraining data.')
atomic(args.out,result)
print(json.dumps({k:v for k,v in result.items() if k!='source_split'},indent=2))
if not result['passed']:raise SystemExit('Leakage audit failed; investigate before promotion')
