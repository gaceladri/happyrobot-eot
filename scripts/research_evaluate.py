#!/usr/bin/env python
"""Official evaluation of a dev-qualified trial. Never modifies or promotes the incumbent."""
import argparse,json,os,subprocess,time
from pathlib import Path
import pandas as pd
from research import ROOT,STATE,event,verify,sha,atomic

p=argparse.ArgumentParser();p.add_argument('id');a=p.parse_args()
spec=json.loads((STATE/'experiments'/f'{a.id}.json').read_text())
if spec['status'] not in ('trained','evaluating') or not spec.get('dev_gate_passed'):
 raise SystemExit('Candidate has not passed its preregistered development gate')
verify();wt=Path(spec['worktree']);ckpt=Path(spec['checkpoint'])
if subprocess.check_output(['git','status','--porcelain'],cwd=wt,text=True).strip():raise RuntimeError('Commit candidate source before evaluation')
env={**os.environ,**spec['environment'],'PYTHONPATH':str(wt/'src')+':'+str(ROOT/'third_party/eot-bench'),
 'HF_HUB_OFFLINE':'1','EOT_CHECKPOINT':str(ckpt),'EOT_DISPLAY_NAME':spec['name']}
env.pop('EOT_ONNX',None)
log=ROOT/'logs'/f'research_{a.id}_eval.log'
def run(args):
 with log.open('a') as f:subprocess.run([str(x) for x in args],cwd=wt,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
event(a.id,'evaluating',checkpoint_sha256=sha(ckpt),evaluation_started=time.time(),evaluation_log=str(log))
run([ROOT/'.venv/bin/python','-m','eot_harness','predict','--path','livekit/eot-bench-data','--name','en','--split','validation',
 '--revision','ca9d98a9686b920a2d8c9eb984224ba9be74e4dd','--adapter','eot.eotbench_adapter:EOTAdapter','--batch-size','64','--output-dir',wt/'eval/eotbench'])
files=list((wt/'eval/eotbench').glob('*/en/*/predictions.parquet'))
if len(files)!=1:raise RuntimeError('Ambiguous candidate predictions')
pred=files[0];folder=pred.parent
ref=ROOT/'eval/eotbench_comparison/en/smart_turn_audio_adapter__cb4f57229a/predictions.parquet'
keys=['id','span_index','timestamp','silence_dur','label']
assert pd.read_parquet(pred)[keys].sort_values(keys).reset_index(drop=True).equals(pd.read_parquet(ref)[keys].sort_values(keys).reset_index(drop=True))
hpy=ROOT/'third_party/eot-bench/.venv/bin/python'
run([hpy,'-m','eot_harness','compute-metrics','--predictions',pred,'--score-point','.2','--output-dir',folder/'metrics'])
inc=json.loads((STATE/'incumbent.json').read_text());parent=Path(inc['quality_dir'])
run([hpy,ROOT/'scripts/eotbench_bootstrap.py','--run-dir',folder,'--paired-with',parent,'--reps','1000'])
run([hpy,ROOT/'scripts/eotbench_bootstrap.py','--run-dir',folder,'--calibration-fraction','.5','--reps','1000','--out',folder/'metrics/calibrated_test.json'])
s=json.loads((folder/'metrics/summary.json').read_text());base=json.loads((parent/'metrics/summary.json').read_text())
def fc(d,b):
 t=pd.read_parquet(d/'metrics/tradeoff.parquet');t=t[(t.policy_type=='model')&(t.mean_latency<=b+1e-9)]
 return float(t.cutoff_rate.min()) if len(t) else 1.
metrics={**spec.get('metrics',{}),**{f'delay_at_{k}_ms':v['mean_latency']*1000 for k,v in s['operating_points'].items() if v},
 'cutoff_at_300ms_pct':100*fc(folder,.3),'cutoff_at_600ms_pct':100*fc(folder,.6)}
point=s['operating_points'].get('5pct');old=base['operating_points']['5pct']['mean_latency']
bs=json.loads((folder/'metrics/bootstrap_paired.json').read_text())['points']['latency_at_5pct']
gates=dict(primary=bool(point and old-point['mean_latency']>=max(.05,.05*old)),
 fc300=fc(folder,.3)<=fc(parent,.3)+.02,fc600=fc(folder,.6)<=fc(parent,.6)+.02,
 delay10=s['operating_points']['10pct']['mean_latency']<=base['operating_points']['10pct']['mean_latency']+.05,
 paired_latency_upper_negative=bs['paired_diff_mean_latency_ci95'][1]<0)
verify()
event(a.id,'quality-qualified' if all(gates.values()) else 'rejected',metrics=metrics,result_dir=str(folder),
 quality_gates=gates,quality_parent=inc['id'],
 decision='Proceed to export, serving and promotion gates' if all(gates.values()) else 'Rejected by predeclared quality gates; incumbent unchanged',evaluation_completed=time.time())
print(json.dumps(dict(metrics=metrics,gates=gates),indent=2))
