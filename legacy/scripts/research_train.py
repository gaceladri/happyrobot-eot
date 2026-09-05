#!/usr/bin/env python
"""Run one registered isolated training trial; preserve the incumbent on every exit."""
import argparse,json,os,signal,subprocess,time
import numpy as np
from pathlib import Path
from research import ROOT,STATE,event,verify,sha

p=argparse.ArgumentParser();p.add_argument('id');p.add_argument('--resume',action='store_true');a=p.parse_args()
spec=json.loads((STATE/'experiments'/f'{a.id}.json').read_text())
if spec['status'] not in ('registered','interrupted','failed'):
 raise SystemExit('Run is not restartable; inspect ledger before proceeding')
verify()
if spec.get('extra_train_samples') and sha(spec['extra_train_samples'])!=spec['extra_train_sha256']:raise RuntimeError('Registered extra manifest changed')
if spec.get('exclude_train_ids') and sha(spec['exclude_train_ids'])!=spec['exclude_train_sha256']:raise RuntimeError('Registered exclusion manifest changed')
wt=Path(spec['worktree']);out=wt/'runs'/a.id.lower();log=ROOT/'logs'/f'research_{a.id}.log'
args=spec['command']+(['--resume'] if a.resume else [])
event(a.id,'training',started_unix=time.time(),log=str(log),runner_pid=os.getpid())
proc=None
try:
 with log.open('a' if a.resume else 'w') as f:
  proc=subprocess.Popen(args,cwd=wt,env={**os.environ,**spec['environment']},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  event(a.id,'training',training_pid=proc.pid)
  def stop(sig,frame):
   os.killpg(proc.pid,signal.SIGTERM)
   raise KeyboardInterrupt
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  code=proc.wait(timeout=900)
 if code:raise RuntimeError(f'training exited {code}; see {log}')
 history=[json.loads(l) for l in (out/'history.jsonl').read_text().splitlines()]
 best=max(h['auc'] for h in history)
 dev_pass=best>=.8773890591
 screening={}
 if spec.get('screening_rule')=='auc_or_low_fpr':
  def tpr(path):
   rows=[json.loads(l) for l in path.read_text().splitlines()]
   rows=[r for r in rows if abs(r['cut_time']-r['pause_start']-.2)<.001]
   y=np.array([r['label'] for r in rows]);p=np.array([r['p_eot'] for r in rows])
   threshold=np.quantile(p[y==0],.95)
   return float((p[y==1]>threshold).mean())
  base_tpr=tpr(ROOT/'runs/opt_clean/dev_scores.jsonl');candidate_tpr=tpr(out/'dev_scores.jsonl')
  screening=dict(dev_200ms_tpr_at_5fpr=candidate_tpr,baseline_dev_200ms_tpr_at_5fpr=base_tpr)
  dev_pass=dev_pass or (candidate_tpr-base_tpr>=.05 and best>=.8673890591)
 fields=dict(metrics={'dev_auc':best,'optimizer_updates':history[-1]['step'],'training_elapsed_s':history[-1]['elapsed_s'],**screening},
  checkpoint=str(out/'model.pt'),completed_unix=time.time(),dev_gate_passed=dev_pass,
  decision='eligible for official benchmark' if dev_pass else 'rejected at development gate; no benchmark accessed')
 wp=out/'wandb.json'
 if wp.exists():
  w=json.loads(wp.read_text());fields.update(wandb_id=w['run_id'],wandb_url=f"https://wandb.ai/{w['entity']}/{w['project']}/runs/{w['run_id']}",wandb_status='training-synced')
 event(a.id,'trained' if dev_pass else 'rejected',**fields)
 print(json.dumps(fields,indent=2))
except (KeyboardInterrupt,subprocess.TimeoutExpired):
 if proc and proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM);proc.wait()
 event(a.id,'interrupted',decision='incumbent unchanged; resume from last atomic checkpoint')
 raise
except Exception as e:
 event(a.id,'failed',failure=str(e),decision='incumbent unchanged')
 raise
