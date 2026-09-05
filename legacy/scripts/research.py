#!/usr/bin/env python
"""Durable research registry. No model uploads and no mutation of official evaluation."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'research'

def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()

def atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name('.'+path.name+'.tmp')
 with tmp.open('w') as f:
  json.dump(data,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def git(*args,**kw):
 return subprocess.check_output(['git',*args],cwd=ROOT,text=True,**kw).strip()

def event(id,status,**fields):
 with (STATE/'.registry.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  p=STATE/'experiments'/f'{id}.json'
  old=json.loads(p.read_text()) if p.exists() else {}
  e=dict(id=id,status=status,time_unix=time.time(),**fields)
  with (STATE/'ledger.jsonl').open('a') as f:
   f.write(json.dumps(e,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
  atomic(p,{**old,**e})
 return {**old,**e}

def freeze():
 p=STATE/'protocol.json'
 if p.exists():raise SystemExit('Protocol already frozen; refusing replacement')
 harness=ROOT/'third_party/eot-bench'
 code=subprocess.check_output(['git','ls-files','-z'],cwd=harness).decode().split('\0')
 files=[harness/f for f in code if f and (harness/f).is_file()]
 files += list((Path.home()/'.cache/huggingface/datasets/livekit___eot-bench-data/en/0.0.0/ca9d98a9686b920a2d8c9eb984224ba9be74e4dd').glob('*.arrow'))
 files += [ROOT/'data/optimization/mixed.jsonl',ROOT/'data/optimization/mined.jsonl',ROOT/'models/smart-turn-public/model.pt']
 if not any(x.suffix=='.arrow' for x in files):raise RuntimeError('pinned dataset cache missing')
 workload='data/mined/smart-turn-ensemble/audio/3fc2e840-d3f2-452f-b93e-6456e7941329_03-ccf51929ace0.wav'
 files.append(ROOT/workload)
 record=dict(created_unix=time.time(),harness_revision=git('-C',str(harness),'rev-parse','HEAD'),
  dataset_revision='ca9d98a9686b920a2d8c9eb984224ba9be74e4dd',dataset='livekit/eot-bench-data',split='validation',language='en',
  score_mode='score_point',score_point_seconds=.2,min_hold_seconds=.2,max_hold_seconds=5.,
  primary='mean_latency_at_5pct_cutoff',sota_reference='live_kit_turn_detector_adapter__turn_detector_v1',
  workload=dict(wav=workload,threads=4,max_inflight=8,concurrency=[1,4,8],requests_each=400,p95_ms=100,errors=0),
  files={str(f):sha(f) for f in files})
 atomic(p,record)
 print('Frozen',len(files),'files')

def verify():
 p=json.loads((STATE/'protocol.json').read_text())
 bad=[f for f,s in p['files'].items() if not Path(f).is_file() or sha(f)!=s]
 if bad:raise RuntimeError('Frozen inputs changed: '+repr(bad[:10]))
 h=ROOT/'third_party/eot-bench'
 if git('-C',str(h),'rev-parse','HEAD')!=p['harness_revision']:raise RuntimeError('harness revision changed')
 if git('-C',str(h),'status','--porcelain','--untracked-files=no'):raise RuntimeError('harness tracked files modified')
 print('Frozen evaluation/training inputs verified:',len(p['files']))

def snapshot():
 p=STATE/'baseline_snapshot.json'
 if p.exists():raise SystemExit('Baseline snapshot already exists')
 before=git('status','--porcelain')
 with tempfile.TemporaryDirectory(prefix='eot-index-') as td:
  env={**os.environ,'GIT_INDEX_FILE':str(Path(td)/'index')}
  git('read-tree','HEAD',env=env)
  git('add','-A',env=env)
  tree=git('write-tree',env=env)
  commit=git('commit-tree',tree,'-p','HEAD',input='Research baseline snapshot: preserve existing working state without altering main branch/index\n')
 branch='research/baseline-20260905'
 git('branch',branch,commit)
 assert before==git('status','--porcelain'),'Main worktree/index changed during snapshot'
 atomic(p,dict(commit=commit,branch=branch,parent=git('rev-parse','HEAD'),tree=tree,
  attribution='Snapshot includes pre-existing user and assistant work; no claim of new authorship.',status_before=before))
 print(commit)

def reconcile():
 for p in (STATE/'experiments').glob('E*.json'):
  d=json.loads(p.read_text())
  if d['status']!='training':continue
  pid=d.get('training_pid');runner=d.get('runner_pid')
  if pid is None:
   print(d['id'],'has no PID record; inspect the recorded worktree/log before marking interrupted')
   continue
  alive=any(Path(f'/proc/{v}').exists() for v in [pid,runner] if v)
  if not alive:event(d['id'],'interrupted',decision='Recorded training and runner processes exited without a terminal record; checkpoint retained')
  else:print(d['id'],'still has a live recorded process; no duplicate launched')

def import_history():
 comparisons=ROOT/'eval/eotbench_comparison/en'
 for f in comparisons.glob('*/manifest.json'):
  m=json.loads(f.read_text());n=m.get('display_name',f.parent.name)
  if not n.startswith(('OPT','R2','R3','R4')):continue
  ident='H-'+f.parent.name.split('__')[-1]
  if (STATE/'experiments'/f'{ident}.json').exists():continue
  s=json.loads((f.parent/'metrics/summary.json').read_text())
  event(ident,'historical',name=n,parent=None,preregistered=False,
   result_dir=str(f.parent),score_mode=s.get('score_mode'),score_point=s.get('score_point_s'),
   metrics={f'delay_at_{k}_ms':1000*v['mean_latency'] for k,v in s['operating_points'].items() if v},
   decision='incumbent' if 'CPU-matched' in n else 'historical evidence; see BACKLOG.md',
   wandb_status='pending',wandb_id='hr'+hashlib.sha256(ident.encode()).hexdigest()[:10])
 print('Historical registry ready')

def sync(id):
 import wandb
 p=STATE/'experiments'/f'{id}.json';d=json.loads(p.read_text())
 runid=d.get('wandb_id','ar'+hashlib.sha256(id.encode()).hexdigest()[:10])
 # Deliberately no Artifact, save(), watch(), log_model(), or checkpoint callbacks.
 try:
  run=wandb.init(project='happyrobot-eot',entity='adrianbrunetto1-reshape',group='autoresearch-20260905',
   id=runid,resume='allow',name=d.get('name',id),job_type='historical' if d['status']=='historical' else 'research',
   tags=['autoresearch','no-model-artifacts',d['status']],config={k:d[k] for k in ('id','parent','hypothesis','config','code_commit') if k in d},
   settings=wandb.Settings(init_timeout=30,disable_code=True),dir=str(ROOT/'wandb'))
  for k,v in d.get('metrics',{}).items():run.summary[k]=v
  history=Path(d.get('training_history',''))
  if d['status']=='historical' and history.is_file() and not d.get('historical_curves_synced'):
   for h in (json.loads(l) for l in history.read_text().splitlines()):
    run.log({'historical/dev_auc':h['auc'],'historical/optimizer_update':h['step']})
  run.summary['decision']=d.get('decision',d['status'])
  run.summary['local_record']=str(p)
  url=run.url;run.finish()
  event(id,d['status'],wandb_id=runid,wandb_status='synced',wandb_url=url,historical_curves_synced=d['status']=='historical')
  print(url)
 except Exception as e:
  event(id,d['status'],wandb_id=runid,wandb_status='pending',wandb_error=type(e).__name__)
  print('W&B pending:',type(e).__name__)

if __name__=='__main__':
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='command',required=True)
 for name in ['freeze','verify','snapshot','import-history','reconcile']:sp.add_parser(name)
 p=sp.add_parser('sync');p.add_argument('id')
 p=sp.add_parser('record');p.add_argument('id');p.add_argument('status');p.add_argument('--json',type=Path,required=True)
 a=ap.parse_args()
 if a.command=='record':event(a.id,a.status,**json.loads(a.json.read_text()))
 elif a.command=='sync':sync(a.id)
 else:globals()[a.command.replace('-','_')]()
