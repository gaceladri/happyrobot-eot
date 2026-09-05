#!/usr/bin/env python
"""Rebuild the local live report/plots from the durable registry; optional small W&B media."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research import ROOT,STATE,atomic,event

def save_plot(fig,path):
 temporary=path.with_name('.'+path.name+'.tmp')
 fig.savefig(temporary,dpi=150,format='png');temporary.replace(path)

p=argparse.ArgumentParser();p.add_argument('--wandb',action='store_true');a=p.parse_args()
root=ROOT/'eval/eotbench_comparison/en'
refs=['live_kit_turn_detector_adapter__turn_detector_v1','ultra_vad_adapter__920c72c9d9',
 'live_kit_turn_detector_mini_adapter__7cd280604d','smart_turn_audio_adapter__cb4f57229a','eot_adapter__1d657c44f3']
scorecard=[]
fig,ax=plt.subplots(figsize=(8,5))
for name in refs:
 folder=root/name;m=json.loads((folder/'manifest.json').read_text());s=json.loads((folder/'metrics/summary.json').read_text())
 t=pd.read_parquet(folder/'metrics/tradeoff.parquet');t=t[t.policy_type=='model']
 d=dict(name=m['display_name'],directory=str(folder),score_mode=s['score_mode'],score_point_s=s['score_point_s'])
 for k in ['2pct','5pct','10pct']:
  x=s['operating_points'].get(k);d['delay_at_'+k+'_ms']=1000*x['mean_latency'] if x else None
 for budget in [.3,.6]:
  x=t[t.mean_latency<=budget+1e-9];d[f'cutoff_at_{int(budget*1000)}ms_pct']=100*x.cutoff_rate.min() if len(x) else None
 scorecard.append(d)
 frontier=t.sort_values(['cutoff_rate','mean_latency']).groupby('cutoff_rate',as_index=False).mean_latency.min()
 frontier['best']=frontier.mean_latency.cummin()
 frontier=frontier[(frontier['best'].diff().fillna(-1)<0)&(frontier.cutoff_rate<=.3)]
 ax.step(100*frontier.cutoff_rate,1000*frontier.best,where='post',label=d['name'])
ax.set(xlabel='HOLD-span cutoff (%)',ylabel='Mean endpointing delay (ms)',title='Frozen English benchmark references — fixed 200 ms score')
ax.legend(fontsize=7);ax.grid(alpha=.25);fig.tight_layout();save_plot(fig,STATE/'plots/baseline_frontier.png');plt.close(fig)
baseline=STATE/'baseline_scorecard.json'
if not baseline.exists():atomic(baseline,scorecard)
else:assert json.loads(baseline.read_text())==scorecard,'Frozen baseline scorecard changed'
experiments=[json.loads(p.read_text()) for p in sorted((STATE/'experiments').glob('*.json'))]
bench_trials=[e for e in experiments if e['id'].startswith('E') and e.get('result_dir')]
if bench_trials:
 fig,ax=plt.subplots(figsize=(8,5))
 comparisons=[('Frozen LiveKit v1',root/refs[0]),('Original incumbent',root/refs[-1])]+[(e['id'],Path(e['result_dir'])) for e in bench_trials]
 for label,folder in comparisons:
  t=pd.read_parquet(folder/'metrics/tradeoff.parquet');t=t[t.policy_type=='model']
  f=t.groupby('cutoff_rate',as_index=False).mean_latency.min().sort_values('cutoff_rate');f['best']=f.mean_latency.cummin()
  f=f[(f.best.diff().fillna(-1)<0)&(f.cutoff_rate<=.3)]
  ax.step(100*f.cutoff_rate,1000*f.best,where='post',label=label)
 ax.set(xlabel='HOLD-span cutoff (%)',ylabel='Mean endpointing delay (ms)',title='Official EN frontier — adaptively selected research candidates')
 ax.grid(alpha=.25);ax.legend();fig.tight_layout();save_plot(fig,STATE/'plots/research_frontier.png');plt.close(fig)
inc=json.loads((STATE/'incumbent.json').read_text())
quality=json.loads((Path(inc['quality_dir'])/'metrics/summary.json').read_text())
performance=json.loads(Path(inc['performance']).read_text())
matched=[r for r in performance['results'] if r['health']['model_sha']==inc['onnx_sha256'][:16] and r['threads']==4]
latencies=' / '.join(f"{r['client_total_ms']['p95']:.1f}" for r in sorted(matched[-1]['levels'],key=lambda r:r['concurrency']))
lines=['# Live research results','',
 f"Incumbent: **{inc['name']}** (`{inc['id']}`), {quality['operating_points']['5pct']['mean_latency']*1000:,.1f} ms endpointing delay at 5% cutoff; HTTP p95 "
 f'**{latencies} ms** at concurrency **1 / 4 / 8**. `incumbent.json` is the authoritative selection. '
 'No SOTA or production-readiness claim.','',
 '![Frozen frontier](plots/baseline_frontier.png)','',
 '| Model | Delay @2% | Delay @5% | Delay @10% | Cutoff @300 ms | Cutoff @600 ms |',
 '|---|---:|---:|---:|---:|---:|']
for d in scorecard:
 vals=[d.get(k) for k in ['delay_at_2pct_ms','delay_at_5pct_ms','delay_at_10pct_ms','cutoff_at_300ms_pct','cutoff_at_600ms_pct']]
 lines.append('| '+d['name']+' | '+' | '.join('—' if v is None else f'{v:.1f}' for v in vals)+' |')
for e in bench_trials:
 m=e['metrics'];keys=['delay_at_2pct_ms','delay_at_5pct_ms','delay_at_10pct_ms','cutoff_at_300ms_pct','cutoff_at_600ms_pct']
 lines.append('| '+e['id']+' ('+e['status']+') | '+' | '.join(f'{m[k]:.1f}' for k in keys)+' |')
if bench_trials:lines+=['','![Research frontier](plots/research_frontier.png)','']
lines += ['', 'Delays are milliseconds; cutoffs are percentages of eligible HOLD spans. Published LiveKit '
 'v1 scores are a reproducible artifact reference, not an independently rerun cloud-model result. '
 'English was already inspected; intervals and improvements remain adaptively selected.', '',
 '## Iterations','', '| ID | State | Dev AUC | Decision | W&B |','|---|---|---:|---|---|']
fig,ax=plt.subplots(figsize=(8,4));curves=0
for e in experiments:
 url=e.get('wandb_url');auc=e.get('metrics',{}).get('dev_auc')
 lines.append(f"| {e['id']} | {e['status']} | {auc if auc is not None else '—'} | {e.get('decision','in progress')} | "+(f'[run]({url})' if url else e.get('wandb_status','pending'))+' |')
 if e.get('worktree'):
  h=Path(e['worktree'])/'runs'/e['id'].lower()/'history.jsonl'
  if h.exists():
   hs=[json.loads(l) for l in h.read_text().splitlines()]
   ax.plot([x['step'] for x in hs],[x['auc'] for x in hs],marker='.',label=e['id']);curves+=1
h=[json.loads(l) for l in (ROOT/'runs/opt_clean/history.jsonl').read_text().splitlines()]
ax.plot([x['step'] for x in h],[x['auc'] for x in h],marker='.',label='Incumbent')
ax.set(xlabel='Optimizer update',ylabel='Source-held-out dev AUC',title='Development learning curves (diagnostic, not benchmark)')
ax.legend();ax.grid(alpha=.25);fig.tight_layout();save_plot(fig,STATE/'plots/dev_curves.png');plt.close(fig)
lines += ['','![Development curves](plots/dev_curves.png)','',
 'Resume from [README](README.md), [bitácora](BITACORA.md), [backlog](BACKLOG.md) and the exact commands '
 'in `experiments/<id>.json`. Model weights/checkpoints remain local. W&B tracks metrics, small '
 'tables and plots only.','']
temporary=STATE/'.RESULTS.md.tmp';temporary.write_text('\n'.join(lines));temporary.replace(STATE/'RESULTS.md')
if a.wandb:
 import wandb
 run=wandb.init(project='happyrobot-eot',entity='adrianbrunetto1-reshape',id='ar20260905report',resume='allow',
  group='autoresearch-20260905',name='Autoresearch live scorecard',job_type='report',
  settings=wandb.Settings(disable_code=True,init_timeout=30),dir=str(ROOT/'wandb'))
 run.log({'baseline_frontier':wandb.Image(str(STATE/'plots/baseline_frontier.png')),
  'dev_curves':wandb.Image(str(STATE/'plots/dev_curves.png')),
  'scorecard':wandb.Table(dataframe=pd.DataFrame(scorecard))})
 if bench_trials:run.log({'research_frontier':wandb.Image(str(STATE/'plots/research_frontier.png'))})
 url=run.url;run.finish();atomic(STATE/'wandb_report.json',{'url':url,'id':'ar20260905report'})
print(STATE/'RESULTS.md')
