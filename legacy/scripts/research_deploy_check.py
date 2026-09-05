#!/usr/bin/env python
"""Build and validate a quality-qualified candidate in isolation, then leave promotion explicit."""
import argparse,json,os,subprocess,shutil
from pathlib import Path
from research import ROOT,STATE,atomic,event,verify,sha

p=argparse.ArgumentParser();p.add_argument('id');a=p.parse_args()
d=json.loads((STATE/'experiments'/f'{a.id}.json').read_text())
if d['status']!='quality-qualified':raise SystemExit('Quality qualification required')
verify();wt=Path(d['worktree']);out=wt/'exports'/a.id.lower();out.mkdir(parents=True,exist_ok=True)
model=out/'eot.onnx';env={**os.environ,**d['environment'],'PYTHONPATH':str(wt/'src')+':'+str(ROOT/'third_party/eot-bench'),'HF_HUB_OFFLINE':'1'}
log=ROOT/'logs'/f'research_{a.id}_deploy.log'
def run(args,cwd=wt):
 with log.open('a') as f:subprocess.run([str(x) for x in args],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
py=ROOT/'.venv/bin/python';folder=Path(d['result_dir'])
run([py,'-m','eot.export_onnx','--checkpoint',d['checkpoint'],'--out',model,'--validation-samples',ROOT/'data/optimization/mixed.jsonl'])
thresholds={.5}
for point in json.loads((folder/'metrics/summary.json').read_text())['operating_points'].values():
 if point:thresholds.add(point['threshold'])
parity=wt/'eval/deployment_point_parity.json'
run([py,ROOT/'scripts/validate_served_scores.py','--model',model,'--predictions',folder/'predictions.parquet','--out',parity,'--thresholds',*sorted(thresholds)])
tests=wt/'eval/regression.xml'
run([py,'-m','pytest','-q',wt/'tests',f'--junitxml={tests}'])
image='happyrobot-eot:research-'+a.id.lower();run(['docker','build','-t',image,'.'])
perf=wt/'eval/docker_performance.json'
run([py,ROOT/'scripts/benchmark_serving.py','--models',model,'--threads','4','--requests','400','--wav',
 ROOT/'data/mined/smart-turn-ensemble/audio/3fc2e840-d3f2-452f-b93e-6456e7941329_03-ccf51929ace0.wav','--image',image,'--out',perf])
verify()
bundle=ROOT/'artifacts/research'/a.id.lower();bundle.mkdir(parents=True,exist_ok=False)
with (bundle/'source.tar').open('wb') as f:subprocess.run(['git','archive','--format=tar',d['code_commit']],cwd=wt,stdout=f,check=True)
for src in [model,model.with_suffix('.json'),parity,tests,perf]:shutil.copy2(src,bundle/src.name)
atomic(bundle/'checksums.json',{p.name:sha(p) for p in bundle.iterdir() if p.is_file()})
ev=dict(onnx=str(model),parity=str(parity),tests_xml=str(tests),performance=str(perf),bundle=str(bundle),image=image)
event(a.id,'quality-qualified',deployment=ev,decision='Deployment evidence recorded; research_promote.py must enforce every gate')
print(json.dumps(ev,indent=2))
