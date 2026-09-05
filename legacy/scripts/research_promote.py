#!/usr/bin/env python
"""Atomically select a fully evidenced release; never merge experimental source into main."""
import argparse,json,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
from research import ROOT,STATE,atomic,event,sha,verify

def check(spec,inc):
 if spec['status']!='quality-qualified':raise ValueError('quality qualification missing')
 if spec.get('quality_parent')!=inc['id']:raise ValueError('quality comparison targets a stale incumbent')
 if not spec.get('quality_gates') or not all(spec['quality_gates'].values()):raise ValueError('quality gates failed')
 if spec.get('replication_requirement') and spec.get('replication_passed') is not True:raise ValueError('required replication incomplete or failed')
 ev=spec.get('deployment') or {}
 for k in ['onnx','performance','parity','tests_xml','bundle']:
  if k not in ev or not Path(ev[k]).exists():raise ValueError('missing evidence: '+k)
 model=Path(ev['onnx']);digest=sha(model);meta=json.loads(model.with_suffix('.json').read_text())
 if meta.get('checkpoint_sha256')!=spec.get('checkpoint_sha256') or sha(spec['checkpoint'])!=spec.get('checkpoint_sha256'):raise ValueError('export/checkpoint does not match the evaluated model')
 if meta.get('sha256')!=digest or meta.get('accepted') is not True:raise ValueError('export not accepted or stale')
 parity=json.loads(Path(ev['parity']).read_text())
 if parity.get('sha256')!=digest or parity.get('passed') is not True or parity.get('n_points')!=1105:raise ValueError('benchmark parity missing or stale')
 if parity['max_probability_drift']>=1e-4 or parity['flips_at_0_5']:raise ValueError('probability parity failed')
 if any(parity.get('flips_by_threshold',{}).values()):raise ValueError('operating-threshold parity failed')
 perf=json.loads(Path(ev['performance']).read_text())['results']
 matches=[x for x in perf if x['health']['model_sha']==digest[:16] and x['threads']==4]
 if not matches:raise ValueError('performance tested a different artifact')
 levels=matches[-1]['levels']
 if {v['concurrency'] for v in levels}!={1,4,8}:raise ValueError('incomplete concurrency grid')
 if any(v['requests']<400 or v['errors']!=0 or not v['client_total_ms']['p95']<100 for v in levels):raise ValueError('request latency/error gate failed')
 suites=list(ET.parse(ev['tests_xml']).getroot().iter('testsuite'))
 if sum(int(s.attrib.get('tests',0)) for s in suites)<50 or any(int(s.attrib.get(k,0)) for s in suites for k in ['failures','errors']):raise ValueError('regression suite incomplete or failed')
 leakage=json.loads(Path(spec.get('leakage_audit',STATE/'leakage_audit.json')).read_text())
 if spec.get('extra_train_sha256') and leakage.get('extra_samples_sha256')!=spec['extra_train_sha256']:raise ValueError('extra data leakage audit missing or stale')
 if spec.get('exclude_train_sha256'):
  if sha(spec['exclude_train_ids'])!=spec['exclude_train_sha256'] or leakage.get('exclude_sha256')!=spec['exclude_train_sha256']:raise ValueError('training subset audit missing or stale')
  if leakage.get('baseline_audit_sha256')!=sha(STATE/'leakage_audit.json'):raise ValueError('parent leakage audit changed')
  if leakage.get('baseline_samples_sha256')!=sha(ROOT/'data/optimization/mixed.jsonl') or not leakage.get('dev_unchanged'):raise ValueError('training subset provenance failed')
 if leakage.get('passed') is not True:raise ValueError('leakage audit failed')
 wt=Path(spec['worktree'])
 if subprocess.check_output(['git','status','--porcelain'],cwd=wt,text=True).strip():raise ValueError('worktree is not clean')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()
 if commit!=spec['code_commit']:raise ValueError('source changed after qualification')
 return dict(id=spec['id'],name=spec['name'],source_snapshot=commit,checkpoint=spec['checkpoint'],
  checkpoint_sha256=sha(spec['checkpoint']),onnx=str(model),onnx_sha256=digest,
  quality_dir=spec['result_dir'],performance=ev['performance'],parity=ev['parity'],
  submission_bundle=ev['bundle'],previous_incumbent=inc['id'],accepted_unix=time.time(),production_ready=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('id');a=p.parse_args();verify()
 spec=json.loads((STATE/'experiments'/f'{a.id}.json').read_text());inc=json.loads((STATE/'incumbent.json').read_text())
 selected=check(spec,inc)
 # The manifest replacement is the promotion transaction. The old source/artifacts stay intact.
 atomic(STATE/'incumbent.json',selected)
 event(a.id,'accepted',decision='All registered quality/deployment gates passed',incumbent_manifest=str(STATE/'incumbent.json'))
 print(json.dumps(selected,indent=2))
