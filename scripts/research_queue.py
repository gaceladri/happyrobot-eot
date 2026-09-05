#!/usr/bin/env python
"""Serial local experiment queue; never runs two training trials on the same GPU."""
import argparse,json,signal,subprocess,time
from research import ROOT,STATE
p=argparse.ArgumentParser();p.add_argument('--after');p.add_argument('ids',nargs='+');a=p.parse_args()
child=None

def stop(sig,frame):
 if child and child.poll() is None:child.terminate();child.wait()
 raise SystemExit(130)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def run(script,*args):
 global child
 child=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts'/script),*args],cwd=ROOT)
 code=child.wait();child=None
 if code:raise SystemExit(code)
if a.after:
 started=time.time()
 while json.loads((STATE/'experiments'/f'{a.after}.json').read_text())['status'] in ('registered','training','trained','evaluating'):
  if time.time()-started>1000:raise SystemExit('Prior run still marked training; reconcile before continuing')
  time.sleep(5)
 if json.loads((STATE/'experiments'/f'{a.after}.json').read_text())['status']=='quality-qualified':
  raise SystemExit('Prior candidate needs deployment validation before extending the queue')
for ident in a.ids:
 run('research_train.py',ident)
 d=json.loads((STATE/'experiments'/f'{ident}.json').read_text())
 if d['status']=='trained':run('research_evaluate.py',ident)
 run('research.py','sync',ident)
 run('research_report.py')
 d=json.loads((STATE/'experiments'/f'{ident}.json').read_text())
 if d['status']=='quality-qualified':
  print('Candidate qualified; stop queue for export/deployment validation:',ident,flush=True)
  break
