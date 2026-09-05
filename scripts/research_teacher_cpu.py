"""D024 CPU console judgments: bounded, causal inputs separated from hindsight, atomic resume."""
import argparse, json, os, subprocess, sys, time, signal
from pathlib import Path
from research import ROOT, atomic, sha, event
sys.path.insert(0,str(ROOT.parent/'happyrobot-eot-experiments/d022/src'))
from eot.audio_teacher import SYSTEM,INSTRUCTION,digest,validate_judgment,validate_train

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=24);a=p.parse_args()
    base=ROOT/'data/research/D024';base.mkdir(parents=True,exist_ok=True);dest=base/'teacher';dest.mkdir(exist_ok=True)
    pilot=json.loads((ROOT/'data/research/D022/pilot.json').read_text());runtime=json.loads((ROOT/'data/research/P023/runtime.json').read_text())
    assert json.loads((ROOT/'data/research/P023/summary.json').read_text())['passed']
    if digest(pilot['cases'])!=pilot['identity']:raise ValueError('pilot changed')
    splits=ROOT/'data/research/D016/splits.json'
    if sha(splits)!=pilot['provenance']['splits_sha256']:raise ValueError('splits changed')
    validate_train(pilot['cases'],json.loads(splits.read_text()))
    for path,h in runtime['runtime']['files'].items():
        if sha(path)!=h:raise ValueError('model changed')
    binary=ROOT/'artifacts/research/tools/llama.cpp/build-cuda129/bin/llama-mtmd-cli'
    if sha(binary)!=runtime['binary_sha256']:raise ValueError('runtime changed')
    schema={'type':'object','properties':{'label':{'type':'string','enum':['EOT','HOLD','ABSTAIN']},'cut_valid':{'type':'string','enum':['yes','no','uncertain']},'reason':{'type':'string','enum':['complete_turn','incomplete_utterance','hesitation','backchannel','interruption','overlap','bad_cut','insufficient_context','other']},'evidence':{'type':'string'}},'required':['label','cut_valid','reason','evidence'],'additionalProperties':False}
    command=[str(binary),'-m',str(ROOT/'models/teacher-qwen3-omni/Qwen3-Omni-30B-A3B-Thinking-Q4_K_M.gguf'),'--mmproj',str(ROOT/'models/teacher-qwen3-omni/mmproj-Qwen3-Omni-30B-A3B-Thinking-bf16.gguf'),'--no-mmproj-offload','-ngl','0','-c','4096','-t','6','--temp','0','--seed','17','-n','256','--json-schema',json.dumps(schema),'--system-prompt',SYSTEM]
    identity=digest({'pilot':pilot['identity'],'command':command,'runtime':runtime,'source_sha':sha(__file__),'instruction':INSTRUCTION})
    atomic(base/'runtime.json',{'identity':identity,'command':command,'parent_runtime':runtime,'pilot_identity':pilot['identity'],'source_sha256':sha(__file__)})
    results=[];consecutive=0;began=time.monotonic();child=None
    def progress(status,**kw):
        atomic(dest/'status.json',{'status':status,'identity':identity,'completed_calls':len(results),**kw})
    def stop(signum,frame):
        if child and child.poll() is None:child.terminate()
        progress('interrupted',signal=signum);event('D024','interrupted',wandb_status='pending');raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    event('D024','running',runner_pid=os.getpid(),source_sha256=sha(__file__),runtime_sha256=sha(base/'runtime.json'),result_path=str(dest))
    for case in pilot['cases'][:a.limit]:
        for view in ['prefix','hindsight']:
            output=dest/f"{case['case']}_{view}.json"
            if output.exists():
                row=json.loads(output.read_text())
                if row['identity']!=identity:raise ValueError('resume identity mismatch')
            else:
                if time.monotonic()-began>45*60:progress('interrupted',reason='wall clock allowance');event('D024','interrupted',wandb_status='pending');return
                names=[('target_prefix','TARGET speaker up to the decision point (END of this clip).')]
                if view=='hindsight':names += [('other_prefix','OTHER speaker aligned with first clip.'),('target_future','TARGET speaker immediately AFTER the decision point.'),('other_future','OTHER speaker in the same future interval.')]
                prompt=INSTRUCTION+('\nUse only the audible prefix; abstain if ambiguous.' if view=='prefix' else '\nThis is retrospective labeling. Later audio may clarify the earlier intent; resuming later does not by itself prove HOLD.')
                paths=[]
                for key,description in names:
                    path=case['audio'][key]
                    if sha(path)!=case['audio_sha256'][key]:raise ValueError('audio changed')
                    paths.append(path);prompt+='\n'+description+'\n<__media__>\n'
                stem=dest/f"{case['case']}_{view}";prompt_path=stem.with_suffix('.prompt.txt');prompt_path.write_text(prompt)
                progress('running',active_case=case['case'],active_view=view);start=time.monotonic();error=None;judgment=None
                with stem.with_suffix('.stdout').open('w') as stdout,stem.with_suffix('.stderr').open('w') as stderr:
                    child=subprocess.Popen(command+['--audio',','.join(paths),'-f',str(prompt_path)],stdout=stdout,stderr=stderr)
                    try:code=child.wait(timeout=min(120,max(1,45*60-(time.monotonic()-began))))
                    except subprocess.TimeoutExpired:child.kill();child.wait();code=-9;error='call/total deadline exceeded'
                text=stem.with_suffix('.stdout').read_text()
                try:
                    if code!=0:raise ValueError('console exit '+str(code))
                    judgment=validate_judgment(text.strip())
                except Exception as e:error=error or str(e)
                row={'identity':identity,'case':case['case'],'view':view,'status':'complete' if judgment else 'failed','judgment':judgment,'elapsed_s':time.monotonic()-start,'error':error,'response':{'local_stdout':str(stem.with_suffix('.stdout'))}}
                atomic(output,row)
            results.append(row);consecutive=consecutive+1 if row['status']=='failed' else 0
            print(json.dumps({k:row[k] for k in ['case','view','status','elapsed_s']}),flush=True)
            if consecutive>=3:
                progress('early-stopped',reason='3 consecutive failed calls');event('D024','inconclusive',decision='Three consecutive failed CPU calls; preserve observations',wandb_status='pending');return
    status='pilot-complete' if a.limit==len(pilot['cases']) else 'partial-smoke'
    progress(status);event('D024',status,metrics={'attempts':len(results),'valid_outputs':sum(r['status']=='complete' for r in results),'human_reviews_completed':0},wandb_status='pending')

if __name__=='__main__':main()
