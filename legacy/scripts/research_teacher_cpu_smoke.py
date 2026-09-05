"""P023: bounded CPU console probe using pinned local assets; no sockets or downloads."""
import json, subprocess, time, sys, os
from pathlib import Path
import numpy as np
import soundfile as sf
from research import ROOT, atomic, sha, event

out=ROOT/'data/research/P023';out.mkdir(parents=True,exist_ok=True)
runtime=json.loads((ROOT/'data/research/D022/runtime.json').read_text())
for p,h in runtime['files'].items():
    if sha(p)!=h:raise ValueError('Teacher weights changed')
cases=json.loads((ROOT/'data/research/D022/pilot.json').read_text())['cases']
path=max((Path(c['audio']['target_prefix']) for c in cases),key=lambda p:float(np.mean(sf.read(p,dtype='float32')[0]**2)))
x,sr=sf.read(path,dtype='float32');silent=out/'silence.wav';sf.write(silent,np.zeros_like(x),sr,subtype='PCM_16')
schema={'type':'object','properties':{'speech_present':{'type':'boolean'}},'required':['speech_present'],'additionalProperties':False}
prompt=out/'prompt.txt';prompt.write_text('Is human speech audible in this audio? Return only JSON: {"speech_present": true} or {"speech_present": false}.')
binary=ROOT/'artifacts/research/tools/llama.cpp/build-cuda129/bin/llama-mtmd-cli'
base=[str(binary),'-m',str(ROOT/'models/teacher-qwen3-omni/Qwen3-Omni-30B-A3B-Thinking-Q4_K_M.gguf'),'--mmproj',str(ROOT/'models/teacher-qwen3-omni/mmproj-Qwen3-Omni-30B-A3B-Thinking-bf16.gguf'),'--no-mmproj-offload','-ngl','0','-c','4096','-t','6','--temp','0','--seed','17','-n','256','--json-schema',json.dumps(schema),'-f',str(prompt)]
identity={'runtime':runtime,'binary_sha256':sha(binary),'script_sha256':sha(__file__),'prompt_sha256':sha(prompt),'command':base,'controls':{'speech_sha256':sha(path),'silence_sha256':sha(silent)}}
atomic(out/'runtime.json',identity)
results=[]
for name,wav in [('speech',path),('silence',silent)]:
    record=out/f'{name}.json'
    if record.exists():
        raise SystemExit('Existing control observation; do not silently repeat probe')
    event('P023','running',active_control=name,runner_pid=os.getpid(),runtime_sha256=sha(out/'runtime.json'))
    started=time.monotonic();status='failed';error=None;parsed=None;returncode=None
    with (out/f'{name}.stdout').open('w') as stdout,(out/f'{name}.stderr').open('w') as stderr:
        try:
            r=subprocess.run(base+['--audio',str(wav)],stdout=stdout,stderr=stderr,timeout=240)
            returncode=r.returncode
        except subprocess.TimeoutExpired:error='timeout after240seconds'
    text=(out/f'{name}.stdout').read_text();decoder=json.JSONDecoder()
    for i,ch in enumerate(text):
        if ch!='{':continue
        try:
            obj,_=decoder.raw_decode(text[i:])
            if isinstance(obj,dict) and set(obj)=={'speech_present'} and isinstance(obj['speech_present'],bool):parsed=obj
        except ValueError:pass
    passed=returncode==0 and parsed is not None and parsed['speech_present']==(name=='speech')
    observation={'control':name,'passed':passed,'answer':parsed,'elapsed_s':time.monotonic()-started,'returncode':returncode,'error':error}
    atomic(record,observation);results.append(observation);print(json.dumps(observation),flush=True)
    if not passed:break
summary={'passed':len(results)==2 and all(r['passed'] for r in results),'controls':results,'limitation':'CPU console with grammar-constrained decoding differs from preregistered D022 GPU HTTP runtime. Audio wiring only; no label-precision claim.'}
atomic(out/'summary.json',summary)
event('P023','probe-complete' if summary['passed'] else 'inconclusive',metrics={'controls_attempted':len(results),'controls_passed':sum(r['passed'] for r in results),'total_seconds':sum(r['elapsed_s'] for r in results)},decision='CPU audio wiring passed' if summary['passed'] else 'CPU feasibility/audio wiring probe failed; inspect local stderr before any retry',result_path=str(out/'summary.json'),wandb_status='pending')
