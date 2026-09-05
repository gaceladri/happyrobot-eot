"""Flag teacher proposals that conflict with observed rapid own-speaker continuation.

Flags are review priorities, not independently adjudicated truth or automatic relabeling.
"""
import json
from pathlib import Path
from research import ROOT,atomic,sha

pilot=json.loads((ROOT/'data/research/D022/pilot.json').read_text())
manifest=ROOT/'data/research/D019/mined/samples.jsonl'
if sha(manifest)!=pilot['provenance']['manifest_sha256']:raise ValueError('source manifest changed')
samples={r['id']:r for r in map(json.loads,manifest.read_text().splitlines())}
flags=[];attempts=0
for case in pilot['cases']:
    sample=samples[case['id']]
    for view in ['prefix','hindsight']:
        path=ROOT/'data/research/D024/teacher'/f"{case['case']}_{view}.json"
        if not path.exists():continue
        result=json.loads(path.read_text());j=result.get('judgment');attempts+=1
        tto=sample.get('time_to_onset')
        if j and j['label']=='EOT' and tto is not None and 0<=tto<=.3 and sample['label_reason'] in {'own_continuation','own_continuation_after_backchannel'}:
            flags.append({'case':case['case'],'id':case['id'],'view':view,'flag':'eot_despite_rapid_own_continuation','time_to_onset_s':tto,'current_rule':sample['label_reason'],'judgment':j,'independently_adjudicated':False,'action':'withhold teacher override; inspect both channels and exact cut before changing labels'})
out=ROOT/'data/research/D024/report/temporal_consistency.json'
atomic(out,{'threshold_s':.3,'threshold_selected_after_case':'002','post_hoc_diagnostic':True,'attempts':attempts,'flags':flags,'flagged_calls':len(flags),'flagged_cases':len({r['case'] for r in flags}),'limitation':'Source labels and onset detector are silver evidence; this is a post-hoc safety filter for proposed relabeling, not new ground truth or an official benchmark metric.'})
print(json.dumps({'attempts':attempts,'flagged_calls':len(flags),'flagged_cases':len({r['case'] for r in flags})}))
