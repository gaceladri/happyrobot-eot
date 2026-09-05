#!/usr/bin/env python
"""Build auditable, source-capped controls from existing audio; never touch benchmark audio."""
import json, random
from collections import Counter,defaultdict
from pathlib import Path
from eot.data import read_samples, atomic_write_jsonl

rng=random.Random(17)
root=Path('data/optimization');root.mkdir(exist_ok=True)
mined=read_samples(Path('data/mined/smart-turn-ensemble/samples.jsonl'))
# A stricter follow-up experiment, not a retroactive change to R2's registered rules.
# Medium/high final clip labels are retained; internal cuts require high agreement.
rows=[r for r in mined if r['cut_confidence']=='high']
groups=defaultdict(list)
for r in rows: groups[r['source']].append(r)
cap=1500
rows=[]
for source,rs in sorted(groups.items()):
 rng.shuffle(rs); rows.extend(rs[:cap])
# Metadata-only repair: unobserved EOT futures are masked by the dataset loader too.
for r in rows:
 if r['label']==1 or r['kind']=='whole': r['fvad_mask']=[0]*4
 else: r['fvad_mask']=[1]*4
finals={r['clip_id']:r for r in mined if r['kind'] in ('final','whole')}
whole=[]
for r in rows:
 f=finals[r['clip_id']]
 whole.append({**f,'id':'whole-control:'+r['id'],'fvad_mask':[0]*4})
app=read_samples(Path('data/mined/apptek-oracle/train/samples.jsonl'))
app=[r for r in app if r['cut_confidence']=='high']
# Balanced source allocation and identical total updates in the training commands.
ag=defaultdict(list)
for r in app: ag[r['source']].append(r)
app=[]
for source,rs in sorted(ag.items()):
 rng.shuffle(rs); app.extend(rs[:max(1,len(rows)//len(ag))])
for r in app:
 if r['label']==1: r['fvad_mask']=[0]*4 # exact future not retained in old artifacts
sets={'mined':rows,'whole':whole,'mixed':rows+app}
for name,rs in sets.items():
 atomic_write_jsonl(root/f'{name}.jsonl',rs)
report={k:{'n':len(v),'labels':dict(Counter(r['label'] for r in v)),
 'sources':dict(Counter(r['source'] for r in v)), 'max_source_share':max(Counter(r['source'] for r in v).values())/len(v)} for k,v in sets.items()}
(root/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
