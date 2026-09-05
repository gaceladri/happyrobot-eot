#!/usr/bin/env python
"""Compare the shipped CPU ONNX against every saved benchmark score at the policy score point."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Audio,load_dataset
from eot.eotbench_adapter import EOTAdapter,score_rows
from eot.data import sha256_file

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--thresholds',type=float,nargs='+',default=[.5])
a=p.parse_args()
ds=load_dataset('livekit/eot-bench-data',name='en',split='validation',
                revision='ca9d98a9686b920a2d8c9eb984224ba9be74e4dd').cast_column('audio',Audio(decode=False))
adapter=EOTAdapter(onnx=str(a.model))
# Each pause contributes its first 200 ms score; a large step skips later grid points.
got=pd.DataFrame(list(score_rows(adapter,ds,grid_step=1000.,batch_size=16)))
ref=pd.read_parquet(a.predictions);ref=ref[np.isclose(ref.silence_dur,.2)]
keys=['id','span_index'];joined=got.merge(ref,on=keys,suffixes=('_onnx','_torch'),validate='one_to_one',how='outer',indicator=True)
if not (joined['_merge']=='both').all(): raise ValueError('score-point keys do not match')
if not (joined.label_onnx==joined.label_torch).all(): raise ValueError('labels differ')
drift=np.abs(joined.p_eot_onnx-joined.p_eot_torch)
flips={str(t):int(((joined.p_eot_onnx>t)!=(joined.p_eot_torch>t)).sum()) for t in a.thresholds}
result=dict(model=str(a.model),sha256=sha256_file(a.model),reference_predictions=str(a.predictions),
            dataset_revision="ca9d98a9686b920a2d8c9eb984224ba9be74e4dd",score_point_seconds=.2,n_points=len(joined),
            max_probability_drift=float(drift.max()),mean_probability_drift=float(drift.mean()),
            flips_at_0_5=int(((joined.p_eot_onnx>=.5)!=(joined.p_eot_torch>=.5)).sum()),flips_by_threshold=flips,passed=bool(drift.max()<1e-4 and not any(flips.values())))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if not result['passed']:raise SystemExit('served score parity failed')
