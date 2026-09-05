#!/usr/bin/env python
"""Collect compatible runs across revision-qualified output directories; never silently mix spans."""
import argparse, json, shutil
from pathlib import Path
import pandas as pd
from eot_harness.metrics import compute_metrics_from_predictions


def collect(root, reference, out):
    expected=pd.read_parquet(reference/'span_set.parquet')
    # Match prediction keys as well as the canonical span table, since dataset revisions can differ.
    refpred=next(reference.glob('smart_turn*/predictions.parquet'))
    keys=['id','span_index','timestamp','silence_dur','label']
    expected_keys=pd.read_parquet(refpred)[keys].sort_values(keys).reset_index(drop=True)
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(reference/'span_set.parquet',out/'span_set.parquet')
    count=0
    for p in sorted(root.glob('*/en/*/predictions.parquet')):
        manifest=json.loads((p.parent/'manifest.json').read_text())
        if manifest.get('adapter')!='eot.eotbench_adapter:EOTAdapter': continue
        df=pd.read_parquet(p)
        if not df[keys].sort_values(keys).reset_index(drop=True).equals(expected_keys):
            raise ValueError(f'prediction points do not match published reference: {p}')
        dest=out/p.parent.name; dest.mkdir(exist_ok=True)
        for name in ('predictions.parquet','manifest.json'): shutil.copy2(p.parent/name,dest/name)
        metrics=dest/'metrics'; metrics.mkdir(exist_ok=True)
        tr,s=compute_metrics_from_predictions(df,score_point_s=float(manifest['score_point']))
        tr.to_parquet(metrics/'tradeoff.parquet',index=False)
        (metrics/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
        count+=1
    for p in reference.iterdir():
        if p.is_dir() and (p/'metrics/summary.json').exists():
            shutil.copytree(p,out/p.name,dirs_exist_ok=True)
    if count==0: raise ValueError('no local model predictions found')
    print(f'collected {count} local runs with matching prediction keys into {out}')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); collect(a.root,a.reference,a.out)
