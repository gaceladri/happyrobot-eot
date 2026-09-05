#!/usr/bin/env python
"""Cluster bootstrap using each model's own fixed policy and score mode.

Intervals on full-benchmark optimized policies are descriptive, conditional on selection.
They are not independent-test confidence intervals. Use --calibration-fraction for policies
selected on separate turns and evaluated on the remaining turns.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from eot_harness.metrics import (
    EPS, span_table_from_predictions, official_score_table, compute_metrics_from_predictions,
    _first_crossing_times, _fire_times_from_crossing, _score_point_fire_times,
)


def contributions(df, summary, policy):
    """Per-turn sufficient statistics; exactly the pinned harness's span denominators."""
    spans = span_table_from_predictions(df, min_hold_span_duration=summary['min_hold_span_duration'],
                                       max_hold_span_duration=summary['max_hold_span_duration'])
    if summary['score_point_s'] is None:
        fire = _fire_times_from_crossing(spans, _first_crossing_times(df, threshold=policy['threshold']))
    else:
        spans = official_score_table(df, spans=spans, score_point_s=summary['score_point_s'])
        fire = _score_point_fire_times(spans, threshold=policy['threshold'])
    hold = spans.label.to_numpy() == 'hold'
    dur = spans.span_dur.to_numpy()
    model_fire = np.maximum(policy['action_delay'], fire)
    cut = hold & ((dur > policy['timeout'] + EPS) | (dur > model_fire + EPS))
    latency = np.where(~hold, np.minimum(model_fire, policy['timeout']), 0)
    return pd.DataFrame({'id': spans.id, 'cut': cut.astype(int), 'hold': hold.astype(int),
                         'latency': latency, 'eot': (~hold).astype(int)}).groupby('id').sum()


def policies(summary, tradeoff=None):
    out = {f'latency_at_{k}': v for k, v in summary['operating_points'].items()}
    if tradeoff is not None:
        tr = tradeoff[tradeoff.policy_type == 'model']
        for b in (0.3, 0.6):
            f = tr[tr.mean_latency <= b + EPS]
            out[f'cutoff_at_{b}s'] = None if f.empty else f.sort_values(['cutoff_rate','mean_latency']).iloc[0].to_dict()
    return out


def bootstrap(a, reps, seed, b=None):
    if reps < 1 or len(a) < 2:
        raise ValueError('need positive reps and at least two turns')
    if b is not None and set(a.index) != set(b.index):
        raise ValueError('paired models must have identical turn ids')
    weights = np.random.default_rng(seed).multinomial(len(a), np.full(len(a), 1 / len(a)), size=reps)
    def rates(x, w):
        z = w @ x[['cut', 'hold', 'latency', 'eot']].to_numpy()
        return z[..., 0] / z[..., 1], z[..., 2] / z[..., 3]
    c,l = rates(a, weights)
    pc,pl = rates(a, np.ones(len(a)))
    out = dict(cutoff_rate=float(pc), mean_latency=float(pl),
               cutoff_rate_ci95=np.percentile(c,[2.5,97.5]).tolist(),
               mean_latency_ci95=np.percentile(l,[2.5,97.5]).tolist())
    if b is not None:
        bc,bl = rates(b.reindex(a.index),weights)
        out.update(paired_diff_cutoff_rate_ci95=np.percentile(c-bc,[2.5,97.5]).tolist(),
                   paired_diff_mean_latency_ci95=np.percentile(l-bl,[2.5,97.5]).tolist(),
                   paired_diff_cutoff_rate_mean=float(np.mean(c-bc)),
                   paired_diff_mean_latency_mean=float(np.mean(l-bl)))
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--paired-with',type=Path)
    ap.add_argument('--reps',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--calibration-fraction',type=float,default=0.)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    def load(p):
        return (pd.read_parquet(p/'predictions.parquet'),
                json.loads((p/'metrics/summary.json').read_text()),
                pd.read_parquet(p/'metrics/tradeoff.parquet'))
    df,s,tr=load(args.run_dir)
    other=load(args.paired_with) if args.paired_with else None
    note='conditional_on_full_benchmark_policy_selection; descriptive_only'
    calibration_ids=[]
    if args.calibration_fraction:
        if not 0 < args.calibration_fraction < 1:
            raise ValueError('calibration fraction must be between 0 and 1')
        ids=sorted(df.id.unique()); np.random.default_rng(args.seed).shuffle(ids)
        calibration_ids=ids[:round(len(ids)*args.calibration_fraction)]
        def split(d,summary):
            cal=d[d.id.isin(calibration_ids)]; test=d[~d.id.isin(calibration_ids)]
            trade,sm=compute_metrics_from_predictions(cal, score_point_s=summary['score_point_s'],
                min_hold_span_duration=summary['min_hold_span_duration'], max_hold_span_duration=summary['max_hold_span_duration'])
            return test,sm,trade
        df,s,tr=split(df,s)
        if other: other=split(other[0],other[1])
        note='policies_selected_on_disjoint_calibration_turns; exploratory_split_of_previously_inspected_benchmark'
    pols=policies(s,tr); opols=policies(other[1],other[2]) if other else {}
    points={}
    for name,pol in pols.items():
        if pol is None: continue
        op=opols.get(name)
        if other and op is None: continue
        a=contributions(df,s,pol)
        b=contributions(other[0],other[1],op) if other else None
        points[name]={**bootstrap(a,args.reps,args.seed,b), 'policy':pol,
                      'paired_policy':op, 'score_mode':s['score_mode']}
    result=dict(run_dir=str(args.run_dir),paired_with=str(args.paired_with) if other else None,
                n_turns=int(df.id.nunique()),reps=args.reps,interpretation=note,
                calibration_ids=calibration_ids,points=points)
    out=args.out or args.run_dir/'metrics'/('bootstrap_paired.json' if other else 'bootstrap.json')
    out.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(out)
if __name__=='__main__': main()
