from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest

from eot.prefix_mining import fvad_targets
from eot.krisp import evaluate
from eot.policy import Endpointer, Policy


def test_future_speech_does_not_follow_eot_label():
    assert fvad_targets(None, is_eot=True) == ([0]*4,[0]*4)
    assert fvad_targets(None, observed_until=.7) == ([0]*4,[1,1,0,0])
    assert fvad_targets(.9, is_eot=True) == ([0,0,1,1],[1]*4)


def test_krisp_counts_timeout_cutoff_and_excludes_boundary():
    holds=[{'label':'hold','span_len':1.5,'points':[(.2,0.)]}]
    eots=[{'label':'eot','span_len':2.,'points':[(.2,0.)]}]
    assert evaluate(holds+eots,.5,.2,1.)['cutoff_rate']==1.
    assert evaluate(holds+eots,.5,.2,1.5)['cutoff_rate']==0.


def test_bootstrap_matches_harness_for_both_modes():
    pytest.importorskip('pandas')
    pytest.importorskip('sklearn')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/eot-bench'))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import pandas as pd
    from eot_harness.metrics import compute_metrics_from_predictions
    from eotbench_bootstrap import contributions, bootstrap
    rows=[]
    for tid in range(4):
        for span,label in [(0,'hold'),(1,'eot')]:
            for t in (.2,.3,.4,.5,.6):
                rows.append(dict(id=str(tid),span_index=span,label=label,timestamp=span+t,
                                 silence_dur=t,p_eot=.8 if (label=='eot' or tid==0) and t>=.4 else .1))
    df=pd.DataFrame(rows)
    pol=dict(threshold=.5,action_delay=.2,timeout=.5)
    for mode in (.2,None):
        tr,sm=compute_metrics_from_predictions(df,score_point_s=mode,thresholds=[.5],action_delays=[.2],timeouts=[.5])
        expected=tr[tr.policy_type=='model'].iloc[0]
        a=contributions(df,sm,pol)
        result=bootstrap(a,100,0,a)
        assert result['cutoff_rate']==pytest.approx(expected.cutoff_rate)
        assert result['mean_latency']==pytest.approx(expected.mean_latency)
        assert result['paired_diff_cutoff_rate_ci95']==[0.,0.]


def test_endpointer_timeout_without_inference_and_stale_generation():
    e=Endpointer(Policy(timeout=1.))
    e.on_vad(-.1,True)
    e.on_vad(0.,False)
    generation=e.generation
    e.on_vad(.3,True)
    e.on_vad(.4,False)
    assert not e.on_score(.8,1.,generation)
    assert not e.on_tick(1.39)
    assert e.on_tick(1.41)
    assert not e.on_tick(2.)


def test_endpointer_latches_early_score_until_action_delay():
    e=Endpointer(Policy(threshold=.5,action_delay=.5,timeout=1.))
    e.on_vad(0.,True)
    e.on_vad(.1,False)
    g=e.request_score(.31)
    assert g is not None
    assert e.request_score(.32) is None
    assert not e.on_score(.35,.8,g)
    assert not e.should_score(.5)
    assert not e.on_tick(.59)
    assert e.on_tick(.61)
    e.reset()
    e.on_vad(1.,False)
    assert not e.should_score(2.)
    assert not e.on_tick(3.)


def test_future_target_rejects_cut_after_onset():
    with pytest.raises(ValueError,match='after the next speech onset'):
        fvad_targets(-.02)


def test_mining_does_not_cut_after_earlier_vad_onset():
    from eot.audio import PauseSpan,Span
    from eot.prefix_mining import mine_clip,Clip
    class Detector:
        detector='fake'
        def __call__(self,x,sr):
            return [PauseSpan(.5,.72,.69,.9,'high',Span(.5,.72),Span(.52,.69),.52)]
    samples=mine_clip(Clip('boundary',np.ones(16000,np.float32)*.1,1),detector=Detector())
    assert not [s for s in samples if s.kind=='internal']


def test_vectorized_krisp_sweep_matches_scalar_replay():
    from eot.krisp import sweep
    rng=np.random.default_rng(7)
    spans=[]
    for i in range(30):
        duration=float(rng.choice([.3,.5,1.,1.2,2.5]))
        points=[(round(float(t),1),float(rng.uniform())) for t in np.arange(.2,duration+1e-6,.1)]
        spans.append(dict(label='hold' if i%2 else 'eot',span_len=duration,points=points))
    fast=sweep(spans)
    for row in fast[::23]:
        ref=evaluate(spans,row['threshold'],row['action_delay'],row['timeout'])
        for key in ['cutoff_rate','mean_latency','median_latency','detect_rate','timeout_rate']:
            assert row[key]==pytest.approx(ref[key])
