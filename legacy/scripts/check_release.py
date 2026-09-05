#!/usr/bin/env python
"""Report measured demo readiness separately from conversational production readiness."""
import argparse,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--performance',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--score-parity',type=Path,default=Path('eval/deployment_point_parity.json'))
    p.add_argument('--audit',type=Path,default=Path('eval/labeling_qa/listening_audit/listening_audit.jsonl'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    meta=json.loads(a.model.with_suffix('.json').read_text())
    sha=hashlib.sha256(a.model.read_bytes()).hexdigest()
    matches=[r for r in json.loads(a.performance.read_text())['results'] if r['health']['model_sha']==sha[:16] and r['threads']==a.threads]
    perf=matches[-1] if matches else None
    audit=[json.loads(l) for l in a.audit.read_text().splitlines()] if a.audit.exists() else []
    answered=sum(type(r.get('cut_ok')) is bool and type(r.get('sounds_complete')) is bool for r in audit)
    parity=meta.get('accepted') is True and meta.get('sha256')==sha
    trace=json.loads(a.score_parity.read_text()) if a.score_parity.exists() else {}
    trace_parity=trace.get('passed') is True and trace.get('sha256')==sha
    latency=perf is not None and all(r['errors']==0 and r['client_total_ms']['p95']<100 for r in perf['levels'])
    result=dict(model=str(a.model),sha256=sha,threads=a.threads,export_parity_passed=parity,
        benchmark_score_parity_passed=trace_parity,
        request_p95_under_100ms_no_errors=latency,demo_ready=parity and trace_parity and latency,
        audit_total=len(audit),audit_answered=answered,
        production_ready=False,production_blockers=['human label audit and adjudication incomplete' if answered<len(audit) or not audit else 'human audit results require review',
            'low-cutoff conversational latency target not yet demonstrated on representative production calls',
            'AppTek supervision is heuristic role-play; training-source suitability must be resolved before production'],
        performance=perf)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='performance'},indent=2))
if __name__=='__main__':main()
