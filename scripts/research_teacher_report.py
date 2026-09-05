"""Summarize teacher observations without treating silver labels as ground truth."""
import argparse,json,math,os
from pathlib import Path
from collections import Counter
from eot.data import atomic_write_json,atomic_write_jsonl
from eot.audio_teacher import digest

def interval(k,n):
    if not n:return None
    z=1.96;den=1+z*z/n;mid=(k/n+z*z/(2*n))/den;half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
    return [max(0,mid-half),min(1,mid+half)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    pilot=json.loads(a.pilot.read_text());cases=pilot['cases'];groups={};review=[];latencies=[]
    for case in cases:
        judged={}
        for view in ['prefix','hindsight']:
            path=a.results/f"{case['case']}_{view}.json"
            if not path.exists():continue
            r=json.loads(path.read_text());group=groups.setdefault(case['stratum']+'/'+view,Counter())
            group['attempted']+=1;latencies.append(r['elapsed_s'])
            if r['status']!='complete':group['failed']+=1;continue
            j=r['judgment'];judged[view]=j;group['valid']+=1;group[j['label']]+=1;group['reason/'+j['reason']]+=1
            if j['label']!='ABSTAIN':
                group['decided']+=1
                if int(j['label']=='EOT')!=case['silver_label']:group['silver_disagreements']+=1
        triggers=[]
        if any(j['label']!='ABSTAIN' and int(j['label']=='EOT')!=case['silver_label'] for j in judged.values()):triggers.append('teacher_vs_silver')
        if len(judged)==2 and judged['prefix']['label']!=judged['hindsight']['label']:triggers.append('context_changes_label')
        if any(j['label']=='ABSTAIN' for j in judged.values()):triggers.append('teacher_abstains')
        if (case['student_p_eot']>=.5)!=bool(case['silver_label']):triggers.append('student_vs_silver')
        if triggers:review.append({'case':case['case'],'id':case['id'],'stratum':case['stratum'],'triggers':triggers,'silver_label':case['silver_label'],'silver_reason':case['silver_reason'],'student_p_eot':case['student_p_eot'],'teacher':judged,'adjudicated_label':None,'reviewer':None,'confirmed_failure':None})
    table=[]
    for key,g in sorted(groups.items()):
        table.append({'group':key,**g,'silver_disagreement_fraction':g['silver_disagreements']/g['decided'] if g['decided'] else None,'descriptive_wilson95':interval(g['silver_disagreements'],g['decided'])})
    summary={'pilot_identity':pilot['identity'],'groups':table,'attempts':len(latencies),'valid_outputs':sum(g['valid'] for g in groups.values()),'review_queue_size':len(review),'human_reviews':0,'confirmed_errors':0,'label_scale_gate_passed':False,'limitations':'Silver disagreement is not teacher error or accuracy. Selected strata are diagnostic. Wilson intervals treat cases as independent and understate conversation clustering. Teacher calls have extra compute and hindsight inputs; not an EoT benchmark or serving result.'}
    if latencies:
        import numpy as np
        summary['teacher_call_seconds']={'mean':float(np.mean(latencies)),'p50':float(np.percentile(latencies,50)),'p95':float(np.percentile(latencies,95))}
    a.out.mkdir(parents=True,exist_ok=True);atomic_write_json(a.out/'summary.json',summary);atomic_write_jsonl(a.out/'adjudication_queue.jsonl',review)
    # A small local review surface: make a blind judgment before revealing machine proposals.
    import numpy as np
    import soundfile as sf
    display=[]
    for case in cases:
        channels=[]
        for who in ['target','other']:
            parts=[sf.read(case['audio'][who+'_'+view],dtype='float32')[0] for view in ['prefix','future']]
            channels.append(np.concatenate(parts))
        if len(channels[0])!=len(channels[1]):raise ValueError('review channels misaligned')
        dest=a.out/f"review_{case['case']}.wav";tmp=dest.with_suffix('.tmp.wav');sf.write(tmp,np.stack(channels,axis=1),16000,subtype='PCM_16');os.replace(tmp,dest)
        item=next((r for r in review if r['case']==case['case']),None)
        display.append({'case':case['case'],'audio':dest.name,'marker':len(sf.read(case['audio']['target_prefix'])[0])/16000,'proposals':item or {'silver_label':case['silver_label'],'student_p_eot':case['student_p_eot']}})
    html='''<!doctype html><meta charset="utf-8"><title>D024 revisión ciega</title>
<style>body{font:17px system-ui;max-width:900px;margin:30px auto}button,select,input{font:inherit;margin:8px;padding:8px}audio{width:100%}pre{white-space:pre-wrap}</style>
<h1>D024: aprender de los desacuerdos</h1><p>Persona objetivo: canal izquierdo. Escucha el contexto y registra tu juicio antes de revelar las propuestas. Son casos de entrenamiento seleccionados; esta revisión no mide precisión global.</p>
<label>Revisor <input id="reviewer"></label><h2 id="title"></h2><audio id="audio" controls></audio><p id="marker"></p>
<label>En el punto marcado <select id="label"><option value="">Sin revisar</option><option>EOT</option><option>HOLD</option><option>ABSTAIN</option></select></label>
<label>Corte válido <select id="cut"><option value="">Sin revisar</option><option>yes</option><option>no</option><option>uncertain</option></select></label><br>
<label>Observación <input id="note" size="65"></label><br><button id="prev">Anterior</button><button id="next">Siguiente</button><button id="reveal">Mostrar propuestas después de juzgar</button><button id="export">Exportar</button><pre id="proposal"></pre>
<script>const cases=DISPLAY_DATA,identity='PILOT_ID',key='teacher-review-'+identity;let i=0,answers=JSON.parse(localStorage.getItem(key)||'{}');const el=id=>document.getElementById(id);
function save(){answers[cases[i].case]={case:cases[i].case,label:el('label').value,cut_valid:el('cut').value,note:el('note').value,reviewer:el('reviewer').value,proposals_revealed:answers[cases[i].case]?.proposals_revealed||false};localStorage.setItem(key,JSON.stringify(answers));}
function show(){let c=cases[i],a=answers[c.case]||{};el('title').textContent='Caso '+c.case+' ('+(i+1)+'/'+cases.length+')';el('audio').src=c.audio;el('marker').textContent='Punto de decisión: '+c.marker.toFixed(3)+' segundos';el('label').value=a.label||'';el('cut').value=a.cut_valid||'';el('note').value=a.note||'';el('proposal').textContent='';}
el('prev').onclick=()=>{save();i=Math.max(0,i-1);show();};el('next').onclick=()=>{save();i=Math.min(cases.length-1,i+1);show();};for(let k of ['label','cut','note'])el(k).onchange=save;
el('reveal').onclick=()=>{save();if(!el('label').value||!el('cut').value){el('proposal').textContent='Registra primero tu juicio.';return;}answers[cases[i].case].proposals_revealed=true;localStorage.setItem(key,JSON.stringify(answers));el('proposal').textContent=JSON.stringify(cases[i].proposals,null,2);};
el('export').onclick=()=>{save();let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:'teacher-adjudication-v1',pilot_identity:identity,answers:Object.values(answers)},null,2)],{type:'application/json'}));a.download='D024_human_adjudication.json';a.click();URL.revokeObjectURL(a.href);};show();</script>'''
    # Escape embedded script boundaries from model-produced evidence; render with textContent only.
    html=html.replace('DISPLAY_DATA',json.dumps(display).replace('<','\\u003c')).replace('PILOT_ID',pilot['identity'])
    tmp=a.out/'.review.tmp.html';tmp.write_text(html);tmp.replace(a.out/'review.html')
    if table:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5));x=list(range(len(table)));bottom=[0]*len(x)
        for label,color in [('EOT','#4c78a8'),('HOLD','#59a14f'),('ABSTAIN','#f28e2b'),('failed','#e15759')]:
            vals=[g.get(label,0) for g in table];ax.bar(x,vals,bottom=bottom,label=label,color=color);bottom=[a+b for a,b in zip(bottom,vals)]
        ax.set_xticks(x,[g['group'].replace('/','\n') for g in table]);ax.set_ylabel('Cases');ax.set_title('D024 audio teacher pilot — proposals, not reviewed gold');ax.legend();fig.tight_layout()
        tmp=a.out/'.teacher_counts.tmp.png';fig.savefig(tmp,dpi=160);tmp.replace(a.out/'teacher_counts.png');plt.close(fig)
    print(json.dumps(summary))

if __name__=='__main__':main()
