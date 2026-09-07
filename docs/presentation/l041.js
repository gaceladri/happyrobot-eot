/* Completed L041 results. Every quality/cost pair keeps its checkpoint and runtime. */
(() => {
'use strict';
const D=window.EOT_DATA,N=D.l041,L=D.research.l040,$=s=>document.querySelector(s),NS='http://www.w3.org/2000/svg';
const names=window.EOT_MODEL_NAMES={tiny:'Whisper Tiny · baseline',teacher:'Cohere · GPU',student:'Whisper-base · CPU',compressed:'Whisper-base · CPU INT8',delivered:'Whisper-base · earlier recipe',diagnostic:'Cohere · GPU INT8 trial'};
const ink='#353132',plum='#703f5b',teal='#225e64',stone='#8a817e',paper='#eeedeb';
const f=v=>v.toLocaleString('en-US',{maximumFractionDigits:1}),round=v=>Math.round(v).toLocaleString('en-US');
const el=(tag,a={},t)=>{const e=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));if(t!==undefined)e.textContent=t;return e;};
const text=(r,x,y,t,a={})=>r.append(el('text',{x,y,fill:ink,'font-size':20,...a},t));
const line=(r,x1,y1,x2,y2,a={})=>r.append(el('line',{x1,y1,x2,y2,stroke:'#ccc8c5',...a}));
const native=(mode,t)=>N.native.http.find(r=>r.mode===mode&&r.threads===Number(t)).p95_ms;
const cohere=dev=>N.localCohere.rows.find(r=>r.device===dev).p95_ms;
const legacy=dev=>Object.fromEntries(D.selectedHardware.devices[dev].levels.map(r=>[r.concurrency,r.all_attempts_ms.p95]));
function renderServing(){
 const r=$('#serving-chart'),P=D.deployment;r.replaceChildren();
 const series=[{name:names.student,v:P.cpu.p95_ms,c:teal},{name:names.teacher,v:P.gpu.p95_ms,c:plum}];
 const top=Math.ceil(Math.max(...series.flatMap(s=>Object.values(s.v)))/20)*20;
 const x=c=>100+(c-1)/7*1130,y=v=>425-v/top*345;
 for(let v=0;v<=top;v+=20){line(r,100,y(v),1230,y(v));text(r,78,y(v)+6,v,{'text-anchor':'end','font-size':19});}
 text(r,100,38,'HTTP response p95 (ms)',{'font-size':22});
 for(const c of [1,4,8])text(r,x(c),466,c,{'text-anchor':'middle','font-size':21});
 text(r,665,507,'Concurrent requests',{'text-anchor':'middle','font-size':22});
 for(const s of series){
  r.append(el('path',{d:[1,4,8].map((c,i)=>(i?'L':'M')+x(c)+','+y(s.v[c])).join(''),fill:'none',stroke:s.c,'stroke-width':4}));
  for(const c of [1,4,8]){r.append(el('circle',{cx:x(c),cy:y(s.v[c]),r:6,fill:s.c}));text(r,x(c)+(c===8?-12:12),y(s.v[c])+(s===series[0]?-15:32),f(s.v[c]),{'text-anchor':c===8?'end':'start','font-size':25,'font-weight':700,fill:s.c});}
 }
 $('#hardware-legend').innerHTML=series.map(s=>`<span><i style="background:${s.c}"></i>${s.name}</span>`).join('');
 $('#hardware-source').textContent=`Intel i9-10900X · RTX 3090 · 400 warm requests per load level · Docker · full HTTP including preprocessing and queuing.`;
}
function renderCombinations(){
 const r=$('#ensemble-chart');r.replaceChildren();const x=v=>300+(v-580)/120*480;
 text(r,0,28,'Same benchmark · each selected policy',{'font-size':19,fill:stone});
 for(const [i,[label,v,color]] of [['Best individual teacher',615,plum],['Prediction ensemble',640.25,stone],['Averaged weights',660,stone]].entries()){
 const y=95+i*82;line(r,300,y,780,y);text(r,0,y+7,label,{'font-size':22});r.append(el('circle',{cx:x(v),cy:y,r:7,fill:color}));text(r,x(v)+16,y-12,f(v)+' ms',{'font-size':24,'font-weight':700,fill:color});}
 for(const v of [580,600,620,640,660,680,700])text(r,x(v),318,v,{'text-anchor':'middle','font-size':17});text(r,540,358,'Mean EoT waiting (ms) · lower is better',{'text-anchor':'middle','font-size':20});
}
function renderQAT(){
 const r=$('#qat-chart');r.replaceChildren();const x=v=>260+v/.03*530;
 for(const [i,[name,v,color]] of [['ONNX export',.0219638,plum],['Native CPU export',N.native.models.qat.max_probability_error,teal]].entries()){
 const y=95+i*108;text(r,0,y+7,name,{'font-size':23});r.append(el('rect',{x:260,y:y-12,width:x(v)-260,height:24,fill:color}));text(r,x(v)+12,y+7,v.toFixed(4),{'font-size':24,'font-weight':700,fill:color});}
 line(r,x(.02),50,x(.02),250,{stroke:plum,'stroke-dasharray':'6 5','stroke-width':2});text(r,x(.02),29,'0.02 tolerance',{'text-anchor':'middle',fill:plum,'font-size':21});
 for(const v of [0,.01,.02,.03])text(r,x(v),290,v.toFixed(2),{'text-anchor':'middle','font-size':19});text(r,520,330,'Absolute probability error · lower is better',{'text-anchor':'middle','font-size':20});
}
const nodes=[
 {id:'Whisper baseline',title:'Start with Whisper Tiny',value:1031.5,x:160,c:teal,model:'whisper',label:['Whisper Tiny','fine-tuned baseline'],individual:true,desc:'Whisper Tiny includes an audio encoder, attention pooling and an EoT classification head. Fine-tune the encoder and head on pause examples. Its 1,032 ms waiting time is the baseline; it is not a headless model.'},
 {id:'Whisper control',title:'Fine-tune Whisper-base and its head',value:896.83,x:435,c:teal,model:'whisper',label:['Whisper-base','hard labels'],desc:'Move to Whisper-base with a normalized audio frontend. Train on the existing end-of-turn labels; keep this recipe as the control for the distillation experiment.'},
 {id:'Cohere before adaptation',title:'Train a head on frozen Cohere',value:788.58,x:435,c:plum,model:'cohere',label:['Frozen encoder','+ trained EoT head'],desc:'Keep the pretrained Cohere encoder fixed and train only the end-of-turn classification head. This establishes the reference for testing whether adapting the encoder helps.'},
 {id:'Cohere fine-tuning',title:'Adapt Cohere to turn endings',value:636.17,x:710,c:plum,model:'cohere',label:['Fine-tune with LoRA'],desc:'Train low-rank adapters throughout Cohere’s encoder. Mean waiting falls from 789 to 636 ms across three runs. Freeze an adapted teacher before using its probabilities to train Whisper.'},
 {id:'Whisper distillation',title:'Learn Cohere’s probabilities',value:850,x:710,c:teal,model:'whisper',label:['Match Cohere’s','probabilities'],desc:'Give Whisper and the frozen Cohere teacher the exact same audio. Train Whisper to match the teacher’s probability of a turn ending. This gain also transfers to the separate Krisp evaluation.'},
 {id:'Pause augmentation',title:'Vary the audio inside pauses',value:801.83,x:985,c:teal,model:'whisper',label:['Vary silence','and pause noise'],desc:'Replace some pause tails with digital silence or low-level pink noise, without changing speech or labels. This tests reliance on recording artifacts. Added to distillation, it improves this benchmark; the extra Krisp benefit remains unresolved.'},
 {id:'Cohere continued training',title:'Fine-tune on more training data',value:D.deployment.gpu.delay5,x:1260,c:plum,model:'cohere',label:['More data + pause audio'],individual:true,desc:'Continue fine-tuning three Cohere models on the expanded data with pause augmentation. The best individual model reaches 615 ms and is selected for GPU serving. The three models jointly supervise the next Whisper student.'},
 {id:'Selected Whisper for CPU',title:'Add data and more pause examples',value:D.deployment.cpu.delay5,x:1260,c:teal,model:'whisper',label:['More data','and pause examples'],individual:true,desc:'Expand to 144,197 examples. Average three teachers’ logits on the same audio, adding cuts every 100 ms in eligible pauses. The combined training recipe reaches 756 ms; the contribution of each change was not isolated.'}
];
function renderHill(step=nodes.length-1,stop=()=>{},select=()=>{}){
 const r=$('#hill');r.replaceChildren();const y=v=>385-(v-550)*.65;
 text(r,75,19,'Mean waiting at 5% cutoffs (ms) · lower is better',{'font-size':21});
 for(const v of [600,800,1000]){line(r,75,y(v),1365,y(v));text(r,58,y(v)+6,round(v),{'text-anchor':'end','font-size':19,fill:'#514c4a'});}
 for(const [name,c,x] of [['Whisper',teal,1090],['Cohere',plum,1250]]){line(r,x,13,x+30,13,{stroke:c,'stroke-width':4});text(r,x+42,20,name,{'font-size':21,fill:c,'font-weight':700});}
 for(const family of ['whisper','cohere']){
  const series=nodes.map((n,i)=>({...n,i})).filter(n=>n.model===family);
  for(let j=1;j<series.length;j++){
   const a=series[j-1],b=series[j],active=b.i<=step;
   const p=el('path',{d:`M${a.x},${y(a.value)}L${b.x},${y(b.value)}`,fill:'none',stroke:b.c,'stroke-width':4,opacity:active?1:.18,class:b.i===step?'hill-segment':''});
   if(b.i===step)p.style.setProperty('--segment-length',Math.hypot(b.x-a.x,y(b.value)-y(a.value)));
   r.append(p);
  }
 }
 const transfer=el('g',{opacity:step>=4?1:.2,'aria-hidden':'true'});
 line(transfer,710,y(636.17)-18,710,y(850)+18,{stroke:plum,'stroke-width':1.5,'stroke-dasharray':'4 5'});
 transfer.append(el('path',{d:`M704,${y(850)+26}L710,${y(850)+18}L716,${y(850)+26}`,fill:'none',stroke:plum,'stroke-width':1.5}));
 text(transfer,730,265,'Distil',{'font-size':20,fill:plum});r.append(transfer);
 nodes.forEach((n,i)=>{
  const py=y(n.value),g=el('g',{class:'hill-milestone'+(i===step?' selected':''),role:'button',tabindex:0,'data-hill-step':i,'aria-label':`${n.title}: ${f(n.value)} ms, ${n.individual?'individual model':'mean of three runs'}. Show decision.`,'aria-pressed':String(i===step),opacity:i<=step?1:.32,style:`--milestone-color:${n.c}`});
  g.append(el('circle',{cx:n.x,cy:py,r:22,fill:'transparent',class:'hill-hit'}));
  g.append(el('circle',{cx:n.x,cy:py,r:15,fill:'none',stroke:n.c,'stroke-width':2,class:'hill-focus'}));
  g.append(el('circle',{cx:n.x,cy:py,r:7,fill:n.individual?n.c:paper,stroke:n.c,'stroke-width':3}));
  text(g,n.x,py-19,round(n.value),{'text-anchor':'middle','font-size':29,'font-weight':700,fill:n.c});
  if(i===7||i===6)text(g,n.x+43,py-20,i===7?'CPU':'GPU',{'font-size':18,fill:n.c});
  const labelY=n.model==='whisper'?414:py+(i===2?51:36);
  n.label.forEach((s,j)=>text(g,n.x,labelY+j*25,s,{'text-anchor':'middle','font-size':21,fill:n.model==='whisper'?ink:plum}));
  const choose=()=>{const focused=document.activeElement===g;stop();select(i);if(focused)r.querySelector(`[data-hill-step="${i}"]`).focus();};g.onclick=choose;g.onkeydown=e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();e.stopPropagation();choose();}};r.append(g);
 });
 const n=nodes[step];$('#hill-id').textContent=n.id;$('#hill-title').textContent=n.title;$('#hill-description').textContent=n.desc;
 $('#hill-detail').style.setProperty('--hill-accent',n.c);
}
function renderRanking(){
 const rows=D.published.map(r=>({name:r.name,v:r.delay5,cls:''}));
 rows.push({name:names.tiny,v:1031.5,cls:'original'},
 {name:names.teacher,v:D.deployment.gpu.delay5,cls:'ours'},
 {name:names.student,v:D.deployment.cpu.delay5,cls:'student-row'},
 {name:names.compressed,v:N.models.nativeQAT.delay5,cls:'student-row'},
 {name:names.delivered,v:809.25,cls:'original'});
 rows.sort((a,b)=>a.v-b.v);const root=$('#leaderboard');root.replaceChildren();
 rows.forEach((r,i)=>{const e=document.createElement('div');e.className='rank-row '+r.cls;e.style.transform=`translateY(${i*23}px)`;e.innerHTML='<span class="rank-label"></span><div class="rank-track"><div class="rank-bar"></div></div><span class="rank-value"></span>';e.querySelector('.rank-label').textContent=r.name;e.querySelector('.rank-bar').style.setProperty('--w',r.v/1600*100+'%');e.querySelector('.rank-value').textContent=round(r.v);root.append(e);});
 $('#ranking-note').textContent=`Selected deployments: Cohere GPU ${round(D.deployment.gpu.delay5)} ms; Whisper CPU ${round(D.deployment.cpu.delay5)} ms. Local readouts, not official entries.`;
}
function renderPareto(){
 const hw=$('#pareto-hardware').value,c=$('#pareto-concurrency').value,include=$('#pareto-teacher').checked,r=$('#pareto-chart');r.replaceChildren();$('#pareto-teacher').closest('label').hidden=hw!=='cpu';
 let points=[],log=hw==='cpu'&&include,note;
 if(hw==='cpu'){
 for(const [k,name] of [['B00','Whisper · hard-label control'],['B01','Whisper · earlier distillation']]){const s=L.serving[k],g=L.groups[k];points.push({name,x:s.levels[c].p95_ms,y:g.official_seed_delay5_ms[g.seeds.indexOf(s.selected_seed)],color:stone,open:true});}
 points.push({name:names.delivered,x:legacy('cpu')[c],y:809.25,color:teal,open:true},{name:'Whisper-base · matched FP32',x:native('fp',2)[c],y:774,color:ink},{name:names.compressed,x:native('qat',2)[c],y:774,color:teal});
 points.push({name:names.student,x:D.deployment.cpu.p95_ms[c],y:D.deployment.cpu.delay5,color:teal});
 if(include)points.push({name:'Cohere · earlier CPU FP32',x:cohere('cpu')[c],y:615,color:plum});
 note='The selected Whisper CPU point is the new ONNX Docker. Native INT8 and earlier models retain their original runtimes and thread counts. Open markers identify earlier runs.';
 }else if(hw==='gpu'){
 points=[{name:names.delivered,x:legacy('gpu')[c],y:809.25,color:teal,open:true},{name:'Cohere · earlier PyTorch',x:cohere('cuda')[c],y:615,color:stone,open:true},{name:names.teacher,x:D.deployment.gpu.p95_ms[c],y:D.deployment.gpu.delay5,color:plum}];
 note='RTX 3090 only. The current GPU delivery is Cohere with TensorRT. The earlier PyTorch point uses the same fine-tuned checkpoint; Whisper is the old ONNX reference. No CPU measurement is plotted as GPU.';
 }else{
 const names={cohere_ensemble:'Cohere · prediction ensemble',cohere_second_pass:'Cohere · extra texture training',cohere_single:'Cohere · individual teacher',cohere_soup:'Cohere · averaged weights',whisper_selected:'Whisper-base · distilled',whisper_weight_qat:'Whisper · QAT run in FP32'};
 points=N.cloud.filter(p=>p.precision==='fp32').map(p=>({name:names[p.name],x:p.http_p95_ms[c],y:p.mean_delay5_ms,color:p.name.startsWith('cohere')?plum:teal}));
 note='H200 cloud measurements, all in full precision. QAT denotes quantization-aware training; that row still executes in FP32. Local CPU and RTX 3090 results have their own views.';
 }
 const max=Math.max(...points.map(p=>p.x)),xmin=log?10:0,xmax=log?10000:Math.ceil(max/25)*25+25;
 const x=v=>85+(log?(Math.log10(v)-1)/3:v/xmax)*900,y=v=>345-(v-550)/450*290;
 for(const v of [550,650,750,850,950]){line(r,85,y(v),985,y(v));text(r,68,y(v)+6,v,{'text-anchor':'end','font-size':18});}
 const ticks=log?[10,30,100,300,1000,3000,10000]:Array.from({length:6},(_,i)=>i*xmax/5);
 for(const v of ticks){line(r,x(v),345,x(v),351);text(r,x(v),377,round(v),{'text-anchor':'middle','font-size':17});}
 text(r,85,24,'EoT waiting at 5% cutoffs (ms)',{'font-size':21});text(r,535,418,`Response p95 with ${c} request${c==='1'?'':'s'} at once (ms)${log?' · logarithmic scale':''}`,{'text-anchor':'middle','font-size':21});
 if(x(100)>=85&&x(100)<=985){line(r,x(100),45,x(100),345,{stroke:plum,'stroke-dasharray':'4 5'});text(r,x(100)+9,48,'100 ms HTTP',{'font-size':17,fill:plum});}
 const front=points.filter(p=>!points.some(q=>q!==p&&q.x<=p.x&&q.y<=p.y&&(q.x<p.x||q.y<p.y))).sort((a,b)=>a.x-b.x);
 if(front.length>1)r.append(el('path',{d:front.map((p,i)=>(i?'L':'M')+x(p.x)+','+y(p.y)).join(''),fill:'none',stroke:teal,'stroke-dasharray':'6 6','stroke-width':2,opacity:.65}));
 text(r,1030,24,'Model and serving configuration',{'font-size':19});
 points.forEach((p,i)=>{const g=el('g');const dot=el('circle',{cx:x(p.x),cy:y(p.y),r:10,fill:p.open?paper:p.color,stroke:p.color,'stroke-width':2});dot.append(el('title',{},p.name+': HTTP '+f(p.x)+' ms; waiting '+f(p.y)+' ms'));g.append(dot);text(g,x(p.x),y(p.y)+5,i+1,{'text-anchor':'middle','font-size':13,'font-weight':700,fill:p.open?p.color:paper});r.append(g);const ly=65+i*51;r.append(el('circle',{cx:1040,cy:ly-6,r:9,fill:p.open?paper:p.color,stroke:p.color,'stroke-width':2}));text(r,1040,ly-2,i+1,{'text-anchor':'middle','font-size':12,fill:p.open?p.color:paper});text(r,1060,ly,p.name,{'font-size':18,fill:p.color});text(r,1060,ly+21,f(p.x)+' ms HTTP · '+f(p.y)+' ms waiting',{'font-size':16,fill:stone});});
 $('#pareto-note').textContent=note;
}
function initialize(){document.querySelectorAll('[data-value]').forEach(e=>{const [kind,metric]=e.dataset.value.split('-');const m=D.deployment[kind];e.textContent=metric==='wait'?round(m.delay5):f(m.p95_ms['8']);});$('#pareto-hardware').onchange=renderPareto;renderCombinations();renderQAT();}
window.L041={initialize,renderServing,renderRanking,renderPareto,renderHill,nodes};
})();
