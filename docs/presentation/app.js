(() => {
'use strict';
const D=window.EOT_DATA,R=D.research,L=R.l040,$=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)],NS='http://www.w3.org/2000/svg';
const names=window.EOT_MODEL_NAMES;
const PLUM='#703f5b',TEAL='#38666a',INK='#353132',STONE='#8a817e',PAPER='#eeedeb';
const fmt=n=>Math.round(n+1e-8).toLocaleString('en-US'), one=n=>n.toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1});
const svg=(tag,a={},text)=>{const el=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;return el;};
function line(root,x1,y1,x2,y2,a={}){root.append(svg('line',{x1,y1,x2,y2,stroke:'#ccc8c5',...a}));}
function txt(root,x,y,t,a={}){root.append(svg('text',{x,y,fill:INK,'font-size':20,...a},t));}
const reduced=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;const slides=$$('.slide');let page=0,timer=null,hillStep=7;const mainCount=slides.filter(s=>s.dataset.section==='main').length;
function fit(){if(innerWidth<=760)return;const k=Math.min(innerWidth/1600,innerHeight/900);Object.assign($('#deck').style,{transform:`scale(${k})`,left:`${(innerWidth-1600*k)/2}px`,top:`${(innerHeight-900*k)/2}px`});}addEventListener('resize',()=>{fit();renderRanking();});fit();
function stop(){clearInterval(timer);timer=null;$('#play-hill').textContent='Play the progression';}
function replayClass(el){if(reduced())return;el.classList.remove('replaying');void el.offsetWidth;el.classList.add('replaying');}
function go(n){stop();page=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>{s.classList.toggle('active',i===page);s.inert=i!==page;});const s=slides[page];$('#replay-slide').disabled=!['cover','labels','adaptation','distillation','system','hill-slide','benchmark'].includes(s.id);$('#deck').dataset.tone=s.dataset.tone||'paper';const appendix=page>=mainCount;$('#appendix-button').textContent=appendix?'Main story':'Technical appendix';$('#slide-number').textContent=appendix?'A'+(page-mainCount+1)+' / '+(slides.length-mainCount):String(page+1).padStart(2,'0')+' / '+mainCount;$('#slide-title').textContent=s.dataset.title;$('#notes-content').textContent=s.dataset.note;$('#progress').style.width=(appendix?(page-mainCount+1)/(slides.length-mainCount):(page+1)/mainCount)*100+'%';$('#previous').disabled=page===0||page===mainCount;$('#next').disabled=page===mainCount-1||page===slides.length-1;history.replaceState(null,'','#'+(page+1));if(s.id!=='cover')$('#audio').pause();if(s.id==='labels')$('#supervision-example').dataset.phase=0;if(s.id==='benchmark')renderRanking();if(innerWidth<=760)scrollTo(0,0);}
addEventListener('hashchange',()=>go((Number(location.hash.slice(1))||1)-1));
$('#previous').onclick=()=>go(page-1);$('#next').onclick=()=>go(page+1);$('#appendix-button').onclick=()=>go(page>=mainCount?mainCount-1:mainCount);const navigate=d=>{const lo=page>=mainCount?mainCount:0,hi=page>=mainCount?slides.length-1:mainCount-1;go(Math.max(lo,Math.min(hi,page+d)));};$('#notes-button').onclick=()=>{const n=$('#notes');n.hidden=!n.hidden;$('#notes-button').setAttribute('aria-pressed',String(!n.hidden));};
async function full(){try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{}}
$('#fullscreen-button').onclick=full;$('#overview-button').onclick=()=>$('#contents').showModal();$('#glossary-button').onclick=()=>$('#glossary').showModal();$('#evidence-button').onclick=()=>$('#evidence').showModal();$$('[data-close]').forEach(b=>b.onclick=()=>b.closest('dialog').close());
slides.forEach((s,i)=>{const b=document.createElement('button');b.textContent=(i<mainCount?String(i+1).padStart(2,'0'):'A'+(i-mainCount+1))+'  '+s.dataset.title;b.onclick=()=>{$('#contents').close();go(i);};$('#contents-list').append(b);});
addEventListener('keydown',e=>{if($$('dialog[open]').length||/INPUT|SELECT|TEXTAREA/.test(e.target.tagName))return;if(/BUTTON|A/.test(e.target.tagName)&&[' ','Enter'].includes(e.key))return;if(['ArrowRight','ArrowDown','PageDown',' '].includes(e.key)){e.preventDefault();navigate(1);}if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();navigate(-1);}if(e.key==='Home')go(0);if(e.key==='End')go(page>=mainCount?slides.length-1:mainCount-1);if(e.key.toLowerCase()==='n')$('#notes-button').click();if(e.key.toLowerCase()==='g')$('#glossary-button').click();if(e.key.toLowerCase()==='o')$('#overview-button').click();if(e.key.toLowerCase()==='f')full();});
const audio=$('#audio');audio.src=D.demo.audioData;const wave=$('#waveform'), bars=[];let playhead;
function buildWave(){wave.replaceChildren();const max=Math.max(...D.demo.peaks), duration=D.demo.duration;D.demo.spans.forEach((s,i)=>{const last=i===D.demo.spans.length-1;const x=(s.start-D.demo.offset)/duration*1380,w=(s.end-s.start)/duration*1380;wave.append(svg('rect',{x,y:7,width:w,height:146,fill:last?'#ffffff1c':'#ffffff0a',rx:3}));wave.append(svg('text',{x:x+7,y:23,fill:last?'#fffafa':'#ccbbc6','font-size':15},last?'EOT · answer':'HOLD · keep listening'));});D.demo.peaks.forEach((p,i)=>{const h=Math.max(3,Math.sqrt(p/max)*85);const b=svg('rect',{x:i*1380/260,y:83-h/2,width:3.2,height:h,rx:1.6,class:'wave-bar'});wave.append(b);bars.push(b);});playhead=svg('line',{x1:0,x2:0,y1:28,y2:144,class:'wave-playhead'});wave.append(playhead);}
buildWave();
function audioTick(){const t=audio.currentTime, absolute=t+D.demo.offset;$('#audio-time').textContent=`${t.toFixed(1)} / ${D.demo.duration.toFixed(1)} s`;const x=t/D.demo.duration*1380;playhead.setAttribute('x1',x);playhead.setAttribute('x2',x);bars.forEach((b,i)=>b.setAttribute('class',i/260<t/D.demo.duration?'wave-progress':'wave-bar'));const words=D.demo.words.filter(w=>w.start<=absolute);$('#spoken-words').textContent=words.length?words.slice(-11).map(w=>w.word).join(' '):'No, but I would…';const idx=D.demo.spans.findIndex(s=>absolute>=s.start&&absolute<=s.end);$('#audio-state').textContent=idx<0?'Caller is speaking':idx===D.demo.spans.length-1?'Turn ends · EOT · answer':'Mid-sentence pause · HOLD · keep listening';}
let audioFrame;function animateAudio(){audioTick();if(!audio.paused)audioFrame=requestAnimationFrame(animateAudio);}audio.addEventListener('play',()=>{cancelAnimationFrame(audioFrame);animateAudio();});audio.addEventListener('pause',()=>cancelAnimationFrame(audioFrame));audio.addEventListener('timeupdate',audioTick);audio.addEventListener('ended',()=>{$('#play-audio').innerHTML='Replay the caller <span>↻</span>';});audio.addEventListener('pause',()=>{if(!audio.ended)$('#play-audio').innerHTML='Play the caller <span>▶</span>';});
async function playAudio(){if(audio.ended)audio.currentTime=0;try{await audio.play();$('#play-audio').innerHTML='Pause <span>Ⅱ</span>';}catch{$('#audio-state').textContent='Playback unavailable. The transcript and annotations remain visible.';}}
$('#play-audio').onclick=()=>audio.paused?playAudio():audio.pause();


// Each animation explains a user-selected transition; all findings are visible initially.
function mine(){stop();$('#supervision-example').dataset.phase=0;const labels=['1. Stop the input at the pause.','2. Observe later behavior in the recording.','3. Assign the label to the earlier audio prefix.'];if(reduced()){$('#labeling-step').textContent=labels[2];return;}let phase=0;function step(){phase++;$('#supervision-example').dataset.phase=phase;$('#labeling-step').textContent=labels[phase-1];if(phase===3)stop();}step();timer=setInterval(step,1500);}
$('#mine-button').onclick=mine;
$$('[data-replay]').forEach(b=>b.onclick=()=>replayClass($(b.dataset.replay==='adaptation'?'#adaptation-bars':'#distill-diagram')));
function replay(){const id=slides[page].id;if(id==='cover'){audio.currentTime=0;playAudio();}else if(id==='labels')mine();else if(id==='adaptation')replayClass($('#adaptation-bars'));else if(id==='distillation')replayClass($('#distill-diagram'));else if(id==='system')replayClass($('#pipeline'));else if(id==='hill-slide')playHill();else if(id==='benchmark')replayClass($('#leaderboard'));else if(id==='factorial'){factorial(domain==='official'?'krisp':'official');}else if(id==='pareto'){$('#pareto-teacher').checked=!$('#pareto-teacher').checked;renderPareto();}}
$('#replay-slide').onclick=replay;
// Metric: actual policy sweeps, no fabricated interpolation.
const publicNames={'LiveKit Turn Detector v1':'LiveKit v1','LiveKit Turn Detector v1-mini':'LiveKit v1-mini','OpenAI GPT Realtime 2':'OpenAI GPT Realtime 2'};
const refSeries=[...D.published.map((r,i)=>({id:'public-'+i,name:publicNames[r.name]||r.name,fullName:r.name,c:r.name==='VAD baseline'?'#7b7470':'#aaa29d',d:r.latencyCurve,public:true,dash:r.name==='VAD baseline'?'8 5':''})),
{id:'tiny',name:names.tiny,fullName:'Our Whisper Tiny · historical baseline',c:INK,d:R.tinyLatencyCurve,dash:'8 4'},
{id:'teacher',name:names.teacher,fullName:'Selected fine-tuned Cohere, TensorRT GPU; quality rechecked after conversion',c:PLUM,d:D.deployment.gpu.latencyCurve},
{id:'kd',name:names.student,fullName:'Selected distilled Whisper-base, optimized ONNX CPU; 756 ms waiting at 5% cutoffs',c:TEAL,d:D.deployment.cpu.latencyCurve},
{id:'combined',name:names.compressed,fullName:'Experimental 8-bit Whisper-base on native CPU; 774 ms waiting at 5% cutoffs',c:'#225e64',d:D.l041.models.nativeQAT.latencyCurve},
{id:'legacy',name:names.delivered,fullName:'Earlier Whisper-base: distillation plus pause texture; 809 ms waiting at 5% cutoffs',c:TEAL,d:R.students.B11.seeds.find(s=>s.seed===333).latencyCurve,dash:'9 4'}];
let lockedSeries=null;const included=()=>true;
const frontierMax=Math.ceil(Math.max(...refSeries.flatMap(s=>s.d.map(p=>p[1]??0)))/20)*20;
const frontierX=v=>72+v/2000*890,frontierY=v=>378-v/frontierMax*322;
function curveValue(s,b){return s.d.find(v=>Math.abs(v[0]-b)<1e-6)?.[1]??null;}
function highlightSeries(id){
 const selected=refSeries.find(s=>s.id===id),budget=Number($('#budget').value);
 refSeries.forEach(s=>{const p=$('#curve-'+s.id),dot=$('#dot-'+s.id),active=!id||s.id===id;p.style.display=included(s)?'':'none';if(!included(s))dot.style.display='none';p.style.opacity=id?(active?'1':'.12'):(s.public?'.6':'1');p.setAttribute('stroke-width',id?(active?4:1.5):(s.public?1.65:3.2));dot.style.opacity=active?'1':'.12';if(id&&active){p.parentNode.append(p);p.parentNode.append(dot);}const row=$('#readout-'+s.id);if(row)row.classList.toggle('highlighted',s.id===id);});
 const value=selected?curveValue(selected,budget):curveValue(refSeries.find(s=>s.id==='kd'),budget);
 $('#frontier-caption').textContent=selected?selected.fullName+' · '+(value===null?'no measured policy meets '+budget+' ms':one(value)+'% false cutoffs at '+budget+' ms'):'Selected Whisper-base CPU: '+(value===null?'no measured policy meets this budget':one(value)+'% false cutoffs at '+budget+' ms')+'. Select a model or drag the chart.';
}
function renderFrontier(){
 const root=$('#frontier');root.replaceChildren();
 for(let v=0;v<=frontierMax;v+=20){line(root,72,frontierY(v),962,frontierY(v));txt(root,58,frontierY(v)+6,v+'%',{'text-anchor':'end',class:'chart-tick'});}
 for(const v of [0,500,1000,1500,2000])txt(root,frontierX(v),407,fmt(v),{'text-anchor':'middle',class:'chart-tick'});
 txt(root,72,25,'False-cutoff rate (%) · lower is better',{class:'chart-label'});
 txt(root,515,446,'Mean waiting after the caller finishes (ms)',{'text-anchor':'middle',class:'chart-label'});
 root.append(svg('line',{id:'budget-guide',y1:56,y2:378,stroke:PLUM,'stroke-dasharray':'4 5'}));
 for(const s of refSeries){let path='',prev=false;for(const v of s.d){if(v[1]===null){prev=false;continue;}path+=prev?`H${frontierX(v[0])}V${frontierY(v[1])}`:`M${frontierX(v[0])},${frontierY(v[1])}`;prev=true;}const p=svg('path',{id:'curve-'+s.id,d:path,fill:'none',stroke:s.c,'stroke-width':s.public?1.65:3.2,'stroke-dasharray':s.dash||''});p.append(svg('title',{},s.fullName));root.append(p);root.append(svg('circle',{id:'dot-'+s.id,r:s.public?3.3:5,fill:s.c,stroke:PAPER,'stroke-width':1}));}
 updateBudget();
}
function updateBudget(){
 const b=Number($('#budget').value);$('#budget-value').textContent=fmt(b);$('#budget').setAttribute('aria-valuetext',b+' milliseconds mean waiting');
 $$('[data-budget]').forEach(el=>el.setAttribute('aria-pressed',String(Number(el.dataset.budget)===b)));
 $('#budget-guide').setAttribute('x1',frontierX(b));$('#budget-guide').setAttribute('x2',frontierX(b));
 const list=$('#frontier-readouts'),sorted=[...refSeries].sort((a,c)=>(curveValue(a,b)??Infinity)-(curveValue(c,b)??Infinity));
 for(const s of sorted){const v=curveValue(s,b),dot=$('#dot-'+s.id);dot.style.display=v===null?'none':'';if(v!==null){dot.setAttribute('cx',frontierX(b));dot.setAttribute('cy',frontierY(v));}
  let row=$('#readout-'+s.id);if(!row){row=document.createElement('button');row.id='readout-'+s.id;row.className='frontier-row'+(s.public?'':' local-solution')+(['kd','teacher'].includes(s.id)?' selected-solution':'');row.style.setProperty('--series',s.c);row.innerHTML='<i></i><span></span><strong></strong>';row.querySelector('span').textContent=s.name;row.onmouseenter=()=>highlightSeries(s.id);row.onmouseleave=()=>highlightSeries(lockedSeries);row.onfocus=()=>highlightSeries(s.id);row.onblur=()=>highlightSeries(lockedSeries);row.onclick=()=>{lockedSeries=lockedSeries===s.id?null:s.id;$$('.frontier-row').forEach(x=>x.setAttribute('aria-pressed',String(x.id==='readout-'+lockedSeries)));highlightSeries(lockedSeries);};}
  row.hidden=!included(s);row.querySelector('strong').textContent=v===null?'n/a':one(v)+'%';row.classList.toggle('unattainable',v===null);row.title=v===null?'No measured model policy meets this latency budget':s.fullName;
  row.setAttribute('aria-label',s.fullName+', '+(v===null?'no measured policy meets '+b+' milliseconds':one(v)+' percent false cutoffs at '+b+' milliseconds waiting')+', highlight curve');row.setAttribute('aria-pressed',String(lockedSeries===s.id));list.append(row);
 }
 $('#metric').dataset.comparison='all';highlightSeries(lockedSeries);
}
$('#budget').oninput=updateBudget;$$('[data-budget]').forEach(b=>b.onclick=()=>{$('#budget').value=b.dataset.budget;updateBudget();});
// Pointer movement controls the same measured 10 ms budget grid as the keyboard-accessible slider.
let draggingBudget=false;
function moveBudget(e){const root=$('#frontier'),p=new DOMPoint(e.clientX,e.clientY).matrixTransform(root.getScreenCTM().inverse());$('#budget').value=Math.max(200,Math.min(2000,Math.round((p.x-72)/890*2000/10)*10));updateBudget();}
$('#frontier').addEventListener('pointerdown',e=>{if(e.button!==0)return;draggingBudget=true;e.currentTarget.setPointerCapture(e.pointerId);moveBudget(e);});
$('#frontier').addEventListener('pointermove',e=>{if(draggingBudget)moveBudget(e);});
$('#frontier').addEventListener('pointerup',()=>{draggingBudget=false;});$('#frontier').addEventListener('pointercancel',()=>{draggingBudget=false;});
// Calibration: coverage has a different denominator from precision.
function renderRecall(){const r=$('#recall-chart');r.replaceChildren();txt(r,0,22,'Coverage-adjusted EOT recall (%)',{class:'chart-label'});for(const [i,v] of [12.4,20.4].entries()){const y=70+i*68;txt(r,0,y+6,i?'Revised rules':'Original rules',{'font-size':19});r.append(svg('rect',{x:160,y:y-15,width:v/25*310,height:25,fill:i?PLUM:STONE}));txt(r,160+v/25*310+12,y+6,v.toFixed(1),{'font-size':24,'font-weight':700});}txt(r,160,180,'0',{'font-size':15});txt(r,470,180,'25%',{'font-size':15,'text-anchor':'end'});txt(r,0,212,'Revised EOT precision: 97.4%',{'font-size':24,'font-weight':700,fill:PLUM});}
// Re-scoring the same predictions separates evaluation semantics from dataset shift.
function renderEvaluation(){const r=$('#evaluation-chart');if(!r)return;r.replaceChildren();const g=R.evaluationGap, a=g.gap_same_rule_score_point,b=g.gap_headline-a,xx=v=>v/500*690;txt(r,0,30,'Additional waiting on Krisp (ms)',{class:'chart-label'});for(const [i,v]of [g.gap_headline,a].entries()){const y=98+i*108;txt(r,0,y-27,i?'Harness rule on both datasets':'Original comparison · mixed rules',{'font-size':20});r.append(svg('rect',{x:0,y,width:xx(a),height:25,fill:STONE}));if(!i)r.append(svg('rect',{x:xx(a),y,width:xx(b),height:25,fill:PLUM}));txt(r,xx(v)+12,y+21,one(v),{'font-size':26,'font-weight':700});}txt(r,0,275,'0',{'font-size':16});txt(r,690,275,'500 ms',{'font-size':16,'text-anchor':'end'});txt(r,0,322,one(b)+' ms of the gap came from changing the rule.',{'font-size':23,fill:PLUM,'font-weight':700});}
// Teacher adaptation: same experiment, same budget; seed ranges are shown.
function renderAdaptation(){const root=$('#adaptation-bars');root.replaceChildren();['L031 C0','L031 C1','L031 C2'].forEach((id,i)=>{const s=D.stages.find(x=>x.id===id),lo=Math.min(...s.values),hi=Math.max(...s.values);const row=document.createElement('div');row.className='adaptation-row';const names=['Frozen encoder','Fine-tune top third','LoRA on all layers'];const subs=['0.7M trainable · head only','630.7M trainable · upper layers','28.2M trainable · low-rank adapters'];row.innerHTML=`<div><h3>${names[i]}</h3><p>${subs[i]}</p><div class="encoder-strip" aria-hidden="true">${Array.from({length:12},(_,j)=>`<i class="${i===1&&j>=8?'tuned':i===2?'lora':''}"></i>`).join('')}</div></div><div class="adapt-bar"><strong>${fmt(s.value)} <small>ms</small></strong><i style="--w:${s.value/1000*100}%"></i><span class="adapt-whisker" style="left:${lo/1000*100}%;width:${(hi-lo)/1000*100}%"></span></div>`;root.append(row);});}
// Factorial: derive every value and contrast from the frozen closure.
let domain='official';function factorial(d){domain=d;const metric=d==='official'?'official_delay5_ms':'krisp_score_point_delay5_ms';$$('[data-domain]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.domain===d)));$$('.factor-cell').forEach(el=>el.querySelector('strong').textContent=one(L.groups[el.dataset.arm].metrics[metric].mean));const a=L.contrasts['B01-B00'][metric],b=L.contrasts['B11-B01'][metric];function effect(v){return `${v.mean<0?'−':'+'}${one(Math.abs(v.mean))} ms`;}function ci(v){return v.ci95.map(x=>(x<0?'−':'+')+one(Math.abs(x))).join(' to ');}$('#factorial-reading').innerHTML=`<h3>What changed against its control</h3><p>Distillation vs hard labels</p><strong>${effect(a)}</strong><small>Paired 95% interval: ${ci(a)} ms</small><p>Add texture to distillation</p><strong>${effect(b)}</strong><small>Paired 95% interval: ${ci(b)} ms</small><p class="annotation">${d==='official'?'Both contrasts favor lower waiting here. Switch to Krisp to test whether the added benefit transfers.':'Distillation transfers. The extra texture benefit is unresolved: its interval crosses zero.'}</p>`;}
$$('[data-domain]').forEach(b=>b.onclick=()=>factorial(b.dataset.domain));
// Research hillclimbing: a shared quality axis; horizontal positions group decisions.
const researchNodes=window.L041.nodes;
function renderHill(step=researchNodes.length-1){hillStep=step;window.L041.renderHill(step,()=>stop(),renderHill);}
function playHill(){stop();if(reduced()){renderHill();return;}renderHill(0);$('#play-hill').textContent='Playing…';timer=setInterval(()=>{if(hillStep===researchNodes.length-1){stop();return;}renderHill(hillStep+1);},1500);}
$('#play-hill').onclick=playHill;
// A single public overlay, with means or the actual serving checkpoints.
function renderRanking(){window.L041.renderRanking();}
$('#replay-ranking').onclick=()=>replayClass($('#leaderboard'));
// Paired CPU/CUDA measurement of one selected export; historical Pareto data stay separate.
function renderServing(){window.L041.renderServing();}
function renderPareto(){window.L041.renderPareto();}
$('#pareto-concurrency').onchange=renderPareto;$('#pareto-teacher').onchange=renderPareto;

function renderTransfer(){
 const r=$('#transfer-evidence');r.replaceChildren();const m=L.contrasts['B01-B00'];
 for(const [i,[label,k]] of [['EoT Bench','official_delay5_ms'],['Krisp','krisp_score_point_delay5_ms']].entries()){
  const gain=-m[k].mean,y=90+i*145;
  txt(r,0,y-33,label,{'font-size':26});
  r.append(svg('rect',{x:0,y,width:gain/120*440,height:15,fill:TEAL}));
  txt(r,495,y+13,fmt(gain)+' ms',{'font-size':43,fill:TEAL,'font-weight':700});
  txt(r,495,y+45,'less waiting',{'font-size':21,fill:INK});
 }
}

window.L041.initialize();renderFrontier();renderRecall();renderEvaluation();renderAdaptation();factorial('official');renderHill(7);renderRanking();renderServing();renderPareto();renderTransfer();
window.EOT_PRESENTATION={go,mainCount,slides:slides.length,data:D,renderHill,renderFrontier,renderRanking,factorial,renderPareto,insert:renderRanking};
go((Number(location.hash.slice(1))||1)-1);window.EOT_READY=true;
})();
