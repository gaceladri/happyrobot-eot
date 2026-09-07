"""Build the four-page hiring brief and its Markdown counterpart from the same text."""
from pathlib import Path
import re, json
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
R=Path(__file__).resolve().parents[1]
D=json.loads((R/'evidence/deployment.json').read_text())
CPU,GPU=D['cpu'],D['gpu']
fmt=lambda v: f'{v:.1f}'
ms=lambda d: str(round(d['delay5']))
for n,f in [('U','DisplaySans-Regular.ttf'),('UB','DisplaySans-Bold.ttf'),('Serif','DisplaySerif-Regular.ttf')]:
    pdfmetrics.registerFont(TTFont(n,str(R/'assets'/f)))
pdfmetrics.registerFontFamily('U',normal='U',bold='UB',italic='U',boldItalic='UB')
INK=colors.HexColor('#222222');TEAL=colors.HexColor('#225e64');PLUM=colors.HexColor('#703f5b');MUTED=colors.HexColor('#514c4a');LINE=colors.HexColor('#d0ccc9');PAPER=colors.HexColor('#eeedeb')
OUT=R.parent/'solution-brief.pdf'
c=canvas.Canvas(str(OUT),pagesize=(595.276,841.89));c.setTitle('Knowing when to answer - short solution document');c.setAuthor('HappyRobot engineering exercise');page=0;y=0
md=['# Knowing when to answer\n\nShort solution document - 7 September 2026.\n\nThis Markdown and the four-page PDF are built from the same source. Figures are observed results, not production qualification.\n']
def clean(s):return s.replace('−','-').replace('–','-').replace('—','-')
def paragraph(s,size=11.2,gap=8,color=INK,record=True):
    global y
    s=clean(s)
    ob=Paragraph(s,ParagraphStyle('p',fontName='Serif' if size>=27 else 'U',fontSize=size,leading=size*1.32,textColor=color))
    _,h=ob.wrap(499,1000)
    assert y-h>64,(page,s[:70],y-h)
    ob.drawOn(c,48,y-h);y-=h+gap
    if record:md.append(re.sub(r'</?b>','**',s).replace('<br>','\n')+'\n')
def h(s):
    paragraph(s,14.5,6,record=False);md.append('### '+s+'\n')
def start(label,title,sub):
    global page,y
    if page:c.showPage()
    page+=1;c.setFillColor(PAPER);c.rect(0,0,595.276,841.89,fill=1,stroke=0)
    c.setFillColor(MUTED);c.setFont('U',9);c.drawString(48,807,'HappyRobot / AI-ML engineering exercise');c.drawRightString(547,807,'7 September 2026');c.setStrokeColor(LINE);c.line(48,792,547,792)
    c.setFont('U',8.3);c.drawString(48,30,'Selected solution, controlled evidence and release boundaries.');c.drawRightString(547,30,f'{page} / 4');y=766
    paragraph(label.upper(),9,10,PLUM,False);paragraph(title,29,12,record=False);paragraph(sub,12,18,MUTED,False)
    md.append('## '+title+'\n\n'+sub+'\n')
def source(s):
    global y
    y=min(y,100);paragraph('Evidence: '+s,8.4,0,MUTED,False);md.append('*Evidence: '+s+'*\n')
def table(headers,rows,widths):
    global y
    cols=[48]
    for w in widths:cols.append(cols[-1]+w)
    for idx,row in enumerate([headers]+rows):
        c.setStrokeColor(LINE);c.line(48,y,547,y)
        for j,val in enumerate(row):
            c.setFillColor(INK if j==0 else TEAL);c.setFont('UB' if idx==0 else 'U',10.2)
            if j==0:c.drawString(cols[j]+2,y-18,str(val))
            else:c.drawRightString(cols[j+1]-3,y-18,str(val))
        y-=27
    y-=12
    md.extend([' | '.join(headers),' | '.join(['---']*len(headers))]+[' | '.join(map(str,row)) for row in rows]+[''])

start('1 / Recommendation','A stronger teacher, a practical student','Train on pauses, adapt Cohere to turn completion, then distil its decisions into a small Whisper encoder.')
paragraph(f'<b>Two inference deployments:</b> distilled Whisper-base on CPU with optimized ONNX, and fine-tuned Cohere on RTX 3090 with TensorRT. At a 5% false-cutoff budget, the deployed models wait <b>{ms(CPU)} ms</b> and <b>{ms(GPU)} ms</b>, respectively. Both are re-evaluated after export.',12.4)
h('Improve the decision people actually experience')
paragraph(f'The historical Whisper Tiny baseline waits 1,032 ms. The selected student reduces this by {round((1-CPU["delay5"]/1031.5)*100)}%. At a fixed 300 ms waiting budget, its false cutoffs fall from 30.5% to {fmt(dict(CPU["latencyCurve"])[300])}%. A false cutoff is measured per eligible mid-sentence pause, not per call.')
h('Put quality and compute in the same decision')
paragraph(f'Whisper provides a compact CPU path at <b>{fmt(CPU["p95_ms"]["8"])} ms HTTP p95</b> with eight requests in flight. Cohere delivers lower waiting on GPU at <b>{fmt(GPU["p95_ms"]["8"])} ms HTTP p95</b> under the same load. Its quality curve and response times refer to the same TensorRT artifact. Each service has its own Docker image.')
h('Keep the two clocks separate')
paragraph('<b>Benchmark waiting</b> excludes inference. <b>HTTP response p95</b> includes audio processing, model execution and local queuing. Adding a service p95 to mean waiting does not give a measured end-to-end response time. LLM generation and first TTS audio still need integration measurement.')
h('Compare under the same evaluation rule')
paragraph(f'The public English overlay uses the same spans, labels, 200 ms score point, HOLD filtering and policy grid as the pinned LiveKit harness. LiveKit v1 reports 543 ms; the deployed Cohere is {round(GPU["delay5"]-543)} ms slower. These are local descriptive comparisons, not official leaderboard entries or a blind test. Repeated benchmark inspection limits generalization claims.')
source('Deployment quality, HTTP and artifact receipts; expanded-data report. Public harness 6594d8b3, dataset ca9d98a9; 400 English turns. Full paths, hashes and policy sweeps are included in the companion evidence.')

start('2 / Supervision','Calibrate automatic labels with existing annotations','We did not manually label a new corpus. We reused public annotations to diagnose and tune an automatic labeling pipeline.')
h('1. Construct the training target at real pauses')
paragraph('A caller who resumes speaking supplies a HOLD candidate. A substantive agent reply followed by caller silence supplies an EOT candidate. Rules handle timing, word counts, overlap and backchannels. Future behavior is used offline to label a prefix; the model receives only audio available before that cut. Agent behavior is a proxy for intent.')
h('2. Derive a calibration reference from TurnBench')
paragraph('TurnBench dev already contains 38 conversations and three annotator tracks per speaker. We map their event annotations into HOLD/EOT references and combine the tracks. The binary reference is therefore derived by our mapper, not an independently supplied gold label for every pause. Strict and overlap-tolerant mappings answer different questions.')
h('3. Tune transferable heuristics, then test candidate recovery')
paragraph('Revised rules increase reference-EOT coverage from <b>12.4% to 20.4%</b> with <b>97.4% precision</b> against the strict derived reference. Rules use timing and word counts, so they transfer to AppTek without TurnBench event labels. Of 275 recovered reference EOTs, only <b>81</b> are eligible at the 200 ms decision point; most others occur after the agent starts.')
paragraph('The headline result uses the calibration conversations. A two-fold check found high global precision, but one marginal rule was unstable; it is not an untouched label-quality test. In AppTek, combined rules recover 1,202 eligible candidates, including 826 training candidates. Subsequent curation experiments did not establish a model-quality gain.')
h('4. Keep pilots and trained data distinct')
paragraph('The otoSpeech pilot produced 2,956 causal examples from 12 calls. Its review UI contained <b>0 completed reviews out of 200</b>; no otoSpeech-trained model or downstream gain is claimed. The selected expanded-data recipe adds AppTek and SmartTurn cuts. It does not inherit every labeling correction simply because that correction was studied.')
source('Historical audit: original labeling commits plus calibration, curation and final training contracts (L001/L008/L009/L020/L040/L041; D019). Pre-existing human annotations do not imply a completed listening audit by us.')

start('3 / Experimental judgment','Make each improvement earn its place','Use controlled comparisons to establish transfer; keep later bundled improvements and unresolved effects explicit.')
h('Match the task, then adapt the representation')
paragraph('Pause-prefix rather than whole-clip training reduces cutoffs at 300 ms waiting from <b>55.6% to 31.9%</b>. With fixed data and budget, frozen Cohere, partial fine-tuning and all-layer low-rank adapters give <b>789 / 710 / 636 ms</b> mean waiting over three runs. Adapter fine-tuning saves 152 ms [paired 95% interval: -191, -113].')
h('Separate distillation from pause augmentation')
table(['Training condition','EoT Bench','Krisp'],[
['Hard-label control','896.8 ms','1,199.9 ms'],['Texture only','900.0 ms','1,202.1 ms'],['Distillation','850.0 ms','1,091.9 ms'],['Distillation + texture','801.8 ms','1,089.4 ms']], [269,115,115])
paragraph('Three runs per condition; means, not ensembles. Distillation improves both evaluations: <b>-46.8 ms</b> on EoT Bench and <b>-108.0 ms</b> on Krisp, with paired intervals below zero. Texture adds another public gain; its extra Krisp effect is unresolved. Teacher and student receive identical realized causal audio; only Whisper runs at inference.')
h('Distinguish research progress from component attribution')
paragraph('Expanded-data research uses 144,197 training examples, a fixed 16,390-row validation set and 353,930 teacher targets. Its selected student reaches 756 ms. Data, teacher continuation and denser targets were changed together, so their separate gains are not identified. This recipe has no new Krisp readout.')
h('Treat failed ideas as decisions, not hidden successes')
paragraph('Curation failed replicated controls; longer training did not repair transfer. Teacher averaging slightly improved AUC yet worsened waiting. An evaluation audit attributed 157.6 ms of an apparent dataset gap to a different firing rule. These findings prevent false attribution and repeated mistakes.')
source('Matched training study; adaptation, scoring audit, controlled distillation and expanded-data studies. Paired intervals condition on policies selected on the inspected benchmarks. The 801.8 ms recipe is the historical control; the selected expanded-data student reaches 756.25 ms in its CPU Docker.')

start('4 / Operation and next step','Deliver the service; qualify the voice interaction','The model API is implemented. The live turn-taking policy, feedback loop and production capacity still need qualification.')
h('Measure the actual inference containers')
table(['Full HTTP p95','1 request','4 requests','8 requests'],[
['Whisper / CPU ONNX',*[fmt(CPU['p95_ms'][str(c)])+' ms' for c in [1,4,8]]],
['Cohere / GPU TensorRT',*[fmt(GPU['p95_ms'][str(c)])+' ms' for c in [1,4,8]]]], [214,95,95,95])
paragraph(f'Intel i9-10900X ({CPU["threads"]} ONNX threads); RTX 3090 (TensorRT {GPU["precision"]}, {GPU["threads"]} frontend threads). One fixed 8 s PCM16 payload, 400 warm requests per load level and service; 2,400 total with zero errors. No CPU quota. These closed-loop sweeps do not establish concurrent-call capacity.')
h('Separate the scorer from the response policy')
paragraph('VAD identifies a pause. The API accepts up to 8 s of audio, uses 4 s internal context, and returns a score, flag and timings. It implements warm-up, bounded admission, deadlines and model identity. The agent must still manage commitment, resumed speech and LLM/TTS cancellation.')
h('Qualify the runtime, then recheck the policy')
paragraph('The CPU export uses fused attention and normalization. TensorRT uses merged adapters, shape-specific profiles, stable CUDA graphs and protected FP32 reductions. Numerical probes precede a full public quality replay. Batch-dependent scores are checked under a shared policy. INT8 conversion diagnostics remain separate from the selected artifact; a historical 612 ms result is not a TensorRT result.')
h('Next investment: representative labels and live behavior')
paragraph('Log decisions, model/policy versions, latency, timeouts and resumed speech. Track interruptions, waiting, corrections, number capture and task completion. Sample uncertain and consequential cases for explicit review; no complaint is weak evidence. Minimize retained audio.')
paragraph('Build an independently reviewed call set by accent, channel and overlap. Freeze selection rules; test, shadow, canary and retain rollback. Limits include benchmark reuse, unresolved short-audio similarities and unknown pretraining overlap. Historical Tiny remains the formal research incumbent; the two deployments are selected for this exercise, not promoted to production.')
source('Deployment quality and HTTP receipts; CPU and GPU service source; reproduction manifest. Checkpoints restored from the archive, exports and engine hashed. Model weights are distributed separately from the document bundle.')
c.save()
(R.parent/'solution-brief.md').write_text('\n'.join(md).rstrip()+'\n')
print(OUT)
print('Pages:',page,'Words:',len(' '.join(md).split()))
