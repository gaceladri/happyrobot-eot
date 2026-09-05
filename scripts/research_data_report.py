"""Export aggregate data/ablation plots without uploading any private corpus content."""
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research import ROOT,STATE,atomic

p=argparse.ArgumentParser();p.add_argument('--wandb',action='store_true');a=p.parse_args()
fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3);width=.36
def auc(i):return json.loads((STATE/'experiments'/f'{i}.json').read_text())['metrics']['dev_auc']
ax.bar(x-width/2,[auc(i) for i in ['E013','E017','E020']],width,label='Ambiguity-filtered data')
ax.bar(x+width/2,[auc(i) for i in ['E014','E018','E021']],width,label='Matched random exclusions')
ax.axhline(.8723890591,color='gray',linestyle='--',label='Original incumbent')
ax.set(xticks=x,xticklabels=['Seed17','Seed42','Fixed weight average'],ylim=(.865,.885),ylabel='Development AUC',title='Data-filter ablation: benefit varies with training seed')
ax.legend(fontsize=8);fig.tight_layout();fig.savefig(STATE/'plots/data_ablation.png',dpi=150);plt.close(fig)
inventory=json.loads((ROOT/'data/research/D016/inventory.json').read_text());pilot=json.loads((ROOT/'data/research/D019/mined/status.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4))
counts=inventory['conversations'];labels=['train','dev','confirmation','quarantine_cross_split']
axes[0].bar(['Train','Dev','Confirmation','Cross-split\nexcluded'],[counts[k] for k in labels]);axes[0].set(ylabel='Conversations',title='Frozen actor-disjoint partition')
axes[1].bar(['Proposed HOLD','Proposed EOT'],[pilot['counts']['label:0'],pilot['counts']['label:1']]);axes[1].set(ylabel='Causal training candidates',title='12-conversation pilot; human review pending')
fig.tight_layout();fig.savefig(STATE/'plots/otospeech_pilot.png',dpi=150);plt.close(fig)
if a.wandb:
 import wandb
 run=wandb.init(project='happyrobot-eot',entity='adrianbrunetto1-reshape',id='ar20260905report',resume='allow',group='autoresearch-20260905',
  name='Autoresearch live scorecard',settings=wandb.Settings(disable_code=True,init_timeout=30),dir=str(ROOT/'wandb'))
 run.log({k:wandb.Image(str(STATE/'plots'/f'{k}.png')) for k in ['data_ablation','otospeech_pilot']});run.finish()
print('Aggregate data plots exported')
