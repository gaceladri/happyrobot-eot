"""Build the spoken-only video script (docs/video-script.txt) from evidence/video-script.json and check it against the deck."""
from pathlib import Path
import json,re
R=Path(__file__).resolve().parents[1];D=R.parent
records=json.loads((R/'evidence/video-script.json').read_text())
sections=[(x['id'],x['action'],x['spoken']) for x in records]
html=(R/'index.html').read_text()
main=re.findall(r'<section\b(?=[^>]*data-section="main")(?=[^>]*id="([^"]+)")(?=[^>]*data-title="([^"]+)")[^>]*>',html)
assert [x[0] for x in main]==[x[0] for x in sections],([x[0] for x in main],[x[0] for x in sections])
count=lambda s:len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b",s))
words=sum(count(s[2]) for s in sections)
print('Spoken words',words,'minutes at 120 wpm plus 34 s',round(words/120+34/60,2),'at 115',round(words/115+34/60,2))
# One source for both spoken-only and slide-directed versions.
(R/'evidence/video-script.json').write_text(json.dumps([{'slide':i,'id':ident,'title':main[i-1][1],'action':cue,'spoken':text,'words':count(text)} for i,(ident,cue,text) in enumerate(sections,1)],indent=2)+'\n')
(D/'video-script.txt').write_text('\n\n\n\n'.join(x[2] for x in sections)+'\n')
