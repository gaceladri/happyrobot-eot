# Interactive presentation

Open `eot-presentation.html` locally: a self-contained deck with 12 main slides and a 10-slide technical appendix. Arrow keys navigate; speaker notes and the glossary carry the technical detail. The spoken script of the video is `../video-script.txt`; the short solution document is `../solution-brief.md` (`../solution-brief.pdf`).

`evidence/` holds the receipts the deck reads (`deployment.json`, `data.json`, `sources.json`, ...). `deployment.json` and `whisper-http.json` are snapshots of the same files under `../../evidence/deployment/`; `cohere-http.json` is a snapshot of `cohere-tuned-http.json` there. Receipts keep the absolute paths of the machine they were measured on as provenance; they are not commands to run elsewhere.

## Rebuild

Edit `index.html`, `style.css`, `data.js`, `l041.js` and `app.js`; never edit the generated `eot-presentation.html` directly. From this directory:

```bash
python3 scripts/build_video.py      # regenerates ../video-script.txt from evidence/video-script.json
python3 scripts/build_pdf.py        # regenerates ../solution-brief.pdf and ../solution-brief.md (needs reportlab: uv sync --extra docs)
python3 scripts/package.py          # embeds fonts, scripts, the PDF and the evidence into eot-presentation.html and writes ../happyrobot-eot-deliverables.zip
```

`package.py` bundles documents only, never model weights. Font and audio licence notices are in `assets/`.
