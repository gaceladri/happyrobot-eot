"""Build the self-contained deck and document ZIP; never bundle model weights."""

import base64
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def embedded(path: Path, mime: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


def main():
    css = (ROOT / "style.css").read_text()
    for name in (
        "DisplaySans-Regular.ttf",
        "DisplaySans-Bold.ttf",
        "DisplaySerif-Regular.ttf",
    ):
        css = css.replace(
            f"url('assets/{name}')",
            f"url('{embedded(ROOT / 'assets' / name, 'font/ttf')}')",
        )
    html = (
        (ROOT / "index.html")
        .read_text()
        .replace('<link rel="stylesheet" href="style.css">', f"<style>{css}</style>")
    )
    for name in ("data.js", "l041.js", "app.js"):
        script = (ROOT / name).read_text().replace("</script", "<\\/script")
        html = html.replace(
            f'<script src="{name}"></script>', f"<script>{script}</script>"
        )
    downloads = [
        ("../solution-brief.pdf", "application/pdf"),
        *[
            (f"evidence/{name}.json", "application/json")
            for name in (
                "deployment",
                "data",
                "sources",
                "reproduction",
                "history-audit",
            )
        ],
    ]
    for name, mime in downloads:
        path = ROOT / name
        html = html.replace(
            f'href="{name}"', f'download="{path.name}" href="{embedded(path, mime)}"'
        )
    (ROOT / "eot-presentation.html").write_text(html)

    # Keep the ZIP's layout identical to docs/ so relative links work.
    files = [ROOT / "eot-presentation.html", ROOT / "README.md"]
    files += [ROOT / name for name, _ in downloads if not name.startswith("../")]
    files += [
        ROOT / "assets" / name for name in ("LICENSE-audio.txt", "LICENSE-fonts.txt")
    ]
    files += [
        ROOT.parent / name
        for name in ("solution-brief.md", "solution-brief.pdf", "video-script.txt")
    ]
    bundle = ROOT.parent / "happyrobot-eot-deliverables.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT.parent))
    print(
        f"Portable HTML: {(ROOT / 'eot-presentation.html').stat().st_size / 1024**2:.2f} MB"
    )
    print(f"Bundle: {bundle}")


if __name__ == "__main__":
    main()
