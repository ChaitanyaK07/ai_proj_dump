"""Render docs/report/REPORT.md to docs/report/WayFinder_Report.pdf.

    python -m pip install markdown
    python scripts/build_report.py

Markdown -> styled HTML (docs/report/_build/, git-ignored) -> PDF through a headless Edge or Chrome,
whichever is installed. Figures are read from results/plots/ via the relative links in REPORT.md.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "report"
SOURCE = REPORT_DIR / "REPORT.md"
BUILD = REPORT_DIR / "_build"
PDF = REPORT_DIR / "WayFinder_Report.pdf"

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Cambria", "Georgia", serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 22pt; margin: 0 0 4pt; color: #1b2a49; }
h2 { font-size: 14.5pt; color: #1b2a49; border-bottom: 1px solid #c9ced8; padding-bottom: 3pt; margin-top: 20pt;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #1b2a49; margin-top: 12pt; page-break-after: avoid; }
p, li { text-align: justify; }
code { font-family: "Consolas", "Courier New", monospace; font-size: 9pt; background: #f3f4f6; padding: 0 2px; }
pre { background: #f3f4f6; border: 1px solid #dfe3ea; padding: 7pt 9pt; font-size: 8.6pt; line-height: 1.35;
      white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 8.8pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9ced8; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #eef1f6; }
img { display: block; max-width: 82%; margin: 8pt auto 12pt; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #c9ced8; margin: 12pt 0; }
a { color: #1f4e9c; text-decoration: none; }
"""


def installed_browsers() -> list[str]:
    found = []
    for b in BROWSERS:
        path = b if Path(b).exists() else shutil.which(b)
        if path and path not in found:
            found.append(path)
    return found


def print_pdf(browser: str, html: Path, out: Path) -> bool:
    """Headless print into a temp folder (some browsers cannot write into synced folders such as OneDrive)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pdf = Path(tmp) / "report.pdf"
        # a separate profile, so an already-open browser window cannot swallow the headless run
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-first-run",
                        f"--user-data-dir={Path(tmp) / 'profile'}", "--no-pdf-header-footer",
                        f"--print-to-pdf={tmp_pdf}", html.as_uri()],
                       capture_output=True, text=True, timeout=180)
        if tmp_pdf.exists() and tmp_pdf.stat().st_size > 10_000:
            shutil.copyfile(tmp_pdf, out)
            return True
    return False


def main() -> None:
    body = markdown.markdown(SOURCE.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists"])
    BUILD.mkdir(parents=True, exist_ok=True)
    html = BUILD / "report.html"
    base = REPORT_DIR.as_uri() + "/"           # so the ../../results/plots links resolve
    html.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><base href="{base}">'
                    f"<title>Routes That Fit You — Project Report</title><style>{CSS}</style></head>"
                    f"<body>{body}</body></html>", encoding="utf-8")
    browsers = installed_browsers()
    if not browsers:
        sys.exit(f"error: no Edge or Chrome found; open {html} in a browser and print it to PDF")
    for browser in browsers:
        if print_pdf(browser, html, PDF):
            print(f"wrote {PDF.relative_to(ROOT)} ({PDF.stat().st_size / 1e6:.1f} MB) with {Path(browser).name}")
            return
    sys.exit(f"error: none of {', '.join(Path(b).name for b in browsers)} produced a PDF; "
             f"open {html} and print it to PDF")


if __name__ == "__main__":
    main()
