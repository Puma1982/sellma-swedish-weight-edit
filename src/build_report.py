"""Build report/report.pdf from report/research_note.md (+ appendix tables and figures).
Needs `markdown` and a local Chrome/Chromium for HTML->PDF (falls back to writing report.html only)."""
import pathlib, subprocess, shutil, markdown
ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "report"
note = (R / "research_note.md").read_text(encoding="utf-8")
app = ["\n\n---\n\n# Appendix (report only)\n"]
for title, f in (("Main results table (greedy decoding, test set)", "results/results_table.md"),
                 ("Held-out perplexity", "results/perplexity_table.md")):
    p = ROOT / f
    if p.exists(): app.append(f"\n## {title}\n\n" + p.read_text(encoding="utf-8"))
for cap, f in (("Swedish-answer rate per condition", "swedish_rate_by_condition.png"), ("Perplexity per condition", "perplexity.png")):
    if (ROOT / "figures" / f).exists(): app.append(f"\n![{cap}](../figures/{f})\n")
sel = ROOT / "results" / "dev_selection.json"
html_body = markdown.markdown(note + "".join(app), extensions=["tables", "fenced_code"])
css = "body{font-family:Georgia,serif;max-width:780px;margin:24px auto;font-size:12px;line-height:1.45}table{border-collapse:collapse;font-size:10.5px}td,th{border:1px solid #999;padding:3px 6px}img{max-width:100%}h1{font-size:20px}h2{font-size:15px}code{font-size:11px}"
(R / "report.html").write_text(f"<!doctype html><meta charset=utf-8><style>{css}</style>{html_body}", encoding="utf-8")
chrome = next((c for c in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", shutil.which("chromium"), shutil.which("google-chrome")] if c and pathlib.Path(c).exists()), None)
if chrome:
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={R/'report.pdf'}", (R / "report.html").as_uri()], check=True, capture_output=True)
    (R / "report.html").unlink()
    print("wrote", R / "report.pdf")
else:
    print("no Chrome found; wrote report.html (print to PDF manually)")
