"""``results.json`` and a single self-contained ``report.html``.

The report exists so that a pick can be argued with. Every frame carries its
score, its measured features and the reasons the score came out that way; a
flat, uninformative ranking is labelled as such instead of being dressed up
in the same confident layout as a real one.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import os
from typing import Iterable

from . import strings_lt as S

RESULTS_JSON = "results.json"
REPORT_HTML = "report.html"

#: Long edge of the preview images embedded in the HTML. The exported files
#: are the deliverable; these are only there to look at.
PREVIEW_LONG_EDGE = 900

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
       background: #f6f6f7; color: #16181d; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 32px 0 8px; }
.sub { color: #5c6270; font-size: 13px; margin-bottom: 20px; }
.card { background: #fff; border: 1px solid #e2e4e9; border-radius: 10px; padding: 16px 18px;
        margin-bottom: 18px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: #4a505c; margin: 6px 0 0; }
.meta b { color: #16181d; font-weight: 600; }
.note { border-left: 3px solid #b9bec9; padding: 8px 12px; margin: 10px 0; background: #f2f3f5;
        font-size: 13px; border-radius: 0 6px 6px 0; }
.note.warn { border-left-color: #d08a00; background: #fdf6e7; }
.note.bad { border-left-color: #c0392b; background: #fbeceb; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px;
        margin-top: 14px; }
.shot { background: #fff; border: 1px solid #e2e4e9; border-radius: 8px; overflow: hidden; }
.shot img { display: block; width: 100%; height: auto; background: #222; }
.shot .body { padding: 10px 12px 12px; }
.rank { font-weight: 700; }
.score { font-variant-numeric: tabular-nums; }
ul.reasons { margin: 6px 0 0; padding-left: 18px; font-size: 13px; color: #4a505c; }
table { border-collapse: collapse; font-size: 13px; margin-top: 8px; }
td, th { border: 1px solid #e2e4e9; padding: 4px 8px; text-align: left; }
code { font: 12px/1.4 ui-monospace, Menlo, Consolas, monospace; }
@media (prefers-color-scheme: dark) {
  body { background: #16181d; color: #e8eaee; }
  .card, .shot { background: #1f2229; border-color: #333843; }
  .note { background: #262a32; border-left-color: #4a505c; }
  .note.warn { background: #33291a; }
  .note.bad { background: #33201e; }
  .meta, ul.reasons { color: #a7adba; }
  .meta b { color: #e8eaee; }
  td, th { border-color: #333843; }
}
"""


def write_results_json(results: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, RESULTS_JSON)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, default=str)
    return path


def _preview_data_uri(path: str, long_edge: int = PREVIEW_LONG_EDGE) -> str | None:
    """Base64 JPEG preview of *path*, or ``None`` if it cannot be read."""
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]
    if cv2 is None:
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
        except OSError:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = long_edge / float(max(width, height))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _short(text: str, limit: int = 120) -> str:
    """Keep a decoder error readable in a metadata row; results.json keeps it whole."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _note(text: str, level: str = "") -> str:
    css = "note" + (f" {level}" if level else "")
    return f'<div class="{css}">{_esc(text)}</div>'


def _meta_row(pairs: Iterable[tuple[str, str]]) -> str:
    cells = "".join(f"<span><b>{_esc(k)}:</b> {_esc(v)}</span>" for k, v in pairs if v not in (None, ""))
    return f'<div class="meta">{cells}</div>'


def _shot_card(frame: dict, out_dir: str) -> str:
    file_name = frame.get("file") or ""
    uri = _preview_data_uri(os.path.join(out_dir, file_name)) if file_name else None
    img = f'<img src="{uri}" alt="{_esc(file_name)}">' if uri else ""
    reasons = "".join(f"<li>{_esc(r)}</li>" for r in frame.get("reasons", []))
    return (
        '<div class="shot">'
        f"{img}"
        '<div class="body">'
        f'<div><span class="rank">#{_esc(frame.get("rank"))}</span> '
        f'<span class="score">{S.REPORT_SCORE}: {frame.get("score", 0.0):.3f}</span> &middot; '
        f'{S.REPORT_TIMESTAMP}: {frame.get("timestamp", 0.0):.3f} s</div>'
        f'<div style="font-size:12px;color:#7a808c"><code>{_esc(file_name)}</code></div>'
        f'<ul class="reasons">{reasons}</ul>'
        "</div></div>"
    )


def _clip_section(clip: dict, out_dir: str) -> str:
    probe = clip.get("probe", {})
    decode = clip.get("decode", {})
    rejects = clip.get("rejects", {})
    selection = clip.get("selection", {})
    confidence = clip.get("confidence", {})

    duration = probe.get("duration")
    duration_text = f"{duration:.1f} s" if isinstance(duration, (int, float)) else S.unknown_duration()
    fps = probe.get("fps")
    parts = [f'<div class="card"><h2>{_esc(probe.get("name", ""))}</h2>']
    parts.append(_meta_row([
        (S.REPORT_DURATION, duration_text),
        (S.REPORT_RESOLUTION, f'{probe.get("width")}x{probe.get("height")}'),
        (S.REPORT_CODEC, probe.get("codec") or ""),
        (S.REPORT_FPS, f"{fps:.3f}" if isinstance(fps, (int, float)) else ""),
        (S.REPORT_DECODE, S.decode_path_hw() if decode.get("path_used") == "hw"
         else S.decode_path_cpu(_short(decode.get("hw_error", "")))),
        (S.REPORT_CANDIDATES, str(clip.get("candidates_evaluated", 0))),
        (S.REPORT_REJECTED, str(rejects.get("total", 0))),
        (S.REPORT_ELAPSED, f'{clip.get("elapsed_s", 0.0):.1f} s'),
    ]))

    for text in clip.get("notes", []):
        parts.append(_note(text))
    if confidence:
        parts.append(_note(confidence.get("message", ""), "" if confidence.get("informative") else "warn"))
    if selection.get("shortfall"):
        parts.append(_note(S.shortfall_header(selection.get("delivered", 0), selection.get("requested", 0)), "warn"))
        for reason in selection.get("shortfall_reasons", []):
            parts.append(_note(reason, "warn"))

    frames = clip.get("frames", [])
    if not frames:
        parts.append(_note(S.REPORT_NO_FRAMES, "bad"))
    else:
        cards = "".join(_shot_card(frame, out_dir) for frame in frames)
        parts.append(f'<div class="grid">{cards}</div>')
    parts.append("</div>")
    return "".join(parts)


def _global_section(results: dict, out_dir: str) -> str:
    frames = results.get("global_top", [])
    if not frames:
        return ""
    cards = "".join(_shot_card(frame, out_dir) for frame in frames)
    return (
        f'<div class="card"><h2>{S.REPORT_GLOBAL}</h2>'
        f"{_note(S.REPORT_GLOBAL_NOTE, 'warn')}"
        f'<div class="grid">{cards}</div></div>'
    )


def write_report_html(results: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    generated = results.get("generated") or _dt.datetime.now().isoformat(timespec="seconds")
    summary = results.get("summary", {})

    head = [
        f"<h1>{S.REPORT_TITLE}</h1>",
        f'<div class="sub">{S.REPORT_GENERATED}: {_esc(generated)}</div>',
        '<div class="card">',
        f"<h2>{S.REPORT_SUMMARY}</h2>",
        _note(S.batch_summary(
            summary.get("files_given", 0),
            summary.get("files_processed", 0),
            summary.get("files_skipped", 0),
            summary.get("frames_requested", 0),
            summary.get("frames_delivered", 0),
        )),
    ]
    if summary.get("throughput_s_per_footage_minute"):
        head.append(_note(S.throughput(summary["throughput_s_per_footage_minute"])))
    for text in results.get("notes", []):
        head.append(_note(text))
    weights = results.get("weights", {})
    weight_rows = "".join(f"<tr><td>{_esc(k)}</td><td>{v}</td></tr>" for k, v in weights.items())
    head.append(f"<h2>{S.REPORT_WEIGHTS}</h2><table>{weight_rows}</table>")
    head.append(_note(S.REPORT_WEIGHTS_NOTE, "warn"))
    skipped = results.get("skipped", [])
    if skipped:
        rows = "".join(
            f'<tr><td><code>{_esc(item.get("path"))}</code></td><td>{_esc(item.get("reason"))}</td></tr>'
            for item in skipped
        )
        head.append(f"<h2>{S.REPORT_SKIPPED_FILES}</h2><table>{rows}</table>")
    head.append("</div>")

    body = "".join(head)
    body += _global_section(results, out_dir)
    body += "".join(_clip_section(clip, out_dir) for clip in results.get("clips", []))

    document = (
        "<!doctype html>\n<html lang=\"lt\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{S.REPORT_TITLE}</title><style>{_CSS}</style></head><body>{body}</body></html>\n"
    )
    path = os.path.join(out_dir, REPORT_HTML)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)
    return path
