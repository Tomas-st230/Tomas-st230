# frame-picker

Drop video files in, get the best 5–6 frames out of each one as images, plus a
one-page HTML report that says **why** every frame was picked.

Intended use: fast selection of stills, thumbnail candidates and B-roll markers
from drone and event footage, without opening an NLE.

This is a standalone tool. It does not import from VideoPipeline Studio v4 and
v4 does not import from it.

---

## Status of the numbers in this tool

Read this before trusting any output.

* **The scoring weights are not calibrated.** `WEIGHTS` in
  `framepicker/scoring.py` is a starting point chosen by argument, not by
  measurement. Every run prints the weights into `results.json` and the report
  says so out loud.
* **The diversity and confidence thresholds are starting values too**
  (`DHASH_MIN_DISTANCE`, `HISTOGRAM_MIN_DISTANCE`, `CONFIDENCE_MIN_SPREAD`).
  They need tuning on real footage.
* **Log conversion without a LUT is an approximation.** It is a percentile
  contrast and saturation stretch, not a colour-managed conversion, and the
  report says that on every clip where it is used.
* **Log detection is a guess** unless you force it with `--convert-log`. The
  guess, and what it was based on, is printed and written to `results.json`.
* **Cross-clip comparison is weaker than the per-clip ranking.** Sharpness,
  dynamic range and colourfulness are converted to percentile ranks *inside a
  clip* before scoring, because absolute thresholds are wrong across cameras,
  lenses and picture profiles. The optional `--global-top` section compares
  clips against each other and is labelled as less reliable.

---

## Install

```bash
python -m pip install -e .            # CLI
python -m pip install -e ".[gui]"     # + PySide6 for the drag-and-drop window
python -m pip install -e ".[dev]"     # + pytest
```

`ffmpeg` and `ffprobe` must be on `PATH`. Nothing else shells out: every
external process in this project goes through `framepicker/proc.py`, and a test
enforces that.

`opencv-contrib-python` is optional. Without it the saliency-based composition
term is dropped, the run continues, and the report says the term was dropped —
it is never silently scored as zero.

### Face detection model

Face detection uses **YuNet** from the official
[`opencv/opencv_zoo`](https://github.com/opencv/opencv_zoo) repository,
`models/face_detection_yunet/`. The filenames published there, checked against
the repository rather than guessed:

| File | Note |
|---|---|
| `face_detection_yunet_2026may.onnx` | newest |
| `face_detection_yunet_2023mar.onnx` | **shipped in `framepicker/models/`** |
| `face_detection_yunet_2023mar_int8bq.onnx` | quantised |

`framepicker/models/face_detection_yunet_2023mar.onnx` is already in this repo,
together with its MIT `LICENSE`, so face detection works out of the box. To use
a different one, drop it into `framepicker/models/` (the newest matching name
wins) or pass `--face-model path.onnx`.

Note that opencv_zoo stores these files with **Git LFS**: a plain
`raw.githubusercontent.com` download returns a 130-byte pointer file, not a
model. Use `git lfs` or the `media.githubusercontent.com/media/...` URL.

**Without a model** the tool runs in statistics-only mode: `face_count` and
`face_max_rel` stay `None`, faces are excluded from scoring, and both the
console and the report say so. `None` and `0` are not the same thing — a frame
with no detector is not a frame with no faces.

---

## Use

```bash
python -m framepicker VIDEO... [--out DIR] [--per-clip 6] [--fps 2]
                               [--min-gap 2.0] [--convert-log auto|on|off]
                               [--lut path.cube] [--jobs N] [--no-faces]
```

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `frame-picker-out` | output directory |
| `--per-clip` | `6` | frames to pick per clip |
| `--fps` | `2` | analysis sampling rate |
| `--min-gap` | `2.0` | minimum seconds between two picked frames (or 3 % of the clip, whichever is larger) |
| `--convert-log` | `auto` | `auto` uses metadata then filename hints; `on`/`off` force it |
| `--lut` | – | `.cube` applied to **every** clip, for analysis *and* for the exported image |
| `--jobs` | auto | files processed in parallel (auto = `min(4, cpu_count)`) |
| `--no-faces` | off | skip face detection entirely |
| `--face-model` | – | explicit path to a YuNet `.onnx` |
| `--format` | `jpg` | `jpg` or `png` |
| `--jpeg-quality` | `2` | ffmpeg `-q:v` (2 = best) |
| `--global-top N` | `0` | also produce a "best of the whole batch" section |
| `--max-candidates` | `3000` | upper bound on analysis frames buffered per clip |
| `--hwaccel` | `auto` | `auto` tries CUDA, `none` forces CPU |

GUI (a wrapper around the same functions — there is no second code path):

```bash
python -m gui.drop_window
```

One window: a drop area, a progress line, a cancel button, and an "open results
folder" button when it finishes. Cancelling deletes the partial output.

### Output

```
frame-picker-out/
  <clip>_01_0004.000s_0.636.jpg     # sortable by rank
  results.json                       # everything, machine-readable
  report.html                        # self-contained, images inlined
```

`results.json` per clip: probe data, the log verdict and what it was based on,
the colour transform used, the decode path (`hw`/`cpu`) and why, candidates
evaluated, rejects broken down by cause, the confidence verdict, the selection
result with any shortfall reason, wall-clock time, and every chosen frame with
its timestamp, score, full feature values and reasons.

Exported frames are cut from the **source file** at full resolution, never from
the 640 px analysis proxy. If a LUT or a normalisation was used for analysis,
the same transform is applied to the exported image so that what you look at is
what was scored; the untouched timestamp stays in `results.json` so a
full-quality re-grab can be redone later.

---

## How a frame is scored

`framepicker/scoring.py`, one dict, one place:

```python
WEIGHTS = {"content": 0.55, "technical": 0.30, "composition": 0.15}
```

`content = max(face_component, landscape_component)`

Four principles, stated so they can be argued with:

1. **Technical quality is a gate, not a ladder.** Past "sharp enough", more
   sharpness buys almost nothing; below it, the frame is punished. A razor
   sharp frame of nothing is still a frame of nothing.
2. **Percentile ranks are relative.** `colorfulness_rank == 1.0` means "the most
   colourful frame in *this* clip", not "beautiful", so the landscape term is
   capped below what a real subject can earn.
3. **A recognisable face is content.** Any face above 1 % of the frame gets a
   high content floor and saturates around a half-frame face. B-roll with no
   people is not punished for having no people; a person in frame is not
   out-ranked by a textured empty landscape.
4. **A missing detector is not a zero.** `None` drops the term and renormalises
   the weights.

`score_frame_explained(features) -> (score, reasons)` returns the reasons in
Lithuanian, e.g. `veidas užima 22 % kadro`,
`ryškumas — 91-as procentilis šiame įraše`, `12 % kadro perdegę`. A score with
no reasons is indistinguishable from a random number.

### Confidence

After ranking, the spread of the top 20 scores is measured. If they fall inside
a narrow band (starting threshold: range < 0.08 on a 0–1 scale), the console and
the report say the ranking carries little information and the frames are
effectively equivalent — choose visually. A flat distribution is never presented
in the same confident format as a real one.

### Selection

Greedy from the top of the ranking, keeping a candidate only if it is

1. at least `--min-gap` seconds (or 3 % of the clip) from every frame already
   kept, **and**
2. visually different from every frame already kept — dHash Hamming distance
   above a threshold, or colour histogram distance above a threshold. dHash is
   implemented directly in `features.py`; it does not justify a dependency.

If the constraints cannot produce the requested count, fewer frames come back
**with a stated reason** ("the clip is 5.0 s long and the minimum gap is 2.0 s —
at most 3 fit", "the remaining candidates are near-identical"). Silently
returning three frames when six were asked for is the failure mode this project
keeps paying for.

---

## Measured performance

Rule: no speed claim that no run produced. The tool prints its own throughput
in seconds per minute of footage at the end of every run, measured on the
machine it just ran on. Read that number, not this one.

The only figure measured so far, for reference:

| Measurement | Value |
|---|---|
| Clip | 1920×1080, H.264, 30 fps, 60 s (ffmpeg `testsrc2`) |
| Machine | 4-core Intel Xeon @ 2.10 GHz container, no GPU |
| Decode path | CPU (`-hwaccel cuda` failed: `Cannot load libcuda.so.1`) |
| Settings | `--per-clip 6 --min-gap 2`, faces on, saliency on |
| Result | **4.8 s per minute of footage** |

This says nothing about the target RTX 2060 machine, about hardware decode, or
about real camera footage. Synthetic `testsrc2` is not representative content —
it is uniform enough that the confidence check correctly reported the ranking as
uninformative. Measure on real clips before quoting anything.

Hardware decode is attempted, not assumed: the tool runs one `-hwaccel cuda`
single-frame decode, and falls back to CPU only if that actually fails. Which
path was used, and the exact reason for any fallback, goes into the console and
into `results.json`.

---

## Layout

```
framepicker/          runs with no Qt and no display (a test asserts it)
  cli.py              entry point + the batch pipeline; the GUI calls run_batch()
  probe.py            ffprobe wrapper
  decode.py           ffmpeg -> downscaled RGB frames, log detection, normalisation
  features.py         pure ndarray -> float measurements, plus optional detectors
  scoring.py          pure feature dict -> score + reasons
  select.py           ranking + diversity constraints + shortfall reasons
  export.py           full-resolution extraction of the chosen timestamps
  report.py           results.json + report.html
  proc.py             the ONLY module that calls subprocess
  strings_lt.py       every Lithuanian user-facing string
  models/             YuNet ONNX + its MIT licence
gui/
  drop_window.py      thin drag-and-drop window, no logic
tests/
```

`gui/` may import `framepicker`; never the reverse.

---

## Tests

```bash
python -m pytest -v
```

Fixtures are generated by ffmpeg at test time (`testsrc2`, plus blurred,
flattened and burnt-in-frame-counter variants). No binary fixtures live in the
repository. ffmpeg is invoked through `framepicker.proc` in the tests too.

Enforced rules, each one a defect that already cost time:

* `test_score_prefers_person_over_empty_aerial` — hand-built feature dicts, so
  the test cannot mirror the pipeline.
* `test_missing_detector_is_none_not_zero` — no model, `face_*` is `None`, the
  run continues, the report states statistics-only mode.
* `test_diversity_rejects_near_duplicates`.
* `test_shortfall_is_reported` — a 5 s clip asked for 6 frames returns fewer
  **and** a stated reason.
* `test_export_timestamp_accuracy` — the exported frame matches the requested
  timestamp within ±1 frame, read back from a burnt-in counter with no OCR and
  no magic offset (the frame-to-marker mapping is measured from the fixture).
* `test_log_clip_is_not_penalised` — a flattened copy does not collapse to
  near-zero landscape scores.
* `test_no_direct_subprocess` — only `proc.py` imports `subprocess`.
* `test_headless` — `import framepicker` with Qt blocked and no `DISPLAY`.
* `test_no_hardcoded_lt_strings`.
* `test_unreadable_file_does_not_abort_batch`.

On Windows every child process is started with `CREATE_NO_WINDOW` and a hidden
`STARTUPINFO`, so no console flashes behind the GUI.

---

## Dependency licences

| Dependency | Licence |
|---|---|
| `numpy` | BSD-3-Clause (bundled components: 0BSD, MIT, Zlib, CC0-1.0) |
| `opencv-python` / `opencv-contrib-python` | Apache 2.0 |
| `PySide6` (GUI only) | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| `pytest` (dev only) | MIT |
| YuNet ONNX model (`opencv/opencv_zoo`) | MIT, © 2020 Shiqi Yu |
| ffmpeg / ffprobe (external, not bundled) | LGPL-2.1-or-later, or GPL-2.0-or-later if the build enables GPL components |

Deliberately **not** used: `pyiqa` (non-commercial licence), InsightFace weights
(commercially restricted), torch / CLIP / LAION aesthetic models (v1 stays fast
and dependency-light — a `--aesthetic` flag degrading gracefully without torch
is a reasonable v2), DaVinci Resolve (not required for anything).

---

## Open questions

Not decided here. The current behaviour is whatever the task document specified,
and each answer is a flag flip rather than a rewrite:

1. **Fixed count or quality threshold?** Right now `--per-clip 6` is a fixed
   target and a shortfall is reported with a reason. A threshold mode that
   returns 2 or 12 frames does not exist yet.
2. **JPEG q2 or PNG?** Default is JPEG `-q:v 2`; `--format png` already works if
   the frames go on for further grading.
3. **A global "top 20 of the batch" page?** Implemented behind `--global-top N`,
   off by default, and labelled less reliable than the per-clip ranking.
4. **Are faces always better?** Currently yes — a face above 1 % of the frame
   sets a high content floor for every clip. If that is true for event footage
   but wrong for drone work, the fix is a per-run switch, not a new model.
