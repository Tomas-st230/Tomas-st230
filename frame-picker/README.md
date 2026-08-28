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
* **Measured flatness does not detect log footage, and no longer pretends to.**
  Checked against 77 real DJI clips from one card — all the same picture profile
  — `luma_span` ran 0.155–0.925 and mean saturation 0.074–0.765 in one
  continuous distribution with no bimodal gap anywhere a threshold could sit.
  The measurement was reading the *scene*, not the profile: four dark sunset
  clips fell under the limits, had their saturation multiplied by 2.5, and came
  out neon. Flatness is now reported as a **suspicion only** and never triggers
  a transform; `--convert-log on` is how you act on it.
* **The looks, the weights and the thresholds are all chosen, not measured.**
  `--look nature` / `--look city` are parametric targets picked by argument. So
  are the four scoring weights and `--min-score`. Every one of them is printed
  into `results.json` and labelled in the report.
* **Cross-clip comparison is weaker than the per-clip ranking.** Sharpness,
  dynamic range and colourfulness are converted to percentile ranks *inside a
  clip* before scoring, because absolute thresholds are wrong across cameras,
  lenses and picture profiles. The optional `--global-top` section compares
  clips against each other and is labelled as less reliable.

---

## Install

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

python -m pip install -e ".[gui,dev]"     # CLI + drag-and-drop window + tests
python -m pip install -e .                # CLI only
python -m pytest -q                        # 62 tests, all should pass
```

`ffmpeg` and `ffprobe` must be on `PATH`. Nothing else shells out: every
external process in this project goes through `framepicker/proc.py`, and a test
enforces that.

**About OpenCV.** The dependency is `opencv-contrib-python`, not plain
`opencv-python`, because `cv2.saliency` — the composition term — exists only in
contrib. The two packages share the `cv2` namespace and
[upstream is explicit](https://github.com/opencv/opencv-python) that only **one**
may be installed in an environment ("If you installed multiple different
packages in the same environment, uninstall them all with `pip uninstall` and
reinstall only one package"), so contrib cannot be offered as an optional extra
that gets *added* to an existing `opencv-python` — that would break the
environment instead of extending it.

If you already have `opencv-python` (or the `-headless` variant) in the
environment, uninstall it first:

```bash
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install opencv-contrib-python
# on a server with no display, use opencv-contrib-python-headless instead
```

Running on plain `opencv-python` anyway is supported: the composition term is
dropped, the run continues, and the report says the term was dropped — it is
never silently scored as zero.

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

`VIDEO...` takes files, **folders**, or **wildcards**. The expansion happens
inside the tool, not in the shell, because neither `cmd` nor PowerShell globs
arguments for a Python program - `*.MP4` would otherwise arrive verbatim and
match nothing:

```powershell
python -m framepicker "D:\video\DJI_0001.MP4" --out D:\pickai
python -m framepicker D:\video --out D:\pickai
python -m framepicker "D:\video\*.MP4" --out D:\pickai
```

An input that matches nothing is named in the console and listed in
`results.json` under `skipped`; it is never silently dropped.

Files are processed **oldest first**, by the file's modification time — which is
recording order for a card copied off a camera, and stays right when the frame
counter has rolled over or two cards are mixed. Ties break on filename so the
order is stable between runs. `--order name` sorts by filename instead, and
`--order none` keeps the order the arguments were given in. `results.json`
records which order was used, and `probe.extra.format_tags` carries the
container's own `creation_time` for every clip, so if a card copy lost its
timestamps you can see that rather than wonder.

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `frame-picker-out` | output directory |
| `--select` | `threshold` | `threshold` keeps every frame above `--min-score`; `count` aims at `--per-clip` |
| `--min-score` | `0.65` | score a frame has to reach in threshold mode — **an uncalibrated starting value** |
| `--max-per-clip` | `12` | upper bound per clip in threshold mode (`0` = no bound) |
| `--per-clip` | `6` | target frames per clip, **count mode only** |
| `--export-height` | `0` | scale exported stills down to this height (e.g. `1080`); `0` keeps the source resolution |
| `--fps` | `2` | analysis sampling rate |
| `--min-gap` | `2.0` | minimum seconds between two picked frames (or 3 % of the clip, whichever is larger) |
| `--convert-log` | `auto` | `auto` uses metadata then filename hints; `on`/`off` force it |
| `--lut` | – | `.cube` applied **only to clips detected as log**, for analysis *and* for the exported image |
| `--lut-all` | off | force `--lut` onto every clip, log or not |
| `--normalise-strength` | `1.0` | strength of the no-LUT log fallback; `0` turns it off |
| `--look` | `none` | named look for the exported stills: `nature`, `city`, `none` |
| `--look-strength` | `0.6` | how far toward the look's targets to travel |
| `--jobs` | auto | files processed in parallel (auto = `min(4, cpu_count)`) |
| `--no-faces` | off | skip face detection entirely |
| `--face-model` | – | explicit path to a YuNet `.onnx` |
| `--format` | `jpg` | `jpg` or `png` |
| `--jpeg-quality` | `2` | ffmpeg `-q:v` (2 = best) |
| `--global-top` | `20` | frames on the "best of the whole batch" section (`0` turns it off) |
| `--max-candidates` | `3000` | upper bound on analysis frames buffered per clip |
| `--hwaccel` | `auto` | `auto` tries CUDA, `none` forces CPU |
| `--order` | `date` | processing order: `date` = oldest file first, `name` = by filename, `none` = as given |

GUI — a wrapper around the same functions. There is no second code path, and
`gui/` holds no logic: it collects paths and settings, hands them to
`run_batch`, and renders what comes back.

```bash
python -m gui.drop_window
```

One window:

* a drop area (files or whole folders; double-click opens a file dialog). A
  dropped folder is expanded immediately, so the table shows one row per file
  in the order they will be processed
* a settings block — the drone LUT with a file picker and the "apply to every
  file" override, the log-detection mode, the quality threshold, the per-clip
  bound, and the output folder
* a table with one row per file: **profile** (log / normal), **colour** (LUT /
  normalised / untouched), **decode path** (GPU / CPU), frames picked, status.
  A skipped file says so and carries the reason as a tooltip — it never goes
  blank.
* a progress bar, the current status line, cancel, "open results folder" and
  "open report"

Cancelling deletes the partial output.

### Output

```
frame-picker-out/
  <clip>_01_0004.000s_0.636.jpg     # sortable by rank
  results.json                       # everything, machine-readable
  report.html                        # self-contained, images inlined
```

### End-of-run check

Before the report is written, every preview is built and the whole output is
verified: each frame the report claims is on disk, non-empty, and has its
preview embedded. Failed exports, unreadable files and unreferenced leftovers
are all listed by name. The verdict prints to the console, appears at the top of
`report.html`, and lands in `results.json` under `integrity`. A frame whose
preview could not be produced renders a stated finding rather than a blank gap —
which is what the first large run did, silently.

### While it runs

The machine is asked not to fall asleep on its own for the duration of the
batch (Windows: `ES_CONTINUOUS | ES_SYSTEM_REQUIRED`, released on the way out,
including on cancel and on exception). `ES_DISPLAY_REQUIRED` is deliberately
**not** set and a test asserts it stays that way: the screen still goes dark,
the machine still locks, and shutdown, restart, log-off and manual sleep all
keep working exactly as before. On other platforms it is a no-op that says so.

`results.json` per clip: probe data, the log verdict and what it was based on,
the colour transform used, the decode path (`hw`/`cpu`) and why, candidates
evaluated, rejects broken down by cause, the confidence verdict, the selection
result with any shortfall reason, wall-clock time, and every chosen frame with
its timestamp, score, full feature values and reasons.

Exported frames are cut from the **source file** at its own resolution, never
from the 640 px analysis proxy: 4K comes out 4K, 2.7K comes out 2.7K.
`--export-height 1080` scales the exported still down; it only ever scales
**down**, so a 1080p source asked for 1080p is left untouched rather than
upscaled. The source resolution stays in `results.json` either way. If a LUT or a normalisation was used for analysis,
the same transform is applied to the exported image so that what you look at is
what was scored; the untouched timestamp stays in `results.json` so a
full-quality re-grab can be redone later.

---

## How a frame is scored

`framepicker/scoring.py`, one dict, one place:

```python
WEIGHTS = {"content": 0.50, "technical": 0.25, "composition": 0.15, "moment": 0.10}
```

`content = max(face_component, landscape_component)`

`moment` answers "is this a moving shot, and is anyone in it": motion is the
mean luma change between consecutive samples, ranked within the clip, and it
counts for more when a face is present than when the frame is empty. A
locked-off shot scores none of this term. The first frame of a clip has no
previous frame, so its motion is `None` — dropped, not zero.

`technical` also carries a **horizon tilt** penalty. A crooked horizon is the
one composition defect in drone footage that is unambiguous and cheap to
measure, so it is measured: the dominant near-horizontal line's angle, withheld
as `None` unless the candidate lines agree on an angle, sit at the same height,
and have genuinely different brightness on each side — otherwise choppy water
and forest canopy report a perfectly level horizon that is not there.

`composition` blends the rule-of-thirds placement with **subject separation**:
how much more salient the main blob is than everything around it.

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

### Log / flat footage and the LUT

Detection order: `--convert-log` flag → colour metadata → filename hint →
**measured frame flatness** → off.

The measurement step exists because the first three usually say nothing about a
real DJI file: DJI does not put the picture profile in the filename, and often
does not tag the colour transfer either. So twelve frames spread over the clip
are decoded and two things are measured — the luma p1–p99 span and the mean HSV
saturation — which is exactly what a log profile compresses. Both have to be
below their threshold for the clip to count as log. It is still a guess, it is
still labelled as one, and the measured numbers go into `results.json` so the
thresholds can be calibrated.

**A LUT follows the log verdict, per clip.** `--lut look.cube` is applied only
to the clips detected as log — a `.cube` is a log-to-display conversion, and
putting one on footage that is already Rec.709 wrecks it. A mixed card of D-Log
and normal clips is the normal case, so this is the default; `--lut-all` forces
the LUT onto everything when you know detection is wrong. When a clip is log and
no LUT is supplied, the percentile normalisation is used instead, and the report
says it is an approximation.

The console names the decision for every file, so you can see which ones got the
LUT:

```
Aptiktas plokščias (log) profilis. Pagrindas: išmatuotas kadrų plokštumas
Išmatuota iš kadrų: šviesumo diapazonas 0.412, sodrumas 0.155
Analizei pritaikytas LUT: D:\LUT\DJI_DLogM_to_Rec709.cube
```

### Looks

`--look nature` and `--look city` grade the **exported stills only** — never the
analysis frames, because scores have to stay comparable between clips and a look
is a taste decision while a score is meant to be a measurement.

A look is not a curve pasted on top. Each preset is a set of *targets* (a
saturation, a contrast span, a warmth tilt, a shadow lift, a highlight
rolloff), and each frame is measured first and then moved `--look-strength` of
the way toward them. A frame that is already rich is barely touched; a flat one
is lifted. Measured on the same source frame at strength 0.6:

| Look | luma span | mean saturation |
|---|---|---|
| none | 0.18 | 0.19 |
| nature | 0.30 | 0.37 |
| city | 0.28 | 0.30 |

Three bounds exist because the unbounded version of this already broke real
exports: the contrast stretch is capped at 1.6×, the saturation multiplier at
1.45×, and the highlight rolloff is asymptotic so a bright sky keeps its
gradation instead of becoming one flat white shape. Without the contrast cap a
foggy frame asks for a 140× stretch and its last trace of colour explodes.

### Confidence

After ranking, the spread of the top 20 scores is measured. If they fall inside
a narrow band (starting threshold: range < 0.08 on a 0–1 scale), the console and
the report say the ranking carries little information and the frames are
effectively equivalent — choose visually. A flat distribution is never presented
in the same confident format as a real one.

### Selection

Two modes. **Threshold is the default**: every frame scoring at least
`--min-score` is taken, so a weak clip can return two frames and a strong one
twelve. `--max-per-clip 12` bounds it so a long file cannot quietly export a
hundred stills; `--max-per-clip 0` removes the bound.

`--min-score 0.65` is **a starting value chosen by argument, not by
measurement** — the same status as the weights, and it means little until the
weights are calibrated. Every run prints the threshold with that warning
attached. When no frame in a clip reaches it, the tool says so, names the best
score it did see, and tells you to lower the bar; it does not quietly write an
empty folder.

`--select count` is the old behaviour: aim at `--per-clip`, and report any
shortfall with a stated reason.

Either mode then walks the ranking greedily, keeping a candidate only if it is

1. at least `--min-gap` seconds (or 3 % of the clip) from every frame already
   kept, **and**
2. visually different from every frame already kept — dHash Hamming distance
   above a threshold, or colour histogram distance above a threshold. dHash is
   implemented directly in `features.py`; it does not justify a dependency.

In count mode, if the constraints cannot produce the requested count, fewer
frames come back **with a stated reason** ("the clip is 5.0 s long and the
minimum gap is 2.0 s — at most 3 fit", "the remaining candidates are
near-identical"). Silently returning three frames when six were asked for is the
failure mode this project keeps paying for. In threshold mode there is no fixed
target to fall short of, so `frames_requested` is `null` in `results.json`
rather than a number the run never promised — but the counts that led to the
result (candidates above the threshold, rejected as duplicates, rejected on the
time gap, capped) are all still reported.

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
| Settings | defaults (`--select threshold --min-score 0.65 --min-gap 2`), faces on, saliency on |
| Result | **4.7 s per minute of footage** |

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
| `opencv-contrib-python` (or `opencv-python`) | Apache 2.0 |
| `PySide6` (GUI only) | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| `pytest` (dev only) | MIT |
| YuNet ONNX model (`opencv/opencv_zoo`) | MIT, © 2020 Shiqi Yu |
| ffmpeg / ffprobe (external, not bundled) | LGPL-2.1-or-later, or GPL-2.0-or-later if the build enables GPL components |

Deliberately **not** used: `pyiqa` (non-commercial licence), InsightFace weights
(commercially restricted), torch / CLIP / LAION aesthetic models (v1 stays fast
and dependency-light — a `--aesthetic` flag degrading gracefully without torch
is a reasonable v2), DaVinci Resolve (not required for anything).

---

## Decisions taken

The four open questions in the task document were put to Tomas and answered.
What that changed:

1. **Per-clip count: quality threshold.** `--select threshold` is now the
   default: every frame above `--min-score` is taken, two from a weak clip and
   twelve from a strong one, bounded by `--max-per-clip 12`. The old fixed
   target is still there as `--select count`. The threshold itself is
   uncalibrated and everything that prints it says so.
2. **Export resolution: native.** Stills come out at the source resolution —
   4K stays 4K, 2.7K stays 2.7K — and `--export-height 1080` is there when a
   1080p copy is wanted. Format stays JPEG `-q:v 2` by default with
   `--format png` available; if "native quality" should also mean lossless
   rather than only full-resolution, that is a one-word change to the default.
3. **Global batch page: on by default.** `--global-top 20` now, still labelled
   in the report as less reliable than the per-clip ranking, because it
   compares raw values across different cameras and picture profiles.
   `--global-top 0` turns it off.
4. **Faces: always better.** Unchanged — any face over 1 % of the frame sets a
   high content floor for every clip, drone footage included.

Asked for separately and also done: the LUT is applied per clip according to the
log verdict rather than to the whole batch, frame flatness was added as a
detection signal because DJI files carry no usable filename or colour tag, and
the window grew a settings block and a per-file table.

## Still open

* The weights, the threshold, and the diversity thresholds are all uncalibrated
  starting values. Calibrating them needs a batch of Tomas's own footage plus
  his own picks to compare against; nothing here can substitute for that.
* Whether `--min-score 0.65` is a sensible default is unknown. It was chosen
  from the structure of the score, not from measurement: on the synthetic
  `testsrc2` fixtures the best frame of a clip scores 0.64, so that clip
  correctly returns nothing and says why. Real footage may want a different
  number in either direction.
