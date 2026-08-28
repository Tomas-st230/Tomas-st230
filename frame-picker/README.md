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
python -m pytest -q                        # 192 tests, all should pass
```

`ffmpeg` and `ffprobe` must be on `PATH`. Nothing else shells out: every
external process in this project goes through `framepicker/proc.py`, and a test
enforces that.

### Updating an existing copy

In PowerShell, from the folder you cloned into (`C:\Users\tomas\Tomas-st230`):

```powershell
cd C:\Users\tomas\Tomas-st230
git fetch origin
git checkout claude/best-frame-picker-tool-e25ekc
git pull origin claude/best-frame-picker-tool-e25ekc

cd frame-picker
python -m pip install -e ".[gui,dev]"    # re-run after every pull; it is quick when nothing changed
python -m pytest -q                       # 192 tests; if any fail, stop and send the output
python -m gui.drop_window
```

**About the virtual environment.** If you never made one, the commands above
are complete — they use the Python on your `PATH`, which is how this copy was
installed. If you did make one, activate it before the `pip install` line.
Find it, if you are not sure:

```powershell
Get-ChildItem C:\Users\tomas -Filter Activate.ps1 -Recurse -ErrorAction SilentlyContinue |
  Select-Object -First 5 FullName
python -c "import sys; print(sys.executable)"        # which Python you are actually using
```

`..\.venv\Scripts\Activate.ps1` failing with *"is not recognized"* simply means
there is no venv at that path. That is not an error to fix — either skip the
line, or create one and install into it:

```powershell
python -m venv C:\Users\tomas\Tomas-st230\.venv
C:\Users\tomas\Tomas-st230\.venv\Scripts\Activate.ps1
python -m pip install -e ".[gui,dev]"
```

Four more things worth knowing while updating:

* **Always re-run `pip install -e ".[gui,dev]"` after a pull.** New modules
  (`sidecar.py`, `runlog.py`, `learn.py`) are picked up automatically by the
  editable install, but a new dependency would not be.
* **`--look auto: invalid choice`** means the code is still the old one: the
  pull has not happened yet, or it landed in a different folder than the one
  you are running from. Check with
  `python -m framepicker --help | Select-String "look"`.
* **Nothing in your output folders is touched by an update.** Each run writes
  into its own `run-<date>-<time>` folder, so old runs stay as they were.
* **If `git pull` refuses because of local changes**, `git stash` them first (or
  `git checkout -- .` to throw them away), then pull.

To see what changed since your copy:

```powershell
git log --oneline HEAD..origin/claude/best-frame-picker-tool-e25ekc
```

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
                               [--lut path.cube] [--look auto] [--jobs N]
                               [--proxy auto|off] [--keyframes] [--no-faces]
```

The whole folder, the drone LUT, the look chosen per clip, oldest file first:

```powershell
python -m framepicker "D:\tomas\Videos\DJI Drone foot" `
  --lut "D:\LUT\DJI Lito X1 D-Log M to Rec.709 LUT.cube" `
  --look auto --min-score 0.60
```

`--out` defaults to `frame-picker-out` inside `D:\tomas\Videos\DJI Drone foot`
when that folder exists, and to `frame-picker-out` in the current directory
otherwise.

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
| `--out` | `frame-picker-out`, or `D:\tomas\Videos\DJI Drone foot\frame-picker-out` when that folder exists | output directory; **every run gets its own dated subfolder inside it** |
| `--no-run-folder` | off | write straight into `--out` instead of a per-run subfolder |
| `--select` | `threshold` | `threshold` keeps every frame above `--min-score`; `count` aims at `--per-clip` |
| `--min-score` | `0.60` | score a frame has to reach in threshold mode — chosen after a 163-file run, not a measured constant |
| `--max-per-clip` | `0` | upper bound per clip in threshold mode; **0 = no bound, the default** |
| `--per-clip` | `6` | target frames per clip, **count mode only** |
| `--export-height` | `0` | scale exported stills down to this height (e.g. `1080`); `0` keeps the source resolution |
| `--fps` | `2` | analysis sampling rate |
| `--min-gap` | `2.0` | minimum seconds between two picked frames (or 3 % of the clip, whichever is larger) |
| `--convert-log` | `auto` | `auto` uses metadata then filename hints; `on`/`off` force it |
| `--lut` | – | `.cube` applied **only to clips detected as log**, for analysis *and* for the exported image |
| `--lut-all` | off | force `--lut` onto every clip, log or not |
| `--lut-strength` | `auto` | how much of the LUT to apply; `auto` measures each clip and keeps the strongest setting that does not overshoot |
| `--normalise-strength` | `1.0` | strength of the no-LUT log fallback; `0` turns it off |
| `--look` | `none` | look for the exported stills: `auto`, `nature`, `city`, `none` |
| `--look-strength` | `0.6` | how far toward the look's targets to travel |
| `--jobs` | auto | files processed in parallel (auto = `min(4, cpu_count)`) |
| `--no-faces` | off | skip face detection entirely |
| `--face-model` | – | explicit path to a YuNet `.onnx` |
| `--format` | `jpg` | `jpg` or `png` |
| `--jpeg-quality` | `2` | ffmpeg `-q:v` (2 = best) |
| `--global-top` | `20` | frames on the "best of the whole batch" section (`0` turns it off) |
| `--max-candidates` | `3000` | upper bound on analysis frames buffered per clip |
| `--hwaccel` | `auto` | `auto` tries CUDA, `none` forces CPU |
| `--no-gpu-scale` | off | do not scale on the GPU even when the ffmpeg build supports it |
| `--proxy` | `auto` | use a DJI `.LRF` proxy for analysis when one sits next to the file; `off` always reads the master |
| `--keyframes` | off | decode only keyframes — much faster, and the sampling grid becomes the camera's keyframe interval |
| `--order` | `date` | processing order: `date` = oldest file first, `name` = by filename, `none` = as given |

GUI — a wrapper around the same functions. There is no second code path, and
`gui/` holds no logic: it collects paths and settings, hands them to
`run_batch`, and renders what comes back.

```bash
python -m gui.drop_window
```

**Everything the command line can do is in the window.** `OPTION_CONTROLS` in
`gui/drop_window.py` maps every field of `Options` to the widget that sets it,
and a test fails if an option is added to the pipeline without one — the two
cannot drift apart. Two fields are deliberately not controls, and say so there:
the input paths (they come from the drop area) and `--face-model` (a switch for
debugging a model file).

One window:

* a drop area (files or whole folders; double-click opens a file dialog). A
  dropped folder is expanded immediately, so the table shows one row per file
  in the order they will be processed
* settings on three tabs:
  * **Pagrindiniai** — the drone LUT with a file picker and the "apply to every
    file" override, the log-detection mode, the look and its strength, the
    quality threshold, the per-clip bound, the output folder, and the two
    switches for the per-run folder and the log
  * **Greitis** — the `.LRF` proxy, keyframe-only decoding, GPU scaling, the
    hardware decoder, files in parallel, the analysis frame rate, the candidate
    cap, and faces off. Each of the three speed switches carries its measured
    cost next to it, not an adjective.
  * **Eksportas ir atranka** — selection mode and per-clip count, minimum gap,
    export height, image format and JPEG quality, the "best of the batch"
    count, file order, and the no-LUT normalisation strength
* a table with one row per file: **profile** (log / normal), **colour** (LUT /
  normalised / untouched), **look** (which one was actually applied — the answer
  `auto` resolved to), **decode path** (GPU / CPU), frames picked, status.
  A skipped file says so and carries the reason as a tooltip — it never goes
  blank.
* a bottom panel with two tabs:
  * **Eiga** — every line the pipeline said, as it says it. The same lines go
    to `log.txt`; the view keeps the last 5000 so a 163-file run cannot fill
    memory.
  * **Reikšmės** — click any file in the table and see everything measured
    about it: the profile verdict and what decided it, `color_md`, the measured
    luma span and saturation, which conversion was applied, the proxy, the
    decode path and whether the GPU scaler was used, frames sampled,
    candidates, what was rejected and why, the confidence spread, the
    nature/city scores and the look actually applied, the best score, and the
    elapsed time. These are the same fields `log.jsonl` carries — one dict,
    built in `framepicker/runlog.py`, so the window, the console and the log
    file cannot disagree.
* a progress bar, the current status line, cancel, "open results folder",
  "open report", **"open log"**, and **"calibrate against my frames"** (asks for
  the folder of frames you kept and prints the analysis from
  `framepicker.learn` into the progress view)

The look box includes **`automatiškai (pagal sceną)`**, and the file dialogs
open in `D:\tomas\Videos\DJI Drone foot` when that folder exists.

**Cancelling means stopping.** The partial output is deleted, the folder this
run created goes with it, the file list is cleared, and the window returns to a
state where a new set of files can be dropped straight in. While a run is going
the file list is frozen and says so, rather than accepting files that would land
in a table row nobody is filling. "Open results folder" opens *this run's*
folder, not the parent.

Closing the window always works: the run is asked to stop and given at most
three seconds to notice, then the window closes regardless. Nothing in this
program can refuse a close or hold up a shutdown — the keep-awake flag is
`ES_SYSTEM_REQUIRED` only, never `ES_DISPLAY_REQUIRED`, and it is released in a
`finally` block.

### Output

Every run creates its own folder, named after the moment it started:

```
frame-picker-out/
  run-20260828-163550/
    <clip>_01_0004.000s_0.636.jpg     # sortable by rank
    results.json                       # everything, machine-readable
    report.html                        # self-contained, images inlined
  run-20260828-171204/
    ...
```

This is not tidiness for its own sake. The 163-file run wrote into a folder
that already held the previous run's stills, and the end-of-run check
correctly reported 12 files that nothing referred to - the report and the
folder no longer agreed with each other. One folder per run makes that
impossible, keeps each run usable as a deliverable on its own, and means a
cancelled run can delete everything it made, including the folder.
`--no-run-folder` restores the old behaviour. `results.json` records the
folder it was written into under `output_dir`.

### End-of-run check

Before the report is written, every preview is built and the whole output is
verified: each frame the report claims is on disk, non-empty, and has its
preview embedded. Failed exports, unreadable files and unreferenced leftovers
are all listed by name. The verdict prints to the console, appears at the top of
`report.html`, and lands in `results.json` under `integrity`. A frame whose
preview could not be produced renders a stated finding rather than a blank gap —
which is what the first large run did, silently.

### The run log

Every run writes two more files into its own folder:

| File | What it is |
|---|---|
| `log.txt` | every console line, with a timestamp. This is the file to send when something looks wrong |
| `log.jsonl` | one JSON object per line — the **values** behind those lines |

`log.jsonl` records `run_started` (the full options and weights), one `clip`
record per file, one `frame` record per exported still, `clip_failed` for
anything skipped, and `run_finished` (the summary and the integrity verdict).
The numbers stay numbers, so the log can be queried rather than only read:

```powershell
# which files were treated as D-Log, and on what evidence?
Get-Content log.jsonl | ConvertFrom-Json |
  Where-Object { $_.event -eq 'clip' } |
  Select-Object file, is_log, log_source, color_md, color_mode
```

```python
# what did the look decide, and how close was it?
import json
for line in open("log.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["event"] == "clip":
        print(r["file"], r["look_applied"], r["look_nature_score"], r["look_city_score"])
```

Both files are linked from `report.html` (under "Šio paleidimo failai"),
counted by the end-of-run check instead of being reported as leftovers, and
deleted with everything else if the run is cancelled. `--no-log` is not a flag;
the switch is in the window, or `write_log=False` on `Options`.

A log must never be able to stop a run: if the files cannot be written, the run
continues and says so at the end.

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

`composition` blends three things, each dropping out if it cannot be measured:

| Part | Share | What it is |
|---|---|---|
| placement | 0.55 | distance from the nearest rule-of-thirds intersection |
| separation | 0.20 | how much more salient the main blob is than its surroundings |
| graphic | 0.25 | the **best** of symmetry, pattern repetition and negative space |

The third part is there because the rule of thirds is not what aerial work is
currently judged on. The 2026 SkyPixel winners and this year's trend write-ups
describe the same three things instead — top-down symmetry, rhythmic pattern
turning a landscape into a graphic, and a small subject in a lot of empty space
([SkyPixel 2026 winners](https://uavcoach.com/skypixel-2026/),
[The Drone Girl](https://www.thedronegirl.com/2026/04/29/best-drone-photos-2026-skypixel/),
[Envato Elements](https://elements.envato.com/learn/photography-trends)). All
three are measurable, so they are measured rather than described:

* **symmetry** — mean absolute difference between the frame and its mirror,
  divided by the frame's own variation. That division matters: without it every
  low-contrast frame scores 0.9 simply because all its differences are small
  (uniform noise measured 0.91 before the fix, 0.30 after). `None` for a blank
  frame, because a blank frame is not symmetric, it is empty.
* **pattern repetition** — FFT autocorrelation after a high-pass and a Hann
  window, with the zero-shift neighbourhood masked out. Measured: stripes 0.97,
  a checkerboard 0.95, a sky gradient 0.00, uniform noise 0.00. Without the
  high-pass a smooth gradient scored 1.0, because a gradient correlates with
  itself at every shift.
* **negative space** — a subject occupying 0.5–18 % of the frame, weighted by
  how cleanly it separates. Outside that band it returns 0.0, and `None` when
  there is no subject measurement at all.

"Best of the three", not the average: a frame built on symmetry owes nothing to
pattern, and one built on negative space owes nothing to either. A top-down
pattern shot with no identifiable subject now gets a composition score instead
of `None`.

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

Detection order, strongest evidence first:

1. `--convert-log on|off` — you said so.
2. **`NAME.SRT`, the caption sidecar** — the camera said so, in words.
3. A log colour-transfer or wide-primaries tag.
4. A filename hint (`dlog`, `hlog`, `slog`, `flat`, …).
5. **A DJI camera plus a 10-bit pixel format.**
6. Measured frame flatness — recorded as *suspicion only*, never acted on.

**Why this needed rebuilding.** The LUT was not landing on the D-Log footage,
and the reason was not the code. Measured across 163 files from one card,
every clip reports:

```
color_transfer: bt709   color_primaries: bt709   color_space: bt709
range: tv   pix_fmt: yuv420p10le   profile: Main 10
```

The picture profile is nowhere in the video stream. It is identical for D-Log M
and for Normal colour, so steps 3 and 4 above can never fire on this camera.
Two things do carry the answer:

**`NAME.SRT` — DJI "Video Captions".** One caption per frame with the camera
settings, including the only plain-text statement of the profile:

```
FrameCnt: 1, DiffTime: 33ms
2026-08-22 19:15:39.000
[iso: 100] [shutter: 1/500.0] [fnum: 1.7] [ev: 0] [color_md: dlog_m] [focal_len: 24.00]
```

`color_md` is read from the first caption (`dlog_m`, `d-log`, `d_cinlike`,
`hlg`, `default` — compared with the punctuation stripped, because DJI spells
it differently per model). This is **decisive in both directions**: it is what
finally keeps the LUT *off* the Normal-colour clips in a mixed dump, and it is
the one signal marked `is_a_guess: false` without a flag. Turn Video Captions
on in DJI Fly and the profile stops being guessed at all.

**The container tags plus the bit depth.** Every file on the card — MP4 and LRF
alike — carries `encoder: DJI Lito X1` at container level (in
`probe.extra.format_tags`). DJI's 10-bit recording modes are D-Log, D-Log M and
HLG; Normal colour records 8-bit
([DJI](https://store.dji.com/content/what-is-10-bit),
[eDrones](https://www.edrones.review/understanding-10-bit-d-log-m-and-hlg-in-videography/)).
So a DJI file in a 10-bit pixel format was not shot in Normal colour, whatever
its colour tags claim. That inference is gated on the encoder tag (a 10-bit
Rec.709 file from another camera is untouched), is reported as a guess with the
evidence in the message, and `--convert-log off` overrides it. If the clip is
actually HLG rather than D-Log, a D-Log LUT is the wrong conversion — so when
the profile is known to be HLG and a LUT is applied, the report says exactly
that instead of quietly grading it.

### The conversion has to prove itself

A LUT is applied on a claim: *this footage is flat, and the LUT makes it
normal.* The claim is now **measured**, per clip, before the export is written.

The same twelve frames are decoded twice - untouched, and through the LUT - and
compared. The conversion is refused if it:

| measurement | limit |
|---|---|
| mean saturation afterwards (when the conversion is what raised it) | > 0.55 |
| saturation multiplied by | > 1.45x |
| newly blown highlights | > 5 % of the frame |
| newly crushed blacks | > 4 % of the frame |
| luma span kept | < 90 % of what went in |

Those limits come from Tomas's own files and his own cube. At full strength it
took mean saturation **0.210 → 0.384** and pushed **12 % of every frame to
black** (0.005 → 0.122), while the luma span barely moved (0.628 → 0.653) -
a conversion that is not opening the picture up, only crushing it. That is what
"too much of everything" was.

**Refusing is not the only answer.** A cube file has no dial inside it, so
"convert, but less" means blending the converted image back over the original -
`split`/`lut3d`/`blend` in one filtergraph, in the analysis *and* in the export,
so what was measured is what gets written. `--lut-strength auto` (the default)
walks 1.0 → 0.75 → 0.5 → 0.25 and keeps the strongest rung that passes the
check; if even the weakest overshoots, no LUT is applied and the original is
kept. Measured on the same clip:

| strength | saturation | luma span | black pixels |
|---|---|---|---|
| 0.00 (untouched) | 0.208 | 0.628 | 0.5 % |
| 0.25 | 0.242 | 0.634 | 1.2 % |
| 0.50 | 0.278 | 0.640 | 2.9 % |
| 0.75 | 0.327 | 0.646 | 6.7 % |
| 1.00 | 0.384 | 0.653 | 12.2 % |

The whole ladder costs **one** extra decode: the blend is linear, so every rung
is computed from the two decoded sets in memory and only the chosen strength is
re-applied by ffmpeg. The run says which rung it landed on and why:

```
LUT pritaikytas tik 50 % stiprumu — pilnas buvo per stiprus šiai medžiagai
(sodrumas 0.208 → 0.278). Stiprumą galima nurodyti ranka: --lut-strength 0..1.

LUT ATŠAUKTAS šiam failui: konvertavimas užgniaužė šešėlius (sodrumas 0.208 →
0.451). Originalas jau atrodo teisingai, todėl paliekamas nekeistas.
```

`--convert-log on` and `--lut-all` are you overruling the measurement: the LUT
is then applied whatever the check says, and the check's objection is printed
instead of acted on. `--lut-strength 0.4` fixes a number and skips the ladder.

**Measured flatness stays evidence, not a trigger.** Across those 163 files the
luma span ran 0.155–0.925 and mean saturation 0.074–0.765 in one continuous
distribution with no gap: the measurement was reading the *scene*, not the
profile. Four dark sunsets fell under the thresholds and had their saturation
multiplied by 2.5, which is what wrecked two named exports. It is still
measured and still reported, and it no longer changes a pixel.

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

### The `.LRF` proxy — and what else is in it

DJI writes a `NAME.LRF` next to every take. Opened and measured, one of
Tomas's:

| Stream | What it is |
|---|---|
| 0 | H.264 **1280x720**, 8.03 Mbit/s, 29.97 fps, 20.354 s, `yuv420p`, tagged bt709 |
| 1 | data, codec tag `djmd`, 75.7 kbit/s — a binary `dvtm_Lito_X1.proto` protobuf: 1831 records over 610 frames, carrying the serial number, firmware strings and `DJI FC9589`. **No plain text**, so GPS and gimbal values would need DJI's undocumented schema; `color_md` is *not* in it |
| 2 | MJPEG 960x540, `attached_pic` — a cover thumbnail |
| format tags | `encoder: DJI Lito X1`, `creation_time` |

So: a full-length 720p proxy of the same take, a thumbnail, and telemetry that
is not readable without DJI's proto file. The proxy is the useful part. Analysis
runs at 640 px anyway, so **`--proxy auto` (the default) analyses the `.LRF` and
exports from the master.** Before it is used it is probed and checked against
the master — duration within 0.5 s or 2 %, the same aspect ratio, and a long
edge no smaller than the analysis size — and every acceptance and every refusal
is named in the report:

```
Analizei naudojamas gretimas peržiūros failas DJI_0243_D.LRF (1280x720).
Eksportuojami kadrai visada imami iš originalo.
Peržiūros failas DJI_0244_D.LRF nenaudojamas: duration differs by 3.100 s (max 0.500 s).
```

What this costs in accuracy is not measured on Tomas's footage yet: the proxy
is 8 Mbit/s of 720p, so fine detail is softer than the master's, and sharpness
is a *within-clip* rank, which softening compresses. If two runs of the same
folder disagree about which frames are best, `--proxy off` is the comparison.

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

**Order matters.** The LUT is a *conversion* and goes first, inside the ffmpeg
call that extracts the frame; the look is a *taste decision* and goes last, on
the already-corrected image. The no-LUT normalisation only ever runs when there
is no LUT, because both do the same job.

#### `--look auto`

Eight frames spread across the clip are measured and the look is chosen from
what is in them — once per clip, never per frame, because a look that changed
from shot to shot inside one take would be worse than none.

What is measured (all shares of the frame, `features.scene_signature`):
vegetation (green hues), sky and water (blue hues), warmth (red through
orange), grey (unsaturated mid-tones), and the share of the straight lines in
the frame that stand up.

```
nature = colour / (colour + grey)                colour = vegetation + sky + warm
city   = grey   / (colour + grey) * structure    structure = 0.4 + 0.6 * vertical_share
```

Both sides are **shares of each other**, never compared with a fixed
threshold. That is deliberate: a D-Log frame is desaturated everywhere, so
counting grey pixels against a constant labels every ungraded sunset a city —
the same scale mistake that once multiplied a dark sunset's saturation by 2.5.

If the two scores are within 0.08 of each other, **no look is applied** and the
report says the evidence was too close to call. An undecided answer is a real
answer.

```
Automatiškai parinktas profilis „gamta“ (gamta 0.71 / miestas 0.12, išmatuota 8 kadr.).
Automatinis profilis nenustatytas: gamta 0.34 ir miestas 0.29 skiriasi mažiau nei 0.08.
```

One honest limitation, measured on Tomas's own frames: on aerial footage the
vertical-line evidence is almost always near zero (0.00–0.03 on a village shot
from above, because buildings foreshorten into roof lines when you look down at
them). So in practice the decision is carried by colour mass, and the printed
numbers show it. A village at dusk with a warm sky comes out `nature`.

### Confidence

After ranking, the spread of the top 20 scores is measured. If they fall inside
a narrow band (starting threshold: range < 0.08 on a 0–1 scale), the console and
the report say the ranking carries little information and the frames are
effectively equivalent — choose visually. A flat distribution is never presented
in the same confident format as a real one.

### Selection

Two modes. **Threshold is the default, and nothing caps it**: every frame
scoring at least `--min-score` is taken. A dull ten-minute clip can return one
frame, or none; a strong one can return fifteen.

There is no per-clip cap by default (`--max-per-clip 0`) on purpose. A cap and
a threshold answer two different questions, and only the threshold is the right
one here: "how many do I want from this clip" is unknowable before looking,
while "is this frame good enough" is the same question for every clip. A cap on
top of a threshold discards frames that already passed the bar. Set
`--max-per-clip N` if a runaway clip ever needs one — the run then says the cap
decided, not the quality.

Two things still limit how many frames a clip can produce, and both are about
not exporting the same picture twice rather than about counting: the minimum
gap (`--min-gap`, or 3 % of the clip, whichever is larger) and the duplicate
test below.

`--min-score 0.60` is **one person's judgement on one card, not a measured
constant**: Tomas ran 163 of his own files at 0.60 and kept the result, which
is the only kind of evidence a number like this can have. Every run prints it
with that caveat attached. When no frame in a clip reaches it, the tool says
so, names the best score it did see, and tells you to lower the bar; it does
not quietly write an empty folder.

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

## Calibration — making the weights yours

The weights above are chosen numbers. `framepicker/learn.py` is the only path
from your own picks to different ones:

```bash
# 1. run a batch, then copy the frames you would actually use into one folder
python -m framepicker.learn "OUT/run-20260828-163550/results.json" --picks "D:/keepers"
```

It matches your kept frames against the run (by filename without extension, so
re-saving a JPEG as PNG or moving it does not break the link), then for each
score component reports the mean among the kept frames, the mean among the
discarded ones, and Cohen's *d* between them. Then it proposes weights in
proportion to how well each component separates your picks — **and checks its
own proposal** by re-ranking every frame with them and reporting how many of
your picks would land in the top 20, before and after.

Three things it will not do:

* Nothing is proposed below 20 kept and 20 discarded frames. A weight vector
  from six examples is noise with a decimal point, and it says so instead.
* No component is ever deleted or allowed to take over: every weight stays
  inside 0.05–0.60, whatever the numbers say.
* **The weights are never applied automatically.** They are printed, with the
  line telling you to edit `WEIGHTS` in `framepicker/scoring.py` yourself. A bad
  evening of picking must not be able to quietly rewrite the tool's judgement.

If the proposal does not rank your picks any higher, the output says that too:
that means the components do not explain your taste, and the honest conclusion
is to leave the weights alone.

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
| Settings | defaults (`--select threshold --min-score 0.60 --min-gap 2`), faces on, saliency on |
| Result | **4.7 s per minute of footage** |

This says nothing about the target RTX 2060 machine, about hardware decode, or
about real camera footage. Synthetic `testsrc2` is not representative content —
it is uniform enough that the confidence check correctly reported the ranking as
uninformative. Measure on real clips before quoting anything.

### Where the time actually goes

Measured on one real DJI file (`DJI_20260822191538_0243_D.LRF`, 1280x720 H.264,
8.0 Mbit/s, 29.97 fps, 20.35 s) on the same 4-core container, CPU decode,
analysis at 640 px:

| Step | Cost |
|---|---|
| decode + scale, `--fps 2`, every frame decoded | 1.19 s (17x realtime) |
| decode + scale, `--fps 2`, `--keyframes` | **0.40 s (51x realtime)** |

And the same measurement against a 4K master of the same content (3840x2160,
10-bit, 130 Mbit/s, 60 fps - the resolution, bit depth and bitrate of Tomas's
own files, transcoded to H.264 because this container has no GPU). Normalised
to **seconds of work per second of footage**, so the two lengths are
comparable:

| Analysed from | Mode | s per s of footage | vs 4K, every frame |
|---|---|---|---|
| 4K 10-bit master | every frame | 0.418 | 1.0x |
| 4K 10-bit master | `--keyframes` | 0.077 | **5.4x faster** |
| 720p `.LRF` proxy | every frame | 0.061 | **6.9x faster** |
| 720p `.LRF` proxy | `--keyframes` | 0.022 | **19x faster** |

Read those as ratios, not as promises: this is CPU H.264 decode on a 4-core
container, while Tomas's run used CUDA on HEVC 10-bit. The ordering is the
point - the proxy is worth roughly as much as keyframe-only decoding, and the
two combine.

Per analysis frame, all features together cost about 52 ms:

| Feature | ms/frame | Feature | ms/frame |
|---|---|---|---|
| saliency subject | 11.5 | face detect (YuNet) | 9.2 |
| horizon tilt | 10.7 | sharpness | 8.1 |
| colour histogram | 4.3 | colourfulness | 3.3 |
| symmetry | 2.3 | pattern repetition | 2.2 |
| dynamic range | 2.3 | motion | 1.2 |
| dHash | 1.1 | exposure clipping | 0.6 |

So on Tomas's 163-file run (3840x2160, 59.94 fps, HEVC 10-bit, 5077 s of
footage, 4063 s wall clock with 4 files in parallel), the features were not the
bottleneck: a 137 s clip evaluated 246 candidates, which is about 13 s of
feature work out of 527 s of elapsed time. The rest was **decoding 8200 frames
of 4K 10-bit HEVC in order to keep 274 of them.** That is what the three new
switches attack:

* `--proxy auto` (default) — read the 720p `.LRF` instead of the 4K master.
* GPU scaling — `-hwaccel cuda` alone decodes on the GPU and then copies every
  full-size frame back to system memory. With `scale_cuda`/`scale_npp` the
  frames are dropped and shrunk *before* the copy. The chain that will be used
  is tested with a single-frame decode first, exactly like `-hwaccel` itself,
  and falls back silently to the plain path if the build cannot do it — so this
  cannot make a working run stop working.
* `--keyframes` — decode intra frames only. Measured 3.0x faster above, and on
  that file it yielded 40 frames where the full decode yielded 41, because DJI
  writes a keyframe roughly every half second. The report always states how
  many frames were actually sampled and at what effective rate.

One number to watch in the per-clip report: `elapsed_s` is wall-clock while up
to four files are processed at once, so a 137 s clip showing 527 s is not 3.8x
realtime per stream - divide by `--jobs`.

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
  grading.py          looks as targets, and the --look auto decision
  sidecar.py          the .SRT and .LRF files DJI writes next to the video
  learn.py            calibration: your own picks -> proposed weights
  keepawake.py        keeps the machine awake while a run is going, screen free
  report.py           results.json + report.html + the link check
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
* `test_every_option_has_a_control_or_a_stated_reason` — every field of
  `Options` is reachable from the window, or listed as deliberately not a
  control. The command line and the window cannot drift apart.
* `test_a_run_writes_both_logs_and_the_report_links_to_them` — the log files
  exist, carry `run_started`/`clip`/`frame`/`run_finished` records with real
  numbers, are linked from the page, and are not reported as leftovers.
* `test_a_log_that_cannot_be_written_does_not_raise` — a log failure is
  recorded, never fatal.
* `test_closing_the_window_is_never_blocked`.
* `test_cancelling_stops_and_leaves_the_window_ready_for_new_files`.
* `test_symmetry_is_measured_against_the_frames_own_contrast` and
  `test_repetition_finds_stripes_and_ignores_gradients` — the two new graphic
  measurements cannot be fooled by a flat frame or a sky gradient.
* `test_a_lut_that_overshoots_is_refused_and_the_original_kept` and
  `test_a_lut_that_is_only_a_little_strong_is_softened_not_dropped` — a
  conversion that makes the picture worse is measured, softened, or dropped;
  forcing it keeps it *and* prints the objection.
* `test_footage_that_arrives_colourful_is_not_blamed_on_the_conversion` — the
  saturation ceiling is about what the conversion added, not what the source
  already was.
* `test_a_handful_of_picks_proposes_nothing` and
  `test_no_weight_is_ever_deleted_or_allowed_to_take_over` — calibration
  refuses to be confident.

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

1. **Per-clip count: quality threshold, and no cap.** `--select threshold` is
   the default: every frame above `--min-score` is taken, none from a weak clip
   and fifteen from a strong one, with no upper bound (`--max-per-clip 0`).
   The old fixed target is still there as `--select count`. The threshold
   itself is one person's judgement and everything that prints it says so.
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
  starting values. `framepicker/learn.py` is now the path to calibrating them,
  but it needs at least 20 frames Tomas actually kept; until those exist the
  numbers stay chosen rather than measured.
* What the `.LRF` proxy costs in ranking accuracy on his own footage is
  untested. The comparison is one run with `--proxy off`.
* The `djmd` telemetry stream in the `.LRF` (GPS, gimbal angles, altitude) is a
  binary `dvtm_Lito_X1.proto` protobuf with no public schema. If DJI's
  `color_md` is worth having per frame rather than per clip, the `.SRT` route
  gives it without reverse-engineering anything.
* On aerial footage the vertical-line evidence in `--look auto` is nearly always
  zero, so the nature/city decision is effectively colour-based. If that turns
  out to pick wrong on urban footage, the fix is a better structure measurement,
  not a bigger threshold.
* `--min-score 0.60` now has exactly one piece of evidence behind it: a
  163-file run of Tomas's own footage whose result he kept. That is a judgement
  on one card in one kind of light, not a measured constant — a different
  camera, or a mixed batch, may want a different number. `framepicker.learn` is
  the way to find out rather than guess.
