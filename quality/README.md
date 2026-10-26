# Quality measurement: benchmark and per-frame record

Goal: an accurate, repeatable way to score every frame, so stacker and decoder changes can be judged by
numbers rather than by eye.

## 1. Synthetic benchmark (most accurate: the true picture is known)

Real decoded frames are taken as the truth and encoded to a clean PAL composite with ld-chroma-encoder.
For each real capture of the side a synthetic capture is made with that capture's measured noise:

    capture = shift(clean + master_noise, dx_master) + capture_noise + real dropouts

* `measure_noise.py` measures the noise model from real per-capture TBC segments (blank VBI lines,
  fields matched by CAV picture number): per-capture noise spectrum, the part shared by captures of one
  master (pressed into the disc) and the independent part (player).  South: shared noise is 65 % of the
  noise at 0-2 MHz and 90-96 % above 3.5 MHz; cross-master coherence ~0.
* `make_bench.py` builds the benchmark (segments of 128 frames, CAV picture numbers copied from the real
  stack so the stacker aligns them like real captures; per-master timing offsets as measured: captures of
  one master agree to +-0.03 samples, South masters differ by -0.3 and -1.3 samples).
* `run_bench.sh <bench> <label> [caps]` runs the production chain (stack.py -> ld-dropout-correct ->
  newdecode) with overridable versions (`STK`, `STACK_EXTRA`, `ND_CD`, `ND_EXTRA`, `ND_CD=chroma:transform3d`
  for the upstream decoder, `NOGPU=1`, `SEGS=03000`).  Pass capture names for reduced / single-copy runs.
* `score.py <bench> <label> [<label2>]` scores against the truth, per frame and per field:
  * composite (stack.tbc, doc.tbc vs clean.tbc): per-band signal-to-error (dB) and error rms (16-bit units),
    0-2 / 2-3.5 / 3.5-5.5 (chroma) / 5.5-8.9 MHz -> judges the stacker and dropout correction alone
  * picture (out.yuv vs truth.yuv): PSNR Y/Cb/Cr, SSIM, luma and chroma per-band SNR and **gain** (detail
    retention, 1 = kept, < 1 = softened, so a denoiser cannot win by blurring) -> judges the whole chain
  * the sub-pixel shift to the truth is measured per field / frame and removed; its spread is reported
    as jitter
  * with two labels, prints the change per metric and per segment

The CommunitySouth benchmark used in development: 5 segments x 128 frames (3000 Test Card F / bars, 8000,
16256, 29696 photos, 50112) + `make_patterns.sh` test patterns, noise model measured from the 12 captures.

Limits: Gaussian noise of the measured spectrum, level-independent; no player spur, no timebase errors,
no undetected disc defects; dropouts are real lists but their positions are independent between captures
(real master-level defects are shared).  Good for comparing denoising, decoding and combining; check
TBC / defect handling on real captures too.

## 2. Whole-side record (every field of a real stack, no reference)

`fq_side.py <side.tbc> <out.csv> [--vits]`: one read of the stacked TBC; per field the blank-line noise per
band, residual dropout samples and (with --vits) ld-process-vits white SNR / black PSNR.  Content
independent, so frames and versions compare directly; flags weak stretches of a side.  The cross-master
blend works on the active picture only, so these lines show the primary-master stack.

## 3. Per-frame quality database

`quality_db.py` keeps one SQLite database (`quality.db` here, or `FQ_DB`) with a run per side and pipeline
version: `runs`, `stack_frames` (stacker LDSTACK_FRAME_STATS: captures available/used, alt-master use, pixels
filled/repaired, residual dropouts), `fields` (`fq_side.py`), `nd_frames` (newdecode saturation guard and frame
class) and `pvf_frames` (`pvf_quality.sh`: PSNR/SSIM of the PVF encode against its master).  The view
`frame_quality` joins them per frame.  `pipeline/side.sh` fills it at the end of each side;
`quality_db.py summary` prints one line per run.

## 4. Dropout-fill test (`dofill/`)

`prep.py` injects real dropout shapes into a benchmark stack, `run.sh` decodes the fill variants, `score_do.py`
scores the error inside the dropouts and `temporal.py` tries the previous/next-frame fill offline.  Results
(`RESULTS.txt`) chose newdecode's defaults PALDEC_DOI=replace and PALDEC_DOT=1.5 (`newdecode/README_dropout.txt`).

## Findings so far
* stack.py wrote each line's every-capture dropouts as ONE span from the first to the last bad sample, so
  two bad pixels flagged the good samples between them for concealment.  Fixed 2026-10-24 (runs, gaps <= 8
  merged; LDSTACK_RESID_DO_GAP).  No change where bad samples are contiguous (South Test Card, 12 captures);
  matters for sides with few captures (EcoDisc_S2: 45 M samples flagged, 636 per field vs South 8.8).
