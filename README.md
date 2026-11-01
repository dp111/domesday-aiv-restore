# BBC Domesday / AIV LaserDisc restoration

The tools and pipeline used in 2026 to rebuild the video of the BBC Domesday (1986) and the other
LaserVision ROM (AIV) discs from multiple RF captures of each disc side. The output feeds the Pi1MHz
player for the BBC Micro:

- a lossless master per side: FFV1, 928×576, 10-bit 4:2:2;
- a `video.pvf` per side for the Pi1MHz: H.264, 832×576 on the BBC MODE 0 pixel grid, under 4 GB.

The RF decoder and the multi-capture stacker live in the ld-decode fork
[dp111/ld-decode](https://github.com/dp111/ld-decode), branch `domesday-aiv` (published together with
this repo); that branch is the ld-decode the pipeline expects in `LD`. This repo holds
everything after the stack, plus the scripts that run the whole thing.

## Pipeline, per side

| Step | Tool | What it does |
|---|---|---|
| 1. Capture decode | `ld-decode --vits_white_level [--V4300D_coherent_subtract]` | One TBC per capture. Captures from an LD-V4300D with its ~8.47 MHz spur (`pipeline/probe_v4300_spur.py`) get the coherent spur subtraction. |
| 2. Stack | `python -m lddecode.stack --cross_master` (fork) | Aligns all captures by CAV picture number (checked against the picture content: a capture that labels every picture one off is renumbered) and registers them sub-pixel, with a robust gain/offset fit that ignores unflagged dropout tails. Each capture first has the disc's FM-beat herringbone removed with newdecode's canceller, its carrier calibrated per master (`LDSTACK_DEHERRING_LIB`, set by `config.sh` to `newdecode/`). It groups captures by **glass master**, using the 5.5–7 MHz noise moulded into each pressing, and averages each master's captures. It then blends the master pictures per frequency band, weighted by each master's error variance (measured on the picture), equalises each alternate master's horizontal response to the primary's, and repairs defects present on every copy of one master from the other masters. Defects that remain in every capture stay flagged but carry the vertical interpolation of the good lines around them. A run of up to 250 CAV pictures missing from every capture is padded with repeats of the last picture, so frame numbers stay in step with the disc (longer runs are logged). Writes `<side>.xm4.tbc` (the name is historical). |
| 3. Dropout correct | `ld-dropout-correct` | Line-copy concealment of residual dropouts. This is only the starting point for step 4. |
| 4. Decode | `newdecode/newdecode.py` (GPU) | Model-based PAL decoder (see below). Run in 5120-frame chunks by `pipeline/nd_chunked.sh`, so a crash costs one chunk. |
| 5. Master | ffmpeg | FFV1 yuv422p10le, co-sited chroma, with the stack's squelched analogue audio as FLAC: `<side>.v8.mkv` (`v8` is the historical pipeline label). |
| 6. PVF | `newdecode/stillweave.py` → `pvf/make_pvfv2.py` → `pvf/verify_pvf.py` | Weaves each frame whose two fields are one instant (stills, 25p film) and deinterlaces only true video. Encodes H.264 at CRF 17, stepping up to 18, 19, 20 while the result would reach 4 GB. Identical frames are stored once. |
| 7. Quality record | `quality/` | Records every frame of the side in `quality.db` (SQLite). |

### newdecode

newdecode starts from ld-chroma-decoder: transform2d for stills, transform3d for video. It then refines
each frame by fitting a model of the PAL composite signal (`palmodel.py`, solver `optcore.py`) to the TBC
samples on the GPU.

- **Stills:** frames whose neighbours show the same picture are solved together ("held" runs). Each block
  of a held frame is tested against its run's median and an outlier block (a changed or damaged one) is
  left out of the average, so saturated and edge blocks keep the averaged, low-noise value.
- **Still or interlaced:** decided per frame from the comb score and the frame-to-frame match, so static
  saturated colour, menus and text pages are solved and averaged as stills, not as video. The decision is
  written to a class log (`ZIP_DUMP`), which `stillweave.py --classes` uses to weave exactly those frames.
- **Film and local motion:** a run of two different pictures is split into stills; the temporal prior is
  weakened where a cursor or highlight changes; a still background with a moving inset (talking heads) is
  solved per field and its background then averaged.
- **Moving video:** solved per field, with motion-compensated temporal priors (`--video 1`).
  A video solve whose luma leans toward its neighbouring frames (ghosting) falls back to transform3d.
- **Before decoding:** the channel's ~0.4 µs edge smear is removed (`deghost.py`), and black and white
  levels are taken from each TBC's own `videoParameters`.
- **Noise model:** the noise spectrum that weights the fit is measured on the active picture after
  deghosting, guarded by a robust blank-line estimate (blank lines alone overstated it on multi-master
  stacks). The temporal estimate is floored by a spatial one, and the
  video and film temporal priors are capped so a higher estimate does not mean more ghosting.
- **Herringbone:** the FM-beat herringbone (a tone at carrier minus subcarrier, per glass master) is
  predicted from each field and subtracted where the picture is flat enough to measure it
  (`deherring.py`).
- **Dropouts (`--dropout_mask 1`):** flagged dropouts carry no data weight in the solve. They are then
  filled from the clean rows above and below, and from the previous and next frames where the picture
  is static. See `newdecode/README_dropout.txt` for the measurements.
- **Fallback chroma scale:** frames output by transform2d/3d (interlaced frames, guard fallbacks) are put on
  the model's chroma scale (U x0.923, V x0.913; `PALDEC_T3_CGAIN`), since the deghost inverse lifts
  4.43 MHz by ~1.18 and those frames were ~9% more saturated than the solved frames around them.
- **Luma equaliser (`lumaeq.py`):** the deghost inverse also lifts luma above ~1 MHz while the real
  record/playback chain is flatter than its smear model. The chain's response is measured from the disc's
  own multiburst VITS (Domesday: field 0 line 20), over frames spread across the whole TBC so chunks agree,
  and the decoded luma is divided by it: never a boost, floor 0.75, off when no multiburst is found.
  On Community South that is x0.95 / 0.93 / 0.89 / 0.87 at 1 / 2 / 3 / 4 MHz. `PALDEC_LUMA_EQ=0` turns it off.

`newdecode/model_params.json` holds the carrier calibration. `vidsplice.py` re-solves video frames of an
existing master offline.

## Layout

```
config.sh                  machine paths (defaults; override in config.local.sh)
config.local.example.sh    the values used on the restoration machine
pipeline/
  sides.conf               the side table: name, Pi1MHz slot, CAV/CLV, capture glob
  run.sh                   start everything: decodeq.sh + one side.sh per side
  decodeq.sh               capture-decode queue (4 at a time, to the drive with most room)
  side.sh                  one side: stack -> QC -> dropout-correct -> newdecode -> PVF -> quality record
  nd_chunked.sh            chunked, resumable newdecode -> master MKV
  nd_gpuwrap.py            runs newdecode with a cuFFT/GPU-memory workaround
  prio.sh, gpugate.sh      CPU / GPU / RAM scheduling between the stages (run beside run.sh)
  stack_quality.py         predicted stack noise from a stack log (QC gate)
  probe_v4300_spur.py      spur probe for new captures; results in spur_probe.txt
newdecode/                 the decoder, stillweave.py (PVF weave)
orc-plugin/                decode-orc sink stage that runs newdecode (PAL), see orc-plugin/README.md
pvf/                       PVF writer and verifier
quality/                   synthetic known-truth benchmark, per-field/per-frame measures, quality.db
```

## Requirements

- **ld-decode:** `dp111/ld-decode` branch `domesday-aiv`, built as its INSTALL.md describes. The Python
  decoder is used, plus the `ld-chroma-decoder` and `ld-dropout-correct` tools (and `ld-chroma-encoder`
  for the benchmark).
- **Python:**
  - an environment with ld-decode's requirements, for everything except newdecode (`PY`);
  - an environment with PyTorch + CUDA, for newdecode (`MLPY`). It was developed on a 12 GB NVIDIA card.
- **ffmpeg / ffprobe:** with FFV1 and libx264.
- **ld-process-vits** (ld-decode tools, for the quality record's VITS columns), the **sqlite3** CLI (for
  querying `quality.db`), and util-linux **flock**, **setsid** and procps **pgrep** / **ps** (scheduling,
  GPU lock, process checks).
- **Memory:** about 48 GB RAM. A newdecode chunk peaks at about 32 GB, and a 12-capture stack also needs
  many GB. `gpugate.sh` keeps the two apart.
- **Disk:** about 77 GB per CAV capture decode, the same again for the stack, and about 25–35 GB per master.

## Running

```
cp config.local.example.sh config.local.sh     # edit the paths
. ./config.sh                                  # sets PY, LD, WORK, ... in this shell
LD=$LD "$PY" pipeline/probe_v4300_spur.py <captures...> >> pipeline/spur_probe.txt   # new captures only
setsid nohup pipeline/prio.sh    >/dev/null 2>&1 &
setsid nohup pipeline/gpugate.sh >/dev/null 2>&1 &
pipeline/run.sh
tail -f "$WORK/pipeline.log"                   # $WORK as config.sh set it (default ~/aiv-work)
```

The UN discs' captures have spaces in their names; `pipeline/sides.conf` expects space-free symlinks
`UN_Disc1_CAV_PAL.raw.oga` ... `UN_Disc7_CAV_PAL.raw.oga` in `$CAP_UN` pointing at them.

A single side runs with `pipeline/side.sh <side>` once its captures are recorded in
`$WORK/state/caps/<side>/`.  A **capture record** is two files per capture: `<id>.path`, holding the TBC
path without `.tbc` (so `<that>.tbc`, `.tbc.json`, `.pcm` must exist), and an empty `<id>.done`
(`<id>.failed` = decode failed, left out).  `decodeq.sh` writes them for each `.ldf` the side table's glob
matches; the id is `dsN` from the capture's `DD86-DSN` name.  **Ready-made TBCs:** write the records by hand
(any id, e.g. `cap1.path` + `cap1.done`); when the side's glob matches no capture file, `side.sh` takes
every `<id>.path` record of the side instead.

**Own side table (`V7_TEST`):** `conf.sh` finally sources the file named in `V7_TEST`, which can replace
the side table (`V7_SIDES`) and any other setting, e.g. for a test side or ready-made TBCs.  Example:
`pipeline/test_sides.example`, used as `V7_TEST=$PWD/my_sides pipeline/side.sh MyTest`.

Every stage is resumable. Control files in `$WORK/state`:

| File | Effect |
|---|---|
| `PAUSE_DECODE` | stops new capture decodes |
| `PAUSE_stack`, `PAUSE_gpu`, `PAUSE_pvf` | hold that slot (manual; `gpugate.sh` uses its own `PAUSE_gpu.auto` / `PAUSE_stack.auto`, so it never clears a manual pause) |
| `wait/<slot>.<order>` | a side queued for a slot ("<side> <pid>"); an entry whose side died is dropped, an **empty** file is a manual hold of that slot for every later side |
| `gpu.lock` | the `ND_GPULOCK` flock (default here; set `ND_GPULOCK` to one shared path, e.g. `/run/lock/aiv-gpu.lock`, so pipelines with separate `WORK` dirs on one machine never run two newdecodes at once) |
| `ND_CD` | a decoder directory that overrides `ND_DIR` for every later newdecode run, also in side chains already running |
| `HOLD.<side>` | set by a failed QC gate; remove it to continue that side |
| `AUTODELETE` | deletes each side's capture decodes after its stack |

Per-frame quality of a finished side:

```
python3 quality/quality_db.py summary
# the database is quality/quality.db unless FQ_DB names another file
sqlite3 quality/quality.db "select frame, used, resid_do, bl_rms, pvf_psnr_y from frame_quality where side='CommunitySouth' order by pvf_psnr_y limit 20"
```

## Findings that shaped this

- **Disc noise is per pressing.** Most of the noise above 3.5 MHz is moulded into the disc: 90–96% of it
  is shared by every copy pressed from one glass master. More copies of the same master stop helping.
  Copies from another master carry independent noise, which is why the stacker blends masters.
  Community South has 12 captures from 3 masters (9 + 2 + 1).
- **Black level.** Real TBCs declare black at about 16030–16110, not 16384. newdecode takes the levels
  from each TBC; fixed levels crushed black by about 2 levels.
- **Dropouts.**
  - In the picture domain, interpolating from the rows above and below beats ld-dropout-correct's line copy.
  - Taking pixels from the neighbouring frames helps only where the picture is static. Each frame of a
    CAV still sequence is a different photograph.
  - Interpolating in the composite domain is worse than both.
  - Measurements: `quality/dofill/RESULTS.txt`.
- **Pi1MHz grid.** 832×576 is the BBC MODE 0 grid. The derivation is in `pvf/make_pvfv2.py`.

## Licence

GPL-3.0 (see LICENSE), the same as ld-decode.
