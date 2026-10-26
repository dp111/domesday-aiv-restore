# BBC Domesday / AIV LaserDisc restoration

The tools and pipeline used in 2026 to rebuild the video of the BBC Domesday (1986) and the other
LaserVision ROM (AIV) discs from multiple RF captures of each disc side. The output feeds the Pi1MHz
player for the BBC Micro:

- a lossless master per side: FFV1, 928×576, 10-bit 4:2:2;
- a `video.pvf` per side for the Pi1MHz: H.264, 832×576 on the BBC MODE 0 pixel grid, under 4 GB.

The RF decoder and the multi-capture stacker live in the ld-decode fork
[dp111/ld-decode](https://github.com/dp111/ld-decode), branch `domesday-aiv`. This repo holds
everything after the stack, plus the scripts that run the whole thing.

## Pipeline, per side

| Step | Tool | What it does |
|---|---|---|
| 1. Capture decode | `ld-decode --vits_white_level [--V4300D_coherent_subtract]` | One TBC per capture. Captures from an LD-V4300D with its ~8.47 MHz spur (`pipeline/probe_v4300_spur.py`) get the coherent spur subtraction. |
| 2. Stack | `python -m lddecode.stack --cross_master` (fork) | Aligns all captures by CAV picture number and registers them sub-pixel. It groups captures by **glass master**, using the 5.5–7 MHz noise moulded into each pressing, and averages each master's captures. It then blends the master pictures per frequency band, weighted by each master's error variance. Defects present on every copy of one master are repaired from the other masters. |
| 3. Dropout correct | `ld-dropout-correct` | Line-copy concealment of residual dropouts. This is only the starting point for step 4. |
| 4. Decode | `newdecode/newdecode.py` (GPU) | Model-based PAL decoder (see below). Run in 5120-frame chunks by `pipeline/nd_chunked.sh`, so a crash costs one chunk. |
| 5. Master | ffmpeg | FFV1 yuv422p10le, co-sited chroma, with the stack's squelched analogue audio as FLAC. |
| 6. PVF | `newdecode/stillweave.py` → `pvf/make_pvfv2.py` → `pvf/verify_pvf.py` | Weaves each frame whose two fields are one instant (stills, 25p film) and deinterlaces only true video. Encodes H.264 at CRF 17 (18 if the result would reach 4 GB). Identical frames are stored once. |
| 7. Quality record | `quality/` | Records every frame of the side in `quality.db` (SQLite). |

### newdecode

newdecode starts from ld-chroma-decoder: transform2d for stills, transform3d for video. It then refines
each frame by fitting a model of the PAL composite signal (`palmodel.py`, solver `optcore.py`) to the TBC
samples on the GPU.

- **Stills:** frames whose neighbours show the same picture are solved together ("held" runs).
- **Moving video:** solved per field, with motion-compensated temporal priors (`--video 1`).
- **Before decoding:** the channel's ~0.4 µs edge smear is removed (`deghost.py`), and black and white
  levels are taken from each TBC's own `videoParameters`.
- **Dropouts (`--dropout_mask 1`):** flagged dropouts carry no data weight in the solve. They are then
  filled from the clean rows above and below, and from the previous and next frames where the picture
  is static. See `newdecode/README_dropout.txt` for the measurements.

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
- **Memory:** about 48 GB RAM. A newdecode chunk peaks at about 32 GB, and a 12-capture stack also needs
  many GB. `gpugate.sh` keeps the two apart.
- **Disk:** about 77 GB per CAV capture decode, the same again for the stack, and about 25–35 GB per master.

## Running

```
cp config.local.example.sh config.local.sh     # edit the paths
pipeline/probe_v4300_spur.py <captures...> >> pipeline/spur_probe.txt   # new captures only
setsid nohup pipeline/prio.sh    >/dev/null 2>&1 &
setsid nohup pipeline/gpugate.sh >/dev/null 2>&1 &
pipeline/run.sh
tail -f $WORK/pipeline.log
```

A single side runs with `pipeline/side.sh <side>` once its captures are recorded in
`$WORK/state/caps/<side>/`.

Every stage is resumable. Control files in `$WORK/state`:

| File | Effect |
|---|---|
| `PAUSE_DECODE` | stops new capture decodes |
| `PAUSE_stack`, `PAUSE_gpu`, `PAUSE_pvf` | hold that slot |
| `HOLD.<side>` | set by a failed QC gate; remove it to continue that side |
| `AUTODELETE` | deletes each side's capture decodes after its stack |

Per-frame quality of a finished side:

```
python3 quality/quality_db.py summary
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
