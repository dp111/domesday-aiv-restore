# Domesday model decoder (newdecode)

Decodes PAL composite video with the Domesday model decoder, `newdecode`
(github.com/dp111/domesday-aiv-restore), and writes a 928x576 `yuv444p16`
YUV4MPEG2 file. newdecode is a Python/PyTorch decoder: it removes the disc's
channel smear and FM-beat herringbone, fits a PAL model to still pictures and
held runs, and (with **video**) solves interlaced video too, falling back to
ld-chroma-decoder's transform decoders where the model does not apply.

## When to use

At the end of a PAL LaserDisc pipeline (TBC Source, stacker, dropout
correction, ...) when you want newdecode's picture instead of the built-in
Video Sink decoders. It needs a working newdecode install: a Python with
numpy, scipy and CUDA PyTorch, and `ld-chroma-decoder` on `PATH`.

## What it does

When triggered it:

1. writes every input frame into a work directory as an ld-decode `.tbc`
   (16-bit, two 313x1135 fields per frame) with a `.tbc.json` (video
   parameters, field order, measured `fieldPhaseID`, and the pipeline's
   dropout hints as `dropOuts`) and `in.held.json` (the frames' held-picture
   hints, for future use);
2. runs `python [launcher] newdecode.py in.tbc OUTPUT --scratch DIR --video
   0|1 --dropout_mask 0|1 [extra args]`, reporting progress from its
   `[newdecode] frame i/N` lines and logging all of its output to
   `newdecode.log` in the work directory;
3. deletes the work directory on success (unless **keep_work**); on failure
   it is kept and the status names the log.

The work directory is `<scratch_dir or the output's folder>/<output name>.ndwork`
and needs about three times the exported TBC (1.42 MB per frame): the TBC, and
newdecode's de-smeared copy plus its scratch.

Samples are widened from the pipeline's 10-bit CVBS to 16 bit by x64, and the
metadata declares the spec levels (black 16384, white 54016), so the decode
uses the same levels the pipeline normalised to.

## Parameters

- **output_path** — the `.y4m` to write. Required.
- **python** — interpreter that has newdecode's dependencies. Default `python3`.
- **newdecode_path** — path to `newdecode/newdecode.py`. Required.
- **wrapper_path** — optional launcher run as `python LAUNCHER newdecode.py ...`
  (e.g. `nd_gpuwrap.py`, which works around cuFFT out-of-memory errors).
- **scratch_dir** — parent of the work directory. Default: beside the output.
- **video** — `--video 1`: model-solve interlaced video. Default on.
- **dropout_mask** — `--dropout_mask 1`: dropouts get zero data weight and are
  filled by the solver. For single-copy sources. Default off.
- **extra_args** — more newdecode arguments, shell-quoted (e.g. `--luma_lpf 0`).
- **keep_work** — keep the work directory after a successful decode.

## Limitations

PAL composite only. The whole input is exported before decoding starts, and
newdecode runs as a separate process (GPU recommended: roughly 1-2 s per
frame). Cancelling kills newdecode and its helper processes.
