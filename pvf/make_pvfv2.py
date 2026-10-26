#!/usr/bin/env python3
"""
make_pvfv2.py - build a .pvf (Pi Video File) for the Pi1MHz hardware player

Supersedes make_pvf.py. Same container, same firmware; what changes is the
picture geometry and that the file now records its own pixel aspect ratio.

WHY 832x576 AND NOT 768x576
---------------------------
The Pi1MHz composites two rasters that have to line up: the LaserDisc
picture and the BBC's own display, exactly as the VP415 and the Beeb did
through a genlock. So the sensible sampling grid for the video is the
Beeb's, and the Beeb's is set by its 16 MHz pixel clock:

    PAL active line                     52 us
    BBC MODE 0, 640 pixels at 16 MHz    40 us
    52 us at 16 MHz                     832 pixels     <- this file's width
    PAL frame, active                   576 lines

At 832 wide, one video pixel IS one MODE 0 pixel and one Beeb display line
is two video frame lines, so the two planes register at integer ratios with
no resampling of the Beeb's raster. At 768 the video is 832/768 = 8.3% short
of the Beeb's grid, which is what made the picture look horizontally
squashed against the computer's display.

832 also throws away less of the source. An ld-decode/ld-chroma-decoder PAL
frame is 4fsc: 4 x 4.43361875 MHz = 17.734475 MHz, so its 52 us active line
is about 922 samples. 922 -> 832 keeps 90% of that; 922 -> 768 keeps 83%.

Both constraints of the hardware decoder still hold: width a multiple of 32
(832 = 26 x 32) and height a multiple of 16 (576 = 36 x 16).

PIXEL ASPECT RATIO
------------------
832x576 is not square-pixel. A 4:3 picture over 832x576 has

    PAR = (4/3) / (832/576) = 12/13 = 0.923

which this tool computes and stores in the header, and also declares in the
H264 VUI (via setsar) so host players show it correctly. 768x576 files come
out with PAR 1/1, so nothing changes for them.

The header stays PVF version 1 on purpose: the firmware rejects any other
version, and it reads a fixed 64-byte header whose last three words were
always zero. The PAR goes in two of those, so new files still play on old
firmware (as square pixels) and old files still read as PAR unspecified.

    word 13  par_num       0 = unspecified, treat as square
    word 14  par_den

FILE LAYOUT (unchanged)
-----------------------
  - video: H264 Annex-B, ALL-INTRA (every frame an IDR with its own SPS/PPS)
    so any frame is randomly accessible with one decode, which is what gives
    the player LaserDisc goto/still/step/reverse for free.
  - audio: s16le stereo at --audio-rate, sliced per video frame.
  - a u32 file offset per frame, loaded into RAM by the player.

USAGE
-----
  ./make_pvfv2.py south.mkv video.pvf --crf 17
  ./make_pvfv2.py south.mkv video.pvf --deinterlace     # interlaced source
  ./make_pvfv2.py south.mkv video.pvf --width 768       # old geometry

Then copy video.pvf into the /BeebVFSn directory of the SD card, install the
full start.elf/fixup.dat and set gpu_mem=64 (see firmware/config.txt).
Check the result with verify_pvf.py before copying gigabytes.

Requires: python3, ffmpeg (with libx264) on PATH.
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

PVF_MAGIC = 0x31465650          # 'PVF1'
PVF_VERSION = 1                 # deliberately still 1 - see the notes above
AUDIO_RATE = 48000              # default; see --audio-rate
AUDIO_CHANNELS = 2
AUDIO_BYTES_PER_SAMPLE = 2

BEEB_PIXEL_CLOCK_HZ = 16_000_000
PAL_ACTIVE_LINE_US = 52
BEEB_GRID_WIDTH = 832           # 52 us at 16 MHz
DEFAULT_HEIGHT = 576
# Held-still dedupe thresholds, 8-bit levels.  Tuned on the South v2 PVF:
# frame-to-frame 16x16 luma block max mean|d| is 1-3 for repeats (noise),
# 5+ for any real change; the 3-4.5 gap holds only black frames with a
# dropout line.  DEDUPE_DC (whole-frame mean shift) keeps slow fades.
DEDUPE_Y = 3.0
DEDUPE_C = 3.0
DEDUPE_DC = 0.75


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pixel_aspect(width, height, display_aspect):
    """PAR that makes width x height display at display_aspect.

    PAR = display aspect / storage aspect. Returned in lowest terms, which
    is what the header and setsar both want."""
    return Fraction(display_aspect.numerator * height,
                    display_aspect.denominator * width)


def probe_source(path):
    """(width, height, sar) of the first video stream, via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,sample_aspect_ratio",
         "-of", "csv=p=0:nk=1", path],
        check=True, capture_output=True, text=True).stdout.strip()
    w, h, sar = out.split(",")[:3]
    try:
        n, d = sar.split(":")
        sar = Fraction(int(n), int(d))
    except ValueError:
        sar = None
    return int(w), int(h), sar


def active_crop(args, src_w, src_h, sar):
    """Return (crop_w, crop_x) removing ld-chroma-decoder's padding, or None.

    ld-chroma-decoder emits the TBC's active window (922 samples for a 52 us
    PAL line at 4fsc) padded on both sides to a multiple of --output-padding
    (default 8), so a PAL frame is 928 wide.  Scaling all 928 onto the 832
    pixels that stand for 52 us squeezes the picture by 6/922 = 0.65%, which
    is measurable against the Domesday alignment grid (picture 2500 on the
    Community discs).  The active width comes from the decode's .tbc.json
    (activeVideoEnd - activeVideoStart); failing that, from the SAR the
    chroma decoder declares, which is exactly the one that makes the active
    line 4:3, so active = display_aspect * height / SAR.  The padding is
    split evenly, so the crop offset is half the difference."""
    if args.no_crop:
        return None
    active = None
    json_path = args.tbc_json
    if json_path is None:
        stem = os.path.splitext(args.input)[0]
        for cand in (stem + ".tbc.json", stem + ".json"):
            if os.path.exists(cand):
                json_path = cand
                break
    if json_path:
        import json
        with open(json_path) as f:
            vp = json.load(f)["videoParameters"]
        active = int(vp["activeVideoEnd"]) - int(vp["activeVideoStart"])
        print(f"active window from {json_path}: {active} samples")
    elif sar:
        da = parse_aspect(args.display_aspect)
        active = round(da.numerator * src_h / (da.denominator * sar))
        print(f"active window from source SAR {sar}: {active} samples")
    if not active or active >= src_w:
        return None
    pad = src_w - active
    return active, pad // 2


def build_filter(args, par, crop=None):
    """The ffmpeg filter chain: optional deinterlace, crop, scale, declared SAR.

    --vf replaces only the scaling part, so the deinterlacer and the SAR
    still apply. (In make_pvf.py --vf replaced the whole chain, which made
    it easy to drop the scale by accident.)"""
    chain = []
    if args.deinterlace:
        chain.append("yadif=1")
    if crop:
        crop_w, crop_x = crop
        chain.append(f"crop={crop_w}:in_h:{crop_x}:0")
    if args.vf:
        chain.append(args.vf)
    else:
        chain.append(f"scale={args.width}:{args.height}:flags={args.scaler}")
    # Declare the pixel aspect on the encoded stream. The Pi player takes
    # its geometry from the header, but this keeps host playback honest.
    chain.append(f"setsar={par.numerator}/{par.denominator}")
    return ",".join(chain)


def encode_video(args, par, h264_path):
    src_w, src_h, sar = probe_source(args.probe_from or args.input)
    crop = active_crop(args, src_w, src_h, sar)
    if crop:
        print(f"source {src_w}x{src_h}: cropping {crop[1]} padding samples each "
              f"side to {crop[0]} before the scale to {args.width}")
    else:
        print(f"source {src_w}x{src_h}: no crop (scaling the full width)")
    x264 = (
        "keyint=1:min-keyint=1:scenecut=0:repeat-headers=1:"
        "bframes=0:ref=1:rc-lookahead=0:threads=auto:"
        # PAL is BT.601 (625-line) limited range; tag the VUI so the decoder
        # never has to guess the YUV->RGB matrix. Set here rather than with
        # ffmpeg's -color_* options, which lose primaries/transfer via libx264
        "colorprim=bt470bg:transfer=bt470bg:colormatrix=bt470bg:range=tv"
    )
    # -ss BEFORE -i: input seeking, so ffmpeg jumps rather than decoding
    # (and, over a network drive, reading) everything before the start
    cmd = ["ffmpeg", "-y"]
    if args.start:
        cmd += ["-ss", str(args.start)]
    cmd += ["-i", args.input]
    if args.frames:
        cmd += ["-frames:v", str(args.frames)]
    vf = build_filter(args, par, crop)
    if args.dedupe:
        # same filtered yuv420p frames to x264 and to a raw pipe, so the
        # duplicate test sees exactly the pictures that get encoded
        cmd += ["-filter_complex", f"[0:v]{vf},format=yuv420p,split[e][s]",
                "-map", "[e]"]
    else:
        cmd += ["-vf", vf]
    cmd += [
        "-an", "-sn",
        "-pix_fmt", "yuv420p",
        "-r", f"{args.fps_num}/{args.fps_den}",
        "-c:v", "libx264",
        "-preset", args.preset,
        "-crf", str(args.crf),
        "-profile:v", "high",
        "-x264-params", x264,
        "-f", "h264", h264_path,
    ]
    if not args.dedupe:
        run(cmd)
        return None
    cmd += ["-map", "[s]"]
    if args.frames:
        cmd += ["-frames:v", str(args.frames)]   # per output: the x264 one has its own
    cmd += ["-an", "-sn", "-r", f"{args.fps_num}/{args.fps_den}",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]
    print(">>", " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 24)
    dup = scan_duplicates(p.stdout, args)
    if p.wait():
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return dup


def scan_duplicates(pipe, args):
    """Held stills: dup[i] = the first frame of the run frame i repeats, or -1.

    A frame repeats the run's first frame when no 16x16 luma block differs
    by more than --dedupe-y mean |d| and no 8x8 U/V block by more than
    --dedupe-c (8-bit levels): noise-level differences only, so a moved
    cursor or a highlighted menu item starts a new run."""
    import numpy as np
    w, h = args.width, args.height
    fs = w * h * 3 // 2

    def blocks(d, b):
        hh, ww = d.shape
        return d[:hh // b * b, :ww // b * b].reshape(hh // b, b, ww // b, b).mean((1, 3)).max()

    dup, rep = [], None
    while True:
        buf = pipe.read(fs)
        if len(buf) < fs:
            break
        fr = np.frombuffer(buf, np.uint8)
        y = fr[:w * h].reshape(h, w).astype(np.int16)
        u = fr[w * h:w * h * 5 // 4].reshape(h // 2, w // 2).astype(np.int16)
        v = fr[w * h * 5 // 4:].reshape(h // 2, w // 2).astype(np.int16)
        if (rep is not None
                and abs(float((y - rep[1]).mean())) <= DEDUPE_DC
                and abs(float((u - rep[2]).mean())) <= DEDUPE_DC
                and abs(float((v - rep[3]).mean())) <= DEDUPE_DC
                and blocks(np.abs(y - rep[1]), 16) <= args.dedupe_y
                and max(blocks(np.abs(u - rep[2]), 8), blocks(np.abs(v - rep[3]), 8)) <= args.dedupe_c):
            dup.append(rep[0])
        else:
            rep = (len(dup), y, u, v)
            dup.append(-1)
    n = sum(d >= 0 for d in dup)
    print(f"dedupe: {n} of {len(dup)} frames repeat the first frame of their run")
    return dup


def encode_audio(args, pcm_path):
    cmd = ["ffmpeg", "-y"]
    if args.start:
        cmd += ["-ss", str(args.start)]
    if args.audio_input:
        # A separate audio file, e.g. the raw .pcm that ld-decode writes
        # alongside the .tbc: far cheaper than re-reading a huge FFV1
        # capture just for its sound track. Raw PCM needs its format
        # given up front.
        if args.audio_input.lower().endswith(".pcm"):
            cmd += ["-f", "s16le", "-ar", str(args.audio_input_rate), "-ac", "2"]
        cmd += ["-i", args.audio_input]
    else:
        cmd += ["-i", args.input]
    if args.frames:
        # match the video's duration exactly
        cmd += ["-t", f"{args.frames * args.fps_den / args.fps_num:.6f}"]
    cmd += [
        "-vn", "-sn",
        "-ar", str(args.audio_rate),
        "-ac", str(AUDIO_CHANNELS),
        "-f", "s16le", pcm_path,
    ]
    try:
        run(cmd)
        return True
    except subprocess.CalledProcessError:
        print("note: no audio track found, continuing without audio")
        return False


def _codes(data, lo):
    """(offset, NAL type, scan resume) of each Annex-B start code in data[lo:],
    exactly as a byte-by-byte scan finds them: 00 00 00 01 when the zero before
    00 00 01 is not part of the previous code, else 00 00 01; the scan resumes
    at the NAL header byte.  bytes.find keeps it linear."""
    n, r = len(data), lo
    while True:
        p = data.find(b"\0\0\1", r)
        if p < 0 or p + 3 >= n:
            return
        yield (p - 1 if p - 1 >= r and data[p - 1] == 0 else p), data[p + 3] & 0x1F, p + 3
        r = p + 3


def start_codes(data):
    """(offset, NAL type) of each start code in an in-memory stream."""
    return ((s, nal) for s, nal, _ in _codes(data, 0))


def file_start_codes(f, block=1 << 26):
    """start_codes over a file read in blocks (absolute offsets), so a
    multi-GB stream is never held in memory."""
    buf, base, r, eof = b"", 0, 0, False
    while True:
        for s, nal, nr in _codes(buf, max(r - base, 0)):
            yield base + s, nal
            r = base + nr
        if eof:
            return
        # an unfinished code needs at most the last 4 bytes, and nothing before r
        keep = max(r, base + len(buf) - 4)
        more = f.read(block)
        eof = not more
        buf, base = buf[keep - base:] + more, keep


def split_nal_types(au):
    """NAL types present in one access unit."""
    return [nal for _, nal in start_codes(au)]


def access_unit_starts(codes):
    """AU start offsets from the start codes (offset, NAL type), in order.

    With repeat-headers + keyint=1 every frame is SPS PPS [SEI] IDR, so a
    new AU starts at each SPS (NAL type 7). Also honour access-unit
    delimiters (type 9) if present; two boundary markers with no VCL NAL
    (type 1 or 5) between them belong to the same AU."""
    starts, vcl = [], False
    for off, nal in codes:
        if nal in (7, 9) and (not starts or vcl):
            starts.append(off)
            vcl = False
        elif nal in (1, 5):
            vcl = True
    if not starts:
        sys.exit("error: no NAL units found - is this an Annex-B H264 stream?")
    if starts[0] != 0:
        starts.insert(0, 0)
    return starts


def split_access_units(data):
    """Split an in-memory Annex-B byte stream into access units."""
    starts = access_unit_starts(start_codes(data)) + [len(data)]
    return [data[a:b] for a, b in zip(starts, starts[1:])]


def parse_aspect(text):
    """'4:3' or '4/3' or '1.3333' -> Fraction."""
    t = text.replace(":", "/")
    try:
        return Fraction(t).limit_denominator(10000)
    except (ValueError, ZeroDivisionError):
        sys.exit(f"error: cannot read display aspect '{text}' (want e.g. 4:3)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--width", type=int, default=BEEB_GRID_WIDTH,
                   help="encoded width (default 832 = 52 us at the Beeb's "
                        "16 MHz pixel clock)")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--display-aspect", default="4:3",
                   help="how the picture should finally be shown "
                        "(default 4:3); sets the stored pixel aspect ratio")
    p.add_argument("--fps-num", type=int, default=25)
    p.add_argument("--fps-den", type=int, default=1)
    p.add_argument("--crf", type=int, default=18,
                   help="x264 quality, lower = better/bigger (default 18)")
    p.add_argument("--preset", default="slow")
    p.add_argument("--scaler", default="lanczos",
                   help="ffmpeg swscale flags for the resize (default "
                        "lanczos, a better fit for 922 -> 832 than bicubic)")
    p.add_argument("--deinterlace", action="store_true",
                   help="prepend yadif=1 for an interlaced source")
    p.add_argument("--vf", default=None,
                   help="replace the scale filter (deinterlace and the "
                        "declared aspect are still applied around it)")
    p.add_argument("--tbc-json", default=None,
                   help="the decode's .tbc.json, for the active window that "
                        "sets the crop (default: <input>.tbc.json if present, "
                        "else derived from the source's declared SAR)")
    p.add_argument("--probe-from", default=None,
                   help="read the source geometry (size, aspect) from this file "
                        "instead of the input, so the input can be a pipe "
                        "(e.g. stillweave.py's per-frame woven/deinterlaced stream)")
    p.add_argument("--no-crop", action="store_true",
                   help="scale the source's full width, padding included "
                        "(the pre-2026-09-07 behaviour, 0.65%% narrow)")
    p.add_argument("--start", default=None, help="ffmpeg -ss start point")
    p.add_argument("--frames", type=int, default=None, help="limit frame count")
    p.add_argument("--expect-frames", type=int, default=None,
                   help="fail (no output written) unless exactly this many frames "
                        "were encoded, e.g. the master's frame count: a reader "
                        "that died early only shortens the input stream")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--dedupe", action="store_true",
                   help="store each held still once: repeats of a run's first frame "
                        "(within --dedupe-y/-c) with identical audio share its record")
    p.add_argument("--dedupe-y", type=float, default=DEDUPE_Y,
                   help=f"max 16x16 luma block mean |diff|, 8-bit (default {DEDUPE_Y})")
    p.add_argument("--dedupe-c", type=float, default=DEDUPE_C,
                   help=f"max 8x8 chroma block mean |diff|, 8-bit (default {DEDUPE_C})")
    p.add_argument("--audio-input", default=None,
                   help="take the sound track from this file instead of the "
                        "video input (a .pcm is read as raw s16le stereo)")
    p.add_argument("--audio-input-rate", type=int, default=44100,
                   help="sample rate of a raw .pcm --audio-input")
    p.add_argument("--audio-rate", type=int, default=AUDIO_RATE,
                   help="PCM sample rate in Hz (48000 for HDMI; 46875 was the "
                        "original PWM-only rate)")
    args = p.parse_args()

    if args.width % 32 or args.height % 16:
        sys.exit("error: width must be a multiple of 32 and height of 16 "
                 "(I420 hardware constraint)")

    par = pixel_aspect(args.width, args.height, parse_aspect(args.display_aspect))
    if par.numerator > 0xFFFFFFFF or par.denominator > 0xFFFFFFFF:
        sys.exit("error: pixel aspect ratio does not fit the header")

    print(f"geometry: {args.width}x{args.height}, "
          f"pixel aspect {par.numerator}/{par.denominator} "
          f"({float(par):.4f}), display {args.display_aspect}")
    if args.width != BEEB_GRID_WIDTH:
        print(f"note: {args.width} is off the Beeb's 16 MHz pixel grid "
              f"({BEEB_GRID_WIDTH} = {PAL_ACTIVE_LINE_US} us at "
              f"{BEEB_PIXEL_CLOCK_HZ // 1_000_000} MHz); the overlay will not "
              "register pixel-for-pixel with the computer's display")

    with tempfile.TemporaryDirectory() as tmp:
        h264_path = os.path.join(tmp, "video.h264")
        pcm_path = os.path.join(tmp, "audio.pcm")

        dup = encode_video(args, par, h264_path)
        has_audio = (not args.no_audio) and encode_audio(args, pcm_path)

        # 2026-10-26: the stream and the sound are no longer read into memory
        # (a 3.5 GB PVF took ~3x its size in RAM): AU boundaries come from one
        # block-wise scan, the index from the AU sizes, then the records are
        # copied from the files.  Same bytes out as before.
        h264 = open(h264_path, "rb")
        n_bytes = os.fstat(h264.fileno()).st_size
        starts = access_unit_starts(file_start_codes(h264)) + [n_bytes]
        extents = list(zip(starts, starts[1:]))
        pcm = open(pcm_path, "rb") if has_audio else None
        pcm_len = os.fstat(pcm.fileno()).st_size if has_audio else 0

        frame_count = len(extents)
        print(f"{frame_count} access units")
        if dup is not None and len(dup) != frame_count:
            sys.exit(f"error: duplicate scan saw {len(dup)} frames, encoder {frame_count}")
        if args.expect_frames is not None and frame_count != args.expect_frames:
            sys.exit(f"error: {frame_count} frames encoded, expected {args.expect_frames} "
                     "(input truncated?)")

        # Per-frame audio slice sizes (fractional accumulation keeps sync
        # for frame rates that do not divide the sample rate)
        bytes_per_sample_pair = AUDIO_CHANNELS * AUDIO_BYTES_PER_SAMPLE

        def audio_slice(i):
            s0 = i * args.audio_rate * args.fps_den // args.fps_num
            s1 = (i + 1) * args.audio_rate * args.fps_den // args.fps_num
            return s0 * bytes_per_sample_pair, s1 * bytes_per_sample_pair

        def audio_chunk(i):
            if not has_audio:
                return b""
            a0, a1 = audio_slice(i)
            pcm.seek(min(a0, pcm_len))
            chunk = pcm.read(a1 - a0)
            if len(chunk) < a1 - a0:      # pad tail with silence
                chunk += b"\0" * (a1 - a0 - len(chunk))
            return chunk

        nominal = (args.audio_rate * args.fps_den // args.fps_num) * bytes_per_sample_pair

        header_size = 64
        index_offset = header_size
        data_offset = index_offset + 4 * frame_count

        # pass 1: the index (record sizes only)
        index = []
        stored = []                 # per frame: True = has its own record
        offset = data_offset
        max_video = 0
        shared = 0
        rep_chunk = None
        for i, (a0, a1) in enumerate(extents):
            vlen = a1 - a0
            max_video = max(max_video, vlen)
            vpad = (-vlen) % 4
            chunk = audio_chunk(i)
            # A repeat of a held still whose audio is also identical (in
            # practice: silent) shares its run's first record: the index
            # entry points back at it.  Index offsets are then no longer
            # strictly increasing; the firmware only ever seeks to index[n].
            if dup is not None:
                if dup[i] < 0:
                    rep_chunk = chunk
                elif chunk == rep_chunk:
                    index.append(index[dup[i]])
                    stored.append(False)
                    shared += 1
                    continue
            index.append(offset)
            stored.append(True)
            offset += 8 + vlen + vpad + len(chunk)
            if offset >= 0xFFFFFFFF:
                sys.exit("error: output exceeds 4GB (FAT32/PVF limit) - "
                         "raise --crf or split the disc")

        header = struct.pack(
            "<16I",
            PVF_MAGIC, PVF_VERSION,
            args.width, args.height,
            args.fps_num, args.fps_den,
            frame_count,
            args.audio_rate if has_audio else 0,
            AUDIO_CHANNELS if has_audio else 0,
            nominal if has_audio else 0,
            index_offset, data_offset,
            max_video,
            par.numerator, par.denominator,     # words 13, 14 - new in v2
            0,
        )

        # pass 2: write, one access unit at a time.  Every AU must be
        # standalone-decodable: at least one IDR slice (type 5) and no
        # non-IDR slices (types 1-4). A leading SEI-only "AU" (possible if
        # the stream did not start with an SPS) or any inter-coded slice
        # would stall the player's frame accounting.
        try:
            with open(args.output, "wb") as out:
                out.write(header)
                out.write(struct.pack(f"<{frame_count}I", *index))
                for i, (a0, a1) in enumerate(extents):
                    h264.seek(a0)
                    au = h264.read(a1 - a0)
                    types = split_nal_types(au)
                    if 5 not in types or any(t in (1, 2, 3, 4) for t in types):
                        sys.exit(f"error: access unit {i} is not a standalone IDR "
                                 f"picture (NAL types {types}) - encoder settings "
                                 "changed?")
                    if not stored[i]:
                        continue
                    chunk = audio_chunk(i)
                    out.write(struct.pack("<II", len(au), len(chunk)))
                    out.write(au)
                    out.write(b"\0" * ((-len(au)) % 4))
                    out.write(chunk)
        except BaseException:
            try:
                os.unlink(args.output)
            except OSError:
                pass
            raise

        total = offset
        if dup is not None:
            print(f"dedupe: {shared} index entries share a record "
                  f"({frame_count - shared} records stored)")
        print(f"wrote {args.output}: {frame_count} frames, "
              f"{total / 1e6:.1f} MB total, "
              f"largest AU {max_video} bytes, "
              f"avg {total // max(frame_count, 1)} bytes/frame"
              f"{f', audio {args.audio_rate} Hz' if has_audio else ''}")
        if max_video > 512 * 1024:
            print("WARNING: an access unit exceeds the player's 512 KB "
                  "staging buffer (H264DEC_INPUT_BUF_SIZE) - raise --crf")


if __name__ == "__main__":
    main()
