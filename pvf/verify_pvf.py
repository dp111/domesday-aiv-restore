#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
verify_pvf.py - sanity check a .pvf built by make_pvfv2.py

Checks the header, index monotonicity, per-record NAL structure
(SPS+PPS+IDR = standalone decodable), that sampled records really do
decode standalone (via ffmpeg), and that the audio duration matches the
video. Run this before copying a multi-gigabyte file to the SD card.

usage: verify_pvf.py video.pvf [--decode-every N]  (0 = only 3 samples)
                     [--expect-frames N]  (fail unless the PVF has N frames)
"""

import argparse
import struct
import subprocess
import sys


def nal_types(au):
    out, i = [], 0
    # bound admits a 3-byte start code ending exactly at the buffer tail
    while i + 3 < len(au):
        if au[i:i + 3] == b"\0\0\1":
            out.append(au[i + 3] & 0x1F)
            i += 3
        elif i + 4 < len(au) and au[i:i + 4] == b"\0\0\0\1":
            out.append(au[i + 4] & 0x1F)
            i += 4
        else:
            i += 1
    return out


def decode_standalone(au, w, h):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0",
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"],
        input=au, capture_output=True)
    return len(p.stdout) // (w * h * 3 // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pvf")
    ap.add_argument("--decode-every", type=int, default=0,
                    help="ffmpeg-decode every Nth record (slow); default: "
                         "first/middle/last only")
    ap.add_argument("--expect-frames", type=int, default=None,
                    help="fail unless the header's frame count is exactly N "
                         "(e.g. the master's), so a truncated build is caught")
    args = ap.parse_args()

    f = open(args.pvf, "rb")
    hdr = struct.unpack("<16I", f.read(64))
    (magic, ver, w, h, fn, fd, fc, ar, ac, abpf,
     idx_off, data_off, maxv) = hdr[:13]
    if magic != 0x31465650 or ver != 1:
        sys.exit("bad magic/version - not a PVF1 file")
    if fc == 0:
        sys.exit("FAIL: zero frames")
    if args.expect_frames is not None and fc != args.expect_frames:
        sys.exit(f"FAIL: {fc} frames, expected {args.expect_frames}")
    print(f"{w}x{h} @ {fn}/{fd} fps, {fc} frames, "
          f"audio {ar} Hz x{ac} ({abpf} B/frame), max AU {maxv} B")
    if w % 32 or h % 16:
        sys.exit("FAIL: dimensions violate the I420 mod32/mod16 rule")
    if maxv > 512 * 1024:
        print("WARNING: max AU exceeds the player's 512 KB staging buffer")

    f.seek(idx_off)
    index = struct.unpack(f"<{fc}I", f.read(4 * fc))
    # make_pvfv2.py --dedupe: a repeated held still points back at its run's
    # record; every other entry must start a new record after the last one
    seen, last = set(), -1
    for o in index:
        if o in seen:
            continue
        if o <= last:
            sys.exit("FAIL: index not monotonic")
        seen.add(o)
        last = o
    if len(seen) < fc:
        print(f"{fc - len(seen)} index entries share an earlier record (dedupe)")
    if index[0] != data_off:
        sys.exit("FAIL: first record not at data_offset")

    to_decode = set(range(0, fc, args.decode_every)) if args.decode_every \
        else {0, fc // 2, fc - 1}

    total_audio = 0
    seen_max = 0
    for n in range(fc):
        f.seek(index[n])
        vlen, alen = struct.unpack("<II", f.read(8))
        seen_max = max(seen_max, vlen)
        total_audio += alen
        au = f.read((vlen + 3) & ~3)[:vlen]
        nals = nal_types(au)
        if not (7 in nals and 8 in nals and 5 in nals):
            sys.exit(f"FAIL: record {n} lacks SPS/PPS/IDR (nals={nals}) - "
                     "not standalone decodable")
        if any(t == 1 for t in nals):
            sys.exit(f"FAIL: record {n} contains a non-IDR slice - "
                     "stream is not all-intra")
        if n in to_decode:
            got = decode_standalone(au, w, h)
            if got != 1:
                sys.exit(f"FAIL: record {n} standalone decode gave "
                         f"{got} frames")
            print(f"record {n}: {vlen} B, decodes standalone OK")

    if seen_max != maxv:
        print(f"note: header max_video_len {maxv} != observed {seen_max}")
    if ar:
        a_s = total_audio / (ac * 2) / ar
        v_s = fc * fd / fn
        print(f"audio {a_s:.3f}s vs video {v_s:.3f}s "
              f"(delta {abs(a_s - v_s) * 1000:.1f} ms)")
        if abs(a_s - v_s) > 0.04:
            sys.exit("FAIL: audio/video duration mismatch")
    print("PASSED")


if __name__ == "__main__":
    main()
