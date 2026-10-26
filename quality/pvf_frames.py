#!/usr/bin/env python3
"""Write a .pvf's video as one Annex-B H.264 stream in frame order (fq, 2026-10-24).

  pvf_frames.py video.pvf > video.h264

Every record is SPS+PPS+IDR (all-intra, checked by verify_pvf.py), so the records concatenated in index
order are a valid stream with exactly one picture per PVF frame; a deduped entry repeats its shared
record, so the stream has as many pictures as the PVF has frames and lines up 1:1 with the master.
"""
import struct
import sys


def main():
    f = open(sys.argv[1], 'rb')
    hdr = struct.unpack('<16I', f.read(64))
    magic, ver, fc, idx_off = hdr[0], hdr[1], hdr[6], hdr[10]
    if magic != 0x31465650 or ver != 1:
        sys.exit('not a PVF1 file')
    f.seek(idx_off)
    index = struct.unpack(f'<{fc}I', f.read(4 * fc))
    out = sys.stdout.buffer
    for o in index:
        f.seek(o)
        vlen, _ = struct.unpack('<II', f.read(8))
        out.write(f.read(vlen))


if __name__ == '__main__':
    main()
