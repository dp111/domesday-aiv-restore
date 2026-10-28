# SPDX-License-Identifier: GPL-3.0-or-later
"""Lay out the CommunityNorth panel and write panel.json.

  select_panel.py <index.json> <stacker stack.py> <out panel.json>

Content was chosen from scan_side.py over the v8 North stack (one 32-line stripe of every 3rd frame:
level, contrast, chroma, detail, and frame-to-frame change for held runs) and checked by eye with
qcrender.py.  North pictures, by kind:
  30-33          black (side start; ds3's first picture)      1400-1405  moving video (rotating UK)
  2236-2239      graph-paper grid, held (2125-2500): dim, low contrast, fine lines
  3003-3007      Test Card F, held (3001-3024)                3030-3034  colour bars, held (3026-3048)
  3080-3083      bars + red field, held (3052-3099); heaviest dropout repair on the side (3082)
  4450-4456      black, held (3100-11776)                     4703-4708  black with the side's 3rd worst residual dropouts
  12656-60, 12776-79, 14108-11, 14840-44, 33872-76, 34790-93  landscape / street photos, low-contrast and dark
  16394-97, 18200-03  saturated (red satellite, menus)        16772-76, 18587-91  dark photos with detail
  25594-98       map with residual dropouts                   27625-31  moving video
  45950-58       held menu where the cursor highlight moves twice mid-run (45953, 45956)
  46292-96       held text page (46096-46327)
  49100-03       held map screen                              50600-03  held map screen (run 50022-50760)
  51070-73, 53404-07  held screens with a text/cursor change  52200-03  aerial photo, held (52138-52645)
  53620-23       held text page
  OS maps (19868-33093 is the side's map section; small text, coloured roads, contours, town plans):
  20370-73, 21280-84 (residual dropouts), 22764-67, 23400-03, 25195-98 (Pitlochry), 25594-98 (contours,
  residual dropouts), 30860-63, 31912-15 (town plan), 32766-69 (town plan), plus the analysis pictures
  in that section (19868, 22075, 24300, 26498, 28694, 30892)
plus black filler (35291-44086).  Not used: 50494-50497 (the side's worst
residual dropouts) and 50016-50020 (a highlight change): picture 50496 / 50020 is unreadable on most
or some captures, so a 4-picture segment there never passes the stacker's sequence sieve.

The stacker analyses (noise, master clustering, cross-master blend weights) a sample of 24 pictures
spread over every picture all captures carry: span[::len(span)//24][:24].  The panel is laid out
so that this pick on the panel returns exactly the full side's 24 pictures: each of them starts or
ends a segment, and every interval between two of them holds exactly STEP panel pictures.  So the
panel stack's analysis, master grouping and weights are the full side's, and its frames can be
compared with the production stack.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcpkg import make_pkg      # noqa: E402
from extract_panel import frame_ranges   # noqa: E402

STEP = 13
SIDE = "CommunityNorth"

# (p0, p1, role); roles: black, motion, still, held, bars, testcard, dropouts, change, saturated, dark
CONTENT = [
    (1400, 1405, "motion"),
    # the next segment is < 50 pictures on, so each guard frame (3006, 3033) is kept and stacked too
    (3003, 3005, "testcard"), (3030, 3032, "bars"), (3080, 3083, "bars dropouts"),
    (4703, 4708, "black dropouts"),
    (12656, 12660, "still"), (12776, 12779, "still lowcontrast"),
    (14108, 14111, "still lowcontrast"), (14840, 14844, "still"),
    (16394, 16397, "still saturated"), (16772, 16776, "still dark"),
    (18200, 18203, "still saturated"), (18587, 18591, "still dark"),
    (20370, 20373, "still map"), (21280, 21284, "still map dropouts"), (22764, 22767, "still map"), (23400, 23403, "still map"), (25195, 25198, "still map"), (25594, 25598, "still map dropouts"),
    (27625, 27631, "motion"), (30860, 30863, "still map"), (31912, 31915, "still map"), (32766, 32769, "still map"),
    (33872, 33876, "still dark"), (34790, 34793, "still lowcontrast"),
    (45950, 45958, "held change"),
    (46292, 46296, "held text"),
    (49100, 49103, "held"), (50600, 50603, "held"),
    (51070, 51073, "held change"), (52200, 52203, "held photo"), (53404, 53407, "held change"),
    (53620, 53623, "held text"),
]
# analysis segments: (pictures before A[i], after); default A[i]..A[i]+3.  A[1] (2239) and A[22]
# (48486) end their segments so the crowded intervals after them have room; A[23] (50782) needs a
# lead-in because ds3 has no picture 50784 (its sieve would only confirm the position at 50785)
LEAD = {1: (3, 0), 22: (3, 0), 23: (4, 3)}
# held runs used for the temporal truth: (r0, r1) inside the run as scanned (refined on extraction)
RUNS = [(2128, 2497, "graph paper"), (3002, 3023, "testcard"), (3027, 3047, "bars"),
        (3053, 3098, "bars red"), (4440, 4720, "black"), (46098, 46325, "text"),
        (50022, 50760, "map"), (52140, 52643, "aerial photo"), (53547, 53705, "text")]


def simulate(stk, caps, segs):
    """Each capture's mini-capture keys as the stacker's sieve will see them."""
    out = {}
    for n, c in caps.items():
        idx = dict(zip(c["keys"], c["fi"]))
        inv = {f: k for k, f in idx.items()}
        fis = set()
        for a, b in frame_ranges(c, segs):
            fis.update(range(a, b + 1))
        raw = [(f, inv[f]) for f in sorted(fis) if f in inv]
        out[n] = sorted({k for _, k in stk.sieve_keys(raw)})
    return out


def layout(stk, caps, A, content):
    """Analysis segments around each A[i], lengths solved so each interval holds STEP keys."""
    segs = []
    for i, a in enumerate(A):
        before, after = LEAD.get(i, (0, 3))
        segs.append({"p0": a - before, "p1": a + after, "role": "analysis", "a": a})
    # a segment needs every picture on every capture (the sieve confirms a position only after 4
    # consecutive picture numbers): move it on past holes, at most 12 pictures
    sets = [set(c["keys"]) for c in caps.values()]
    for p0, p1, r in content:
        for d in range(13):
            if all(all(p in ks for ks in sets) for p in range(p0 + d, p1 + d + 2)):
                break
        else:
            raise SystemExit(f"segment {p0}-{p1}: no fully covered position nearby")
        if d:
            print(f"  segment {p0}-{p1} ({r}) moved by {d}: holes on some capture")
        segs.append({"p0": p0 + d, "p1": p1 + d, "role": r})
    for it in range(60):
        segs.sort(key=lambda s: s["p0"])
        sim = simulate(stk, caps, segs)
        inter = sorted(set.intersection(*[set(v) for v in sim.values()]))
        step = len(inter) // 24
        pick = inter[::max(1, step)][:24]
        if pick == A and step == STEP:
            return segs, sim, inter
        # per interval: count, then grow / shrink the analysis segment at its end
        cnt = []
        for i in range(24):
            hi = A[i + 1] if i < 23 else 10 ** 9
            cnt.append(sum(1 for k in inter if A[i] <= k < hi))
        changed = False
        for i in range(24):
            want = STEP
            if i == 23 and STEP <= cnt[i] < 2 * STEP:
                continue
            if cnt[i] != want:
                s = next(s for s in segs if s.get("a") == A[i])
                if s["p1"] == A[i] and s["p0"] < A[i]:
                    raise SystemExit(f"interval {i} ({A[i]}): {cnt[i]} pictures, need {want}: edit CONTENT")
                new = s["p1"] + (want - cnt[i])
                if new < s["p0"] + 3:
                    raise SystemExit(f"interval {i} ({A[i]}): {cnt[i]} pictures, more than {want}")
                s["p1"] = new
                changed = True
        if not changed:
            raise SystemExit(f"layout did not converge: counts {cnt}, step {step}, pick {pick}")
    raise SystemExit("layout did not converge")


def main():
    index_p, stacker, out = sys.argv[1:4]
    index = json.load(open(index_p))
    top, mod, _ = make_pkg(stacker, os.path.join(os.path.dirname(os.path.abspath(out)), "pkg"))
    sys.path.insert(0, top)
    stk = importlib.import_module(mod)
    A = index["analysis"]
    segs, sim, inter = layout(stk, index["caps"], A, CONTENT)
    union = sorted(set.union(*[set(v) for v in sim.values()]))
    held = [s for s in segs if "held" in s["role"] or s["role"] in ("testcard", "bars", "bars dropouts")]
    runs = []
    for r0, r1, nm in RUNS:
        st = [k for k in union if r0 <= k <= r1]
        if st:
            runs.append({"r0": r0, "r1": r1, "name": nm, "stacked": st,
                         "textured": nm in ("testcard", "text", "map", "aerial photo", "graph paper")})
    panel = {"side": SIDE, "caps": sorted(index["caps"], key=lambda n: int(n.split("ds")[-1])),
             "segments": segs, "stack_keys": union, "analysis": A, "step": STEP,
             "intersection": len(inter), "runs": runs,
             "reference": {"tbc": "/mnt/s/domsday/aiv_final/v8/CommunityNorth.xm4.tbc",
                           "frames_tsv": "/mnt/s/domsday/aiv_final/v8/CommunityNorth_stack_frames.tsv"}}
    json.dump(panel, open(out, "w"), indent=1)
    print(f"{len(segs)} segments, {len(union)} pictures stacked ({len(inter)} on every capture, step "
          f"{len(inter) // 24}); held segments {len(held)}; runs {len(runs)}")
    for n, v in sim.items():
        miss = sorted(set(union) - set(v))
        if miss:
            print(f"  {n}: lacks {len(miss)}: {miss[:12]}")


if __name__ == "__main__":
    main()
