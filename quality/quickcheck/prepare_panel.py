# SPDX-License-Identifier: GPL-3.0-or-later
"""Finish a panel after extract_panel.py: check the mini captures index as planned, and choose the
fixed measurement regions from the production reference frames.

  prepare_panel.py <panel dir> <stacker stack.py>

* indexes every mini capture with the stacker's own TBCFrameSource and checks that the union of
  pictures is panel["stack_keys"] and that the stacker's analysis pick on the panel is the full
  side's 24 pictures (panel["analysis"])
* stills: per frame and field, up to 12 flat and 12 textured 16x64 composite blocks, chosen on the
  OTHER field of the production frame (so the choice is independent of the noise being measured)
* decode: decodes the reference frames of the bars / test card / one frame per still segment with
  ld-chroma-decoder transform2d, finds the bars (columns, rows) and the decoded flat / textured
  32x64 luma blocks (chosen on the other field's rows), and sets the contact-sheet crops
"""
import importlib
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcmetrics as qm          # noqa: E402
import qcrender                 # noqa: E402
import qcmap                    # noqa: E402
from qcpkg import make_pkg      # noqa: E402

NB = 12
OWc, OHc = qcrender.OW, qcrender.OH
# map detail: pictures in these ranges, any segment whose role says map, and these held runs
MAP_RANGES = [(19868, 33093)]           # North: the OS-map section of the side
MAP_RUNS = ("map", "graph paper", "aerial photo")
TEXT_CROP_PICS = [25196, 32767, 22765]   # small lettering (Pitlochry), town-plan street names
# contact-sheet crops: (picture, label, x, y, w, h) in the 928x576 decoded frame
CROPS = [(3031, "bars", 60, 120, 300, 120), (3081, "bars + red, dropouts", 330, 300, 300, 120),
         (3004, "test card F", 300, 150, 300, 160), (2237, "graph paper (dim)", 330, 220, 300, 120),
         (16774, "dark photo", 300, 230, 300, 140), (16396, "saturated", 300, 230, 300, 140),
         (12778, "low-contrast photo", 300, 300, 300, 140), (46294, "held text", 200, 160, 300, 120),
         (45953, "held menu, cursor change", 150, 330, 300, 140), (1402, "moving video", 300, 200, 300, 140),
         (4706, "black + dropouts", 300, 220, 300, 140), (52201, "held aerial photo", 300, 220, 300, 140)]


def choose_blocks(ref_frame):
    grid = qm.block_grid()
    flat, tex = [], []
    for f in (0, 1):
        o = ref_frame[1 - f].astype(np.float64)
        rms, det = [], []
        for y, x in grid:
            r = qm.block_residual(o, y, x)
            rms.append(np.sqrt((r ** 2).mean()))
            det.append(qm.band_ms(r)[1])
        rms, det = np.array(rms), np.array(det)
        lvl = np.array([o[y:y + 16, x:x + 64].mean() for y, x in grid])
        ok = lvl < 52000                      # not clipped white
        fi = [i for i in np.argsort(rms) if ok[i] and rms[i] < 600][:NB]
        ti = [i for i in np.argsort(-det) if ok[i] and rms[i] > 1500][:NB]
        flat += [[f, *grid[i]] for i in fi]
        tex += [[f, *grid[i]] for i in ti]
    return flat, tex


def decoded_blocks(Y):
    """32x64 luma blocks; each is measured on one row parity, chosen on the other."""
    out = {"flat": [], "tex": []}
    ys, xs = range(16, qcrender.OH - 48, 32), range(16, qcrender.OW - 80, 48)
    for par in (0, 1):
        rms, det, pos = [], [], []
        for y in ys:
            for x in xs:
                b = Y[y + 1 - par:y + 32:2, x:x + 64].astype(np.float64)
                b = b - b.mean(axis=1, keepdims=True)
                rms.append(np.sqrt((b ** 2).mean()))
                det.append(qm.band_ms(b)[1:3].sum())
                pos.append((y, x))
        rms, det = np.array(rms), np.array(det)
        out["flat"] += [[par, *pos[i]] for i in np.argsort(rms)[:NB] if rms[i] < 400]
        out["tex"] += [[par, *pos[i]] for i in np.argsort(-det)[:NB] if rms[i] > 1500]
    return out


def find_bars(Y):
    """Colour bars: columns where the median row profile is flat over +-4 px (runs >= 40 px, trimmed),
    rows from the top while each row matches that profile (gaps of <= 4 rows bridged: dropouts)."""
    prof = np.median(Y[60:200].astype(np.float64), axis=0)
    rng = prof.max() - prof.min()
    flat = np.zeros(len(prof), bool)
    flat[4:-4] = np.abs(prof[8:] - prof[:-8]) < 0.01 * rng
    bars, x = [], 0
    while x < len(flat):
        if flat[x]:
            e = x
            while e < len(flat) and flat[e]:
                e += 1
            if e - x >= 40:
                bars.append([int(x + 4), int(e - 4)])
            x = e
        else:
            x += 1
    cols = np.concatenate([np.arange(a, b) for a, b in bars])
    err = np.sqrt(((Y[:, cols].astype(np.float64) - prof[cols]) ** 2).mean(axis=1))
    good = err < 0.04 * rng
    # bridge short bad gaps, then take the longest good run
    g = good.copy()
    i = 0
    while i < len(g):
        if not g[i]:
            j = i
            while j < len(g) and not g[j]:
                j += 1
            if j - i <= 4 and i > 0 and j < len(g):
                g[i:j] = True
            i = j
        else:
            i += 1
    best, bs, i = 0, 0, 0
    while i < len(g):
        if g[i]:
            j = i
            while j < len(g) and g[j]:
                j += 1
            if j - i > best:
                best, bs = j - i, i
            i = j
        else:
            i += 1
    return {"bars": bars, "rows": [int(max(bs + 6, 16)), int(min(bs + best - 6, len(g) - 16))]}


def main():
    pdir, stacker = sys.argv[1:3]
    opt = dict(zip(sys.argv[3::2], sys.argv[4::2]))          # --in FILE --out FILE (default panel.json)
    pj = os.path.join(pdir, opt.get("--in", "panel.json"))
    panel = json.load(open(pj))
    pj = os.path.join(pdir, opt.get("--out", os.path.basename(pj)))
    top, mod, _ = make_pkg(stacker, os.path.join(pdir, "pkg"))
    sys.path.insert(0, top)
    stk = importlib.import_module(mod)
    # ---- index check
    srcs = [stk.TBCFrameSource(os.path.join(pdir, "caps", n), cav=not panel.get("clv"))   # caps/ may hold symlinks
            for n in panel["caps"]]
    ks = [s.keys() for s in srcs]
    union = sorted(set.union(*ks))
    inter = sorted(set.intersection(*ks))
    pick = inter[::max(1, len(inter) // 24)][:24]
    full = panel.get("analysis")
    print(f"index: {len(union)} pictures (planned {len(panel['stack_keys'])}), {len(inter)} on every capture; "
          + ("analysis pick not planned" if full is None else
             "analysis pick " + ("== full side" if pick == full else "!= full side: " + str(pick))))
    if union != panel["stack_keys"]:
        print("  differs from plan: +", sorted(set(union) - set(panel["stack_keys"]))[:20],
              "-", sorted(set(panel["stack_keys"]) - set(union))[:20])
        panel["stack_keys"] = union
    panel["analysis_reproduced"] = pick == full
    for s in srcs:
        s.close()
    # ---- composite blocks on every non-black still / held frame
    ref = np.load(os.path.join(pdir, "ref", "ref.npy"), mmap_mode="r")
    rkeys = [int(k) for k in np.load(os.path.join(pdir, "ref", "ref_keys.npy"))]
    role = {}
    for s in panel["segments"]:
        for p in range(s["p0"], s["p1"] + 2):
            role.setdefault(p, s["role"])
    stills = {}
    for i, k in enumerate(rkeys):
        if "motion" in role.get(k, ""):
            continue
        flat, tex = choose_blocks(np.asarray(ref[i]))
        stills[str(k)] = {"flat": flat, "tex": tex, "role": role.get(k, "")}
    panel["stills"] = stills
    # ---- decode: bars, test card, one frame per segment, the crop frames
    want = set()
    for s in panel["segments"]:
        if s["role"] in ("bars", "bars dropouts", "testcard"):
            want.update(k for k in rkeys if s["p0"] <= k <= s["p1"])
        elif "black" not in s["role"] and s["role"] != "analysis":
            want.add(s["p0"] + 1)
    crops = [tuple(c) for c in panel.get("crop_spec", CROPS)]
    want |= {c[0] for c in crops}
    # map detail pictures: OS maps, map screens, graph paper, aerial photo
    mr = panel.get("map_ranges", MAP_RANGES)
    run_map = {k for r in panel.get("runs", []) if r["name"] in MAP_RUNS for k in r["stacked"]}
    if panel.get("map_pics") is not None:
        # chosen by eye: genuine maps / town plans / lettering only
        maps = sorted(k for k in panel["map_pics"] if k in rkeys)
    else:
        maps = sorted(k for k in rkeys if ("map" in role.get(k, "") or k in run_map
                                           or any(a <= k <= b for a, b in mr)) and "motion" not in role.get(k, ""))
    # bars / test card: their own edge metric (bar edges are not map lines)
    graphic = sorted(k for k in rkeys if (k in set(panel.get("graphic_pics", [])) or role.get(k, "").startswith("bars")
                                          or role.get(k, "").startswith("testcard")) and k not in maps)
    want |= set(maps) | set(graphic)
    want = sorted(k for k in want if k in rkeys)
    meta = qcrender.load_meta(panel["reference"]["tbc"])
    with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
        # reference frames come from ref.npy; the field metadata from the production stack
        from extract_panel import ref_pic2frame
        p2f = ref_pic2frame(panel["reference"])
        qcrender.frames_tbc(None, meta, [p2f[k] for k in want], os.path.join(td, "r"),
                            frames=np.stack([ref[rkeys.index(k)] for k in want]))
        yuv = qcrender.decode(os.path.join(td, "r"))
    dec = {"bar_regions": {}, "tex_blocks": {}, "crops": [], "frames": want}
    for k, y in zip(want, yuv):
        r = role.get(k, "")
        if r.startswith("bars"):
            dec["bar_regions"][str(k)] = find_bars(y[0])
        else:
            dec["tex_blocks"][str(k)] = decoded_blocks(y[0])
    dec["crops"] = [{"pic": p, "label": f"{p} {lab}", "x": x, "y": y, "w": w, "h": h}
                    for p, lab, x, y, w, h in crops if p in want]
    mp = {}
    for k, y in zip(want, yuv):
        if k in maps:
            mp[str(k)] = qcmap.select(np.asarray(ref[rkeys.index(k)]), y[0], seed=k)
    panel["map"] = mp
    gp = {}
    for k, y in zip(want, yuv):
        if k in graphic:
            gp[str(k)] = qcmap.select(np.asarray(ref[rkeys.index(k)]), y[0], seed=k)
    panel["graphic"] = gp
    # 2x crops of small text: around the strongest text block of two map pictures
    for k in [k for k in panel.get("text_crop_pics", TEXT_CROP_PICS) if str(k) in mp][:2]:
        par, ty, tx = mp[str(k)]["dec"]["text"][0]
        x0, y0 = int(min(max(tx - 43, 0), OWc - 150)), int(min(max(ty - 16, 0), OHc - 64))
        dec["crops"].append({"pic": k, "label": f"{k} map text (2x)", "x": x0, "y": y0, "w": 150, "h": 64,
                             "scale": 2})
    map_nd = [k for k in panel.get("text_crop_pics", TEXT_CROP_PICS) if str(k) in mp][:3]
    dec["nd_frames"] = panel.get("nd_spec") or ([p for p, *_ in crops if p in want][:7] + map_nd)[:10]
    panel["decode"] = dec
    print(f"graphic (bars / test card) pictures {len(gp)}; map pictures {len(mp)}: " + ", ".join(f"{k}:{len(v['comp']['edges'])}/{len(v['dec']['edges'])}e"
                                                for k, v in list(mp.items())[:6]))
    for k, br in dec["bar_regions"].items():
        print(f"bars {k}: {len(br['bars'])} bars, rows {br['rows']}")
    print(f"stills {len(stills)}, decoded {len(want)}; tex/flat blocks: "
          + ", ".join(f"{k}:{len(v['flat'])}/{len(v['tex'])}" for k, v in list(dec["tex_blocks"].items())[:8]))
    json.dump(panel, open(pj + ".tmp", "w"), indent=1, default=int)
    os.replace(pj + ".tmp", pj)


if __name__ == "__main__":
    main()
