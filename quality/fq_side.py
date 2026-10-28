# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-field / per-frame quality record of a whole side's stacked (or dropout-corrected) TBC (fq, 2026-10-24).

  fq_side.py <side.tbc> <out.csv> [--vits]

One sequential read of the TBC.  Per field:
  picture            CAV picture number (VBI)
  bl_<band>          noise power on the blank vertical-interval lines (field rows 5-14, 20-21, samples
                     200-1050), per band 0-2 / 2-3.5 / 3.5-5.5 / 5.5-8.9 MHz, 16-bit units^2; plus bl_rms total
  do_samples         samples still flagged as dropout in the json (left for dropout-correct / the decoder)
  wSNR, bPSNR        (--vits) ld-process-vits white-flag SNR and black-level PSNR (run on a copy of the
                     json, the original is not touched).  2026-10-26: the installed ld-process-vits does not
                     write vitsMetrics into its json, so the values come from its -d debug lines ("Field #N
                     has wSNR of X (..) and bPSNR of Y (..)", N 1-based); 0 = not measured -> empty (NULL).
Content-independent (blank / test lines only), so frames and versions compare directly.  Caveat: the
cross-master blend works on the active picture only, so blank / VITS lines show the primary-master stack;
the benchmark (make_bench.py / score.py) measures the picture itself.
"""
import sys, os, re, json, csv, subprocess, tempfile
import numpy as np

sys.path.insert(0, os.environ.get('LD', os.path.expanduser('~/ld-decode')))
from lddecode.stack import cav_picture, field_vbi

W, H, FS = 1135, 313, 17734475.0
ROWS = list(range(5, 15)) + [20, 21]
X0, X1 = 200, 1050
N = X1 - X0
WIN = np.hanning(N); WN = (WIN ** 2).sum()
BANDS = [(0, 2e6), (2e6, 3.5e6), (3.5e6, 5.5e6), (5.5e6, FS / 2)]
F = np.fft.rfftfreq(N, 1 / FS)
MASK = [(F >= lo) & (F < hi) for lo, hi in BANDS]


def main():
    tbc, out = sys.argv[1], sys.argv[2]
    js = json.load(open(tbc + '.json'))
    x = np.memmap(tbc, dtype='<u2', mode='r').reshape(-1, H, W)
    nf = min(len(x), len(js['fields']))
    vits = {}
    if '--vits' in sys.argv:
        with tempfile.TemporaryDirectory(dir=os.path.dirname(out) or '.') as td:
            p = subprocess.run(['nice', '-n', '10', 'ld-process-vits', '-d', '-n', '-t', '4', '--input-json', tbc + '.json',
                                '--output-json', f'{td}/v.json', tbc], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=True, errors='replace')
        if p.returncode:
            sys.exit(f'ld-process-vits failed ({p.returncode}): {p.stderr[-500:]}')
        for m in re.finditer(r'Field #(\d+) has wSNR of ([-\d.e+]+) \(.*?\) and bPSNR of ([-\d.e+]+)', p.stderr):
            w, b = float(m[2]), float(m[3])
            vits[int(m[1]) - 1] = {'wSNR': w or None, 'bPSNR': b or None}
        if not vits:
            sys.exit('ld-process-vits gave no per-field metrics (debug output changed?)')
        print(f'VITS: {len(vits)} fields measured, wSNR on {sum(v["wSNR"] is not None for v in vits.values())}', flush=True)
    rows = []
    B = 64
    for k0 in range(0, nf, B):
        blk = np.asarray(x[k0:min(nf, k0 + B)][:, ROWS, X0:X1], dtype=np.float64)
        blk -= blk.mean(-1, keepdims=True)
        P = (np.abs(np.fft.rfft(blk * WIN, axis=-1)) ** 2).mean(1) * 2.0 / WN   # per field, units^2 per bin
        for i in range(len(blk)):
            k = k0 + i
            fj = js['fields'][k]
            do = fj.get('dropOuts') or {}
            dos = int(np.sum(np.array(do.get('endx', []), dtype=int) - np.array(do.get('startx', []), dtype=int))) if do else 0
            bp = [float(P[i][m].sum() / N) for m in MASK]
            r = {'field': k, 'picture': cav_picture(field_vbi(fj)), 'first': int(bool(fj.get('isFirstField', k % 2 == 0))),
                 **{f'bl_{lo / 1e6:g}-{hi / 1e6:.1f}': round(v, 1) for (lo, hi), v in zip(BANDS, bp)},
                 'bl_rms': round(float(np.sqrt(sum(bp))), 2), 'do_samples': dos}
            if vits:
                r['wSNR'] = vits.get(k, {}).get('wSNR'); r['bPSNR'] = vits.get(k, {}).get('bPSNR')
            rows.append(r)
        if k0 % 6400 == 0:
            print(f'{k0}/{nf} fields', flush=True)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    a = np.array([r['bl_rms'] for r in rows])
    print(f'{nf} fields: blank-line noise rms median {np.median(a):.1f}, p95 {np.percentile(a, 95):.1f}, '
          f'worst {a.max():.1f} (field {int(a.argmax())}); dropout samples left {sum(r["do_samples"] for r in rows)}')
    if vits:
        ws = np.array([r['wSNR'] for r in rows if r.get('wSNR') is not None], dtype=float)
        if ws.size:
            print(f'VITS wSNR median {np.median(ws):.2f} dB, p5 {np.percentile(ws, 5):.2f} dB')


if __name__ == '__main__':
    main()
