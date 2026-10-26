#!/usr/bin/env python3
"""Per-frame picture-quality record of every side and version, in one SQLite database (fq, 2026-10-24).

User: "can a record of the quality of each frame at the end" / "I think we should use the sql stuff".
Database: quality.db beside this script (override with FQ_DB).  One run = one side built by one pipeline
version; re-importing a source for a run replaces that source's rows, so it is safe to re-run.

  quality_db.py run <side> <label> [key=value ...]      create/update a run, print its id
        keys recorded as run metadata, e.g. ld=862df7c6 stacker=3dda464d nd=v5-levels master=yuv422p10le
  quality_db.py stack <run> <stack_frames.tsv>           stacker per frame (LDSTACK_FRAME_STATS)
  quality_db.py fields <run> <fq_side.csv>               per field: blank-line noise bands, dropouts, VITS
  quality_db.py nd <run> <satratio.txt> [classes.txt]    newdecode per frame: sat guard, frame class
  quality_db.py pvf <run> <prefix>                       PVF encode loss per frame (pvf_quality.sh logs)
  quality_db.py summary [run ...]                        one line per run (all runs if none given)

Tables: runs, stack_frames, fields, nd_frames, pvf_frames; view frame_quality joins them per frame
(fields averaged over the frame's two fields).  Frame = output frame index of the stack = master frame.
"""
import csv
import os
import re
import sqlite3
import sys
import time

DB = os.environ.get('FQ_DB', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quality.db'))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(run INTEGER PRIMARY KEY, side TEXT, label TEXT, created TEXT, meta TEXT,
    UNIQUE(side, label));
CREATE TABLE IF NOT EXISTS stack_frames(run INTEGER, frame INTEGER, picture INTEGER, avail INTEGER, used INTEGER,
    used_alt INTEGER, filled INTEGER, repaired INTEGER, resid_do INTEGER, used_names TEXT, alt_names TEXT,
    PRIMARY KEY(run, frame));
CREATE TABLE IF NOT EXISTS fields(run INTEGER, field INTEGER, frame INTEGER, picture INTEGER, first INTEGER,
    bl_0_2 REAL, bl_2_35 REAL, bl_35_55 REAL, bl_55_89 REAL, bl_rms REAL, do_samples INTEGER,
    wsnr REAL, bpsnr REAL, PRIMARY KEY(run, field));
CREATE TABLE IF NOT EXISTS nd_frames(run INTEGER, frame INTEGER, satratio REAL, sat_fallback INTEGER,
    class TEXT, PRIMARY KEY(run, frame));
CREATE TABLE IF NOT EXISTS pvf_frames(run INTEGER, frame INTEGER, psnr_y REAL, psnr_u REAL, psnr_v REAL,
    psnr_avg REAL, ssim_y REAL, ssim_all REAL, PRIMARY KEY(run, frame));
DROP VIEW IF EXISTS frame_quality;
CREATE VIEW frame_quality AS
  SELECT r.run, r.side, r.label, f.frame, COALESCE(s.picture, f.picture) picture, s.avail, s.used, s.used_alt,
         s.filled, s.repaired, s.resid_do, f.bl_rms, f.bl_0_2, f.bl_2_35, f.bl_35_55, f.bl_55_89, f.wsnr,
         f.bpsnr, f.do_samples, n.class, n.satratio, n.sat_fallback, p.psnr_y AS pvf_psnr_y,
         p.ssim_y AS pvf_ssim_y
  FROM (SELECT run, frame, MIN(picture) picture, AVG(bl_rms) bl_rms, AVG(bl_0_2) bl_0_2, AVG(bl_2_35) bl_2_35,
               AVG(bl_35_55) bl_35_55, AVG(bl_55_89) bl_55_89, AVG(wsnr) wsnr, AVG(bpsnr) bpsnr,
               SUM(do_samples) do_samples
        FROM fields GROUP BY run, frame) f JOIN runs r USING(run)
  LEFT JOIN stack_frames s USING(run, frame)
  LEFT JOIN nd_frames n USING(run, frame)
  LEFT JOIN pvf_frames p USING(run, frame);
"""


def db():
    c = sqlite3.connect(DB)
    c.executescript(SCHEMA)
    return c


def num(x, f=float):
    try:
        return f(x)
    except (TypeError, ValueError):
        return None


def cmd_run(c, side, label, *kv):
    meta = ' '.join(kv)
    c.execute("INSERT INTO runs(side, label, created, meta) VALUES(?,?,?,?) "
              "ON CONFLICT(side, label) DO UPDATE SET meta = CASE WHEN excluded.meta != '' "
              "THEN excluded.meta ELSE meta END", (side, label, time.strftime('%Y-%m-%d %H:%M'), meta))
    print(c.execute('SELECT run FROM runs WHERE side=? AND label=?', (side, label)).fetchone()[0])


def cmd_stack(c, run, tsv):
    run = int(run)
    rows = []
    with open(tsv) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            rows.append((run, int(r['frame']), num(r['picture'], int), int(r['avail']), int(r['used']),
                         int(r['used_alt']), int(r['filled']), int(r['repaired']), int(r['resid_do']),
                         r['used_names'], r['alt_names']))
    c.execute('DELETE FROM stack_frames WHERE run=?', (run,))
    c.executemany('INSERT INTO stack_frames VALUES(?,?,?,?,?,?,?,?,?,?,?)', rows)
    print(f'stack_frames: {len(rows)} frames')


def cmd_fields(c, run, path):
    run = int(run)
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            k = int(r['field'])
            rows.append((run, k, k // 2, num(r['picture'], int), int(r['first']), num(r['bl_0-2.0']),
                         num(r['bl_2-3.5']), num(r['bl_3.5-5.5']), num(r['bl_5.5-8.9']), num(r['bl_rms']),
                         int(r['do_samples']), num(r.get('wSNR')), num(r.get('bPSNR'))))
    c.execute('DELETE FROM fields WHERE run=?', (run,))
    c.executemany('INSERT INTO fields VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    print(f'fields: {len(rows)} fields')


def cmd_nd(c, run, sat, classes=None):
    run = int(run)
    fr = {}
    for ln in open(sat):
        p = ln.split()
        if len(p) >= 3:
            fr[int(p[0])] = [float(p[1]), int(p[2]), None]
    if classes and os.path.exists(classes):
        for ln in open(classes):
            p = ln.split()
            if len(p) >= 2:
                fr.setdefault(int(p[0]), [None, None, None])[2] = ' '.join(p[1:])
    c.execute('DELETE FROM nd_frames WHERE run=?', (run,))
    c.executemany('INSERT INTO nd_frames VALUES(?,?,?,?,?)', [(run, k, *v) for k, v in sorted(fr.items())])
    print(f'nd_frames: {len(fr)} frames')


def cmd_pvf(c, run, prefix):
    run = int(run)
    ps, ss = {}, {}
    for ln in open(prefix + '.psnr.log'):
        d = dict(re.findall(r'(\w+):([-\w.]+)', ln))
        ps[int(d['n']) - 1] = [num(d.get('psnr_y')), num(d.get('psnr_u')), num(d.get('psnr_v')), num(d.get('psnr_avg'))]
    for ln in open(prefix + '.ssim.log'):
        d = dict(re.findall(r'(\w+):([-\w.]+)', ln))
        ss[int(d['n']) - 1] = [num(d.get('Y')), num(d.get('All'))]
    rows = [(run, k, *ps[k], *ss.get(k, [None, None])) for k in sorted(ps)]
    c.execute('DELETE FROM pvf_frames WHERE run=?', (run,))
    c.executemany('INSERT INTO pvf_frames VALUES(?,?,?,?,?,?,?,?)', rows)
    print(f'pvf_frames: {len(rows)} frames')


def cmd_summary(c, *runs):
    q = 'SELECT run, side, label, meta FROM runs' + (f' WHERE run IN ({",".join("?" * len(runs))})' if runs else '')
    for run, side, label, meta in c.execute(q, runs).fetchall():
        s = c.execute('''SELECT COUNT(*), AVG(used), MIN(used), SUM(used < avail), SUM(filled), SUM(repaired),
                                SUM(resid_do) FROM stack_frames WHERE run=?''', (run,)).fetchone()
        f = c.execute('SELECT AVG(bl_rms), AVG(wsnr), SUM(do_samples) FROM fields WHERE run=?', (run,)).fetchone()
        n = c.execute('SELECT COUNT(*), AVG(satratio), SUM(sat_fallback) FROM nd_frames WHERE run=?', (run,)).fetchone()
        p = c.execute('SELECT AVG(psnr_y), MIN(psnr_y), AVG(ssim_y) FROM pvf_frames WHERE run=?', (run,)).fetchone()
        fmt = lambda v, f='%.2f': '-' if v is None else f % v
        print(f'run {run} {side} {label}: {s[0]} frames, captures used avg {fmt(s[1])} min {fmt(s[2], "%d")}, '
              f'{fmt(s[3], "%d")} frames with a capture rejected, filled {fmt(s[4], "%d")} px, repaired '
              f'{fmt(s[5], "%d")} px, residual dropout {fmt(s[6], "%d")} samples | blank-line noise {fmt(f[0])}, '
              f'VITS wSNR {fmt(f[1])} dB | newdecode {n[0]} video frames, satratio {fmt(n[1], "%.3f")}, '
              f'fallbacks {fmt(n[2], "%d")} | PVF PSNR-Y avg {fmt(p[0])} min {fmt(p[1])}, SSIM-Y {fmt(p[2], "%.4f")}'
              f'{"  [" + meta + "]" if meta else ""}')


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('run', 'stack', 'fields', 'nd', 'pvf', 'summary'):
        sys.exit(__doc__)
    c = db()
    globals()['cmd_' + sys.argv[1]](c, *sys.argv[2:])
    c.commit()


if __name__ == '__main__':
    main()
