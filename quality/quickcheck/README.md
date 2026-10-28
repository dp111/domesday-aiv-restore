# quickcheck: fast regression check for stacker / decoder changes

Instead of re-stacking and decoding a whole side (1-2 hours) every candidate is stacked and scored on a
fixed **panel**: mini captures holding ~320 chosen pictures of every capture of a side.  Scoring is done
in memory while the stacker is still writing (frames are read back from /dev/shm as they land), the
decode runs several ld-chroma-decoder processes at once, and nothing but the results is written.

    cd /mnt/e/domsday_scratch/quickcheck
    PY=/mnt/s/domsday/ldenv/bin/python QC=/mnt/s/domsday/domesday-aiv-restore/quality/quickcheck
    $PY $QC/quickcheck.py --stacker /path/to/stack.py --baseline north2/results/base_3dda464d
    #   --panel DIR        default /mnt/e/domsday_scratch/quickcheck/north2 (south_s3000 = e2e South bars)
    #   --decoder t2d      (default) transform2d bars / test card / stills / maps; none = composite only;
    #                      v6 adds newdecode v6 on the key pictures (panel decode.nd_frames), each decoded
    #                      inside its run of consecutive panel pictures (--nd-context 8 either side: newdecode's
    #                      held-still averaging and still/video classification need the neighbours), and
    #                      ALWAYS the baseline's kept stack through the same newdecode on the same pictures
    #                      (cached in the baseline dir per newdecode + wrapper md5 and picture list; the
    #                      baseline must have been run with --keep, else a FAIL row).  Skipped while
    #                      aiv-gpu.lock is held, nd_gpuwrap runs or < 6 GB GPU memory is free.
    #   --out DIR          default <panel>/results/<md5>_<time>
    #   --jobs 12 --block 10   stacker workers (capped so MemAvailable stays > ~6 GB) / LDSTACK_BLOCK
    #   --stack-extra "..."    extra stacker arguments;  LDSTACK_* environment variables pass through
    #   --keep             keep the panel stack (stk.tbc, 450 MB) in the results dir
    #   --from-stack DIR   rescore a kept stack without stacking (after changing a metric)
    #   --sequential       no overlaps (for timing)

`--stacker` is a stack.py file (copied, so it can be edited while a check runs; run as package
`qcstk` beside the production ld-decode modules, never as lddecode_xm3, which the scheduler stops) or a
package directory holding stack.py.  Exit status 1 when any check FAILs.  Output: `results.json`
(log, per-frame and summary metrics, checks), `stack.log`, `crops.npz` and `contact.png`
(fixed crops, candidate | baseline | difference x4).  A baseline scored by an older quickcheck is
re-summarised from its per-frame data, so metric changes do not need the baseline re-run.

## Checks (candidate vs baseline unless absolute)

| check | how | FAIL when |
|---|---|---|
| master weights | the blend weights in the stacker log against the per-master error covariance measured on the panel (`master_cov`, see below): optimum w = M^-1 1, predicted blend error w'Mw | blend error > 10 % above optimum in any band, or a master's weight > 0.15 above its optimum (absolute, not vs baseline) |
| noise, flat stills | 16x64 composite blocks, each line minus DC + slope + fsc carrier, per band 0-2 / 2-3.5 / 3.5-5.5 / 5.5+ MHz; blocks chosen on the other field of the production frame | > 1.10 x baseline |
| error vs held-run truth | panel frames of held stills against the mean of the production stack's frames of the same run that the panel does not stack (same PAL 4-frame phase), minus that mean's own noise | > 1.10 x baseline |
| detail | textured blocks' band energy minus the frame's flat-block noise (composite 0-2 and 2-3.5 MHz; decoded Y 2-3.5 and 3.5-5.5 MHz); held textured runs: gain = <cand, truth> / <truth, truth> | < 0.95 x baseline |
| error at dropout pixels | rms vs the held-run truth on the scored held-run frames, at the pixels flagged on any capture or by the candidate or baseline stack (+-2 samples): the same pixels for both | > 1.10 x baseline |
| residual dropouts (info) | samples still flagged in the stacked .tbc.json: depends on each stacker's flagging convention, so not gated | - |
| flicker | std over a held run's frames of level offset, chroma phase and saturation vs the truth (hue / saturation only on runs with colour) | > 1.05 x baseline (+0.5 level / 0.02) |
| bars grain | Y per bar, rms after row and column mean removal (South s3000 incl. 3092 = stack frame 152 / North 3030-3083); transform2d and newdecode v6 | mean, worst bar, or any single bar (cand/base per bar) > 1.10 x baseline |
| newdecode flat grain | v6 Y in flat 32x64 blocks | > 1.10 x baseline |
| decoded flat grain | transform2d Y in flat 32x64 blocks, same | > 1.10 x baseline |
| map line / text sharpness | 10-90 % rise width (samples) of up to 400 strong edges per map picture (ink on paper, step > 8000), positions fixed from the production frame, re-located +-2 and followed while monotone; stack ([1,2,1] luma), transform2d Y, newdecode Y (v6) | > 1.01 x baseline |
| map text-area detail | 2-5 MHz energy of the 32 most detailed blocks less that of the 32 flattest bright paper blocks (chosen on the other field / row parity); stack, t2d, v6 | < 0.98 x baseline |
| info | noise on black frames (the old stacker flattened them: not comparable), match to the production stack | - |

**Why this truth.** A held still is several disc frames of one picture, each with its own disc and player
noise.  The mean of 4-20 production frames of the same run and PAL phase (frames k, k+4, ...: a plain
mean cancels the chroma) has noise ~1/sqrt(n) of a stacked frame and shares no noise with the scored
frames, which are never in it; its remaining noise is measured and subtracted.  Systematic properties
of the production stack are in it (so it is a near-truth, not a truth); the master weights are judged
on frame-to-frame differences, where those cancel.

**master_cov** (`measure_masters.py`): every capture's field of 6 held-run frames is registered to the
truth (shift, chroma complex gain, gain/offset; black: offset only), residuals of consecutive frames are
differenced (fixed errors cancel) and the per-band covariance between captures is accumulated, lines with
a dropout on any capture left out.  The part all captures share (median cross-master covariance: the
source's own per-frame noise; no weighting changes it) is removed.  North (2-3.5 MHz): captures of a master
correlate 0.6-0.8 (disc noise), different masters 0.16 (= the shared part).

Robustness (2026-10-28): split captures (ds1a / ds1b) count as one capture; each capture pair uses only the
rows clean on both captures in both frames (>= 40), and a pair needs >= 4 measured field pairs; per held
run sums are kept, and the weights check uses the set of runs that every capture it keeps has (most
masters, then most captures, then most runs), leaving out captures that were never measured together;
the shared part is the 20th percentile of the cross-capture covariances; the master error covariance is
projected onto the PSD cone and a band is judged only if its smallest eigenvalue is >= 5 % of its largest
variance.  "insufficient data" (no verdict) when a master has no measured capture or no band is
separable; weights measured on a single held run are reported but not gated.

## The North panel (`north2/`, 6.9 GB; `north/` holds only the results of its first version)

Chosen from `scan_side.py` (one 32-line stripe of every 3rd frame of the v8 stack: level, contrast,
chroma, detail and change from frame to frame) and checked by eye (`qcrender.py`); listed in
`select_panel.py`: black (side start, held black, black with residual dropouts), moving video (1400-05,
27625-31), graph-paper grid held (dim, low contrast), Test Card F held, colour bars held, bars + red held
(heaviest dropout repair, 3082), landscape / street photos (low contrast, dark with detail), saturated
satellite / menu screens, held menu with the cursor highlight moving twice mid-run (45950-58), held text
pages, held map, held aerial photo, OS maps (the side's map section 19868-33093: small lettering,
coloured roads, contours, town plans; 2 with residual dropouts), plus black filler.  84 map pictures
(OS maps, graph paper, map screens, aerial photo) carry the map detail check; 2x crops of map text
are on the contact sheet.  320 pictures in 53
segments (each >= 4 pictures: the stacker's sequence sieve only accepts a jump after 4 consecutive
picture numbers; one trailing guard frame per segment).  Held runs (truth): graph paper, test card,
bars, bars red, black, 2 text pages, map, aerial photo.

The stacker analyses 24 pictures spread over all pictures every capture carries; the panel is laid out so
that this pick on the panel is exactly the full side's 24 (every interval between two of them holds 13
panel pictures), so the panel stack's analysis, master groups and blend weights equal production's.
Verified: the 12 mini captures index to the same original frames as the full captures for every picture;
stacked with the production stacker (7b0b2eda) the log's noise / masters / weights are identical to the
v8 North stack, and 164 of 256 frames that are not the first of a segment match production to +-1 LSB
(rms <= 0.05).  The rest differ where the registration warm start differs: a segment's first frame (and
any frame that was a block start in production but not in the panel, or vice versa) starts its sub-pixel
search from another hint, and the difference fades over 1-3 frames (median rms 1.3; on black frames,
where registration has nothing to lock to, up to rms 500 over 3 frames).  Candidate and baseline are
stacked from the same panel, so this does not enter a comparison.

## Rebuilding a panel (when captures are re-decoded)

    PY=/mnt/s/domsday/ldenv/bin/python QC=/mnt/s/domsday/domesday-aiv-restore/quality/quickcheck STK=/mnt/s/domsday/xm_study/merge/stack_m.py
    $PY $QC/scan_side.py <side stack.tbc> scan.npz --step 3          # ~10 min, sparse reads
    $PY $QC/index_caps.py index.json $STK <capture bases...>         # stacker's own index + analysis sample
    $PY $QC/select_panel.py index.json $STK panel/panel.json          # edit CONTENT / RUNS / LEAD first
    $PY $QC/extract_panel.py panel/panel.json index.json panel        # mini captures + reference (double-buffered)
    $PY $QC/prepare_panel.py panel $STK                               # index check, blocks, bars, crops
    $PY $QC/measure_masters.py panel --frames-per-run 6               # master_cov

**South** (CommunitySouth captures re-decoded): same steps with the new capture bases and the South stack
(`reference` in select_panel.py: the stack's .tbc and _stack_frames.tsv), new CONTENT / RUNS from its
scan (Test Card F 3000-3024, 75 % bars 3026-3048, bars + red 3050-3099 as on North; known dark maps
~22991, photo 14001, colour map 50001), and `LEAD` re-solved for its 24 analysis pictures (the
script reports an interval that cannot be filled; move content or LEAD until it converges).  Until then
`south_s3000/` (`make_segment_panel.py south_s3000_spec.json south_s3000`) uses the e2e s3000 segment
decodes of ds1/10/11/12 (220 pictures, stacked whole, truth from the e2e stack's own unscored frames).

## Speed

Measured 2026-10-27 on the shared machine (load 40-47 from production decodes and other work):
North panel, 12 captures, 320 pictures, 16 workers, blocks of 10: installed stacker 544 s total
(stack 540 s, scoring after the stack 2 s, transform2d 3 s), old stacker 1224 s; stacker peak PSS
5.6-6.5 GB.  South s3000 (4 captures, 220 pictures): 178 s / 352 s.  Scoring is overlapped with the
stack (sequential: 39 s scoring + 9 s decode after it).  The stacker is CPU-bound (~70 CPU-minutes for
the North panel), so on a quiet machine expect ~5 min.  `--decoder v6` adds ~200 s (newdecode on 10
frames, mostly compile + solves; ~3 s per decoded picture) for the candidate and the same again,
once, for the baseline (then cached).  Why the v6 stage decodes runs, not lone frames: cand_h6 passed the
transform2d bars check but came out ~9x grainier per bar under newdecode on South s3000 (lead's
h6check): newdecode classifies and averages held stills from the stack's frame-to-frame behaviour.  With 12 workers (RAM rule) cand_h6 took stack 618 s on North (PSS 4.2 GB) and
210 s on South.  The v6 stage runs newdecode under `flock aiv-gpu.lock` (one newdecode at a time across
all users); South s3000 = 68 pictures, North = 41, ~12-17 min each on the shared GPU, and lock waits of up
to an hour when other GPU work queues.  Results: `north2/results/{base_3dda464d,h6_a0f7aabd}`,
`south_s3000/results/{base_3dda464d,cand_7b0b2eda,h6_a0f7aabd}` (stacks kept for --from-stack; v6
caches `v6cache_*.npz` beside them); first-panel results in `north/results/`.  The South e2e
reference stack (e2e/s3000/new/stk.tbc) was made by an earlier stacker build: it matches 3dda464d to
rms 44 and 7b0b2eda to rms 414, which is fine for a near-truth of held runs but not a match test.
Panel extraction: 693 s for 12 captures with reads overlapped with writes (read 680 s + write 147 s
sequential = 827 s), reference + run means 256 s.

## Generalisation suite (other disc sets)

    $PY $QC/quickcheck_suite.py --stacker X [--decoder Y] \
        [--baseline-stacker /mnt/s/domsday/prefix_backup_2026-10-26/stack_m.py] \
        [--baseline-decoder /mnt/s/domsday/paldec_dev/v6_0] [--panels DIR ...] [--par 2] [--jobs 6]

runs quickcheck on every panel (North, South s3000 and the panels in
`/mnt/e/domsday_scratch/quickcheck/gen/`), each against its baseline (old stacker + newdecode v6_0 =
production as it was; run once per panel and kept in `<panel>/results/base_<stacker md5>_<decoder md5>`),
and writes `suite/<tag>/suite.txt` (one line per side, then every side's full check table),
`suite.json` and one contact sheet per side.  Exit 1 on any FAIL.

Panels (`gen/<side>/`, built by `gen_select.py` + `gen_build.py` from a spec `gen/<side>.spec.json`):
segments chosen from a sparse scan of the side's existing stack (`gen/scan/`): the longest held run
with detail (40 pictures, 6 scored against the mean of the other 34 in a reference stack of the panel
made with the baseline stacker), a held run with colour (30), real motion (30), and 8 pictures each of
the most detailed bright frame (maps, text), the most saturated, the darkest with detail and the most
residual dropouts.  Each capture's segments are short RF decodes (`ld-decode -t 4 --PAL
--vits_white_level [--V4300D_coherent_subtract] --noEFM --seek p0-3 -l n+6`) or cuts of a full decode,
joined back to back into one mini capture per copy (picture numbers / CLV time codes as decoded).  The
stacker's 24-picture analysis sample comes from the panel itself; the masters formed are compared with
the full-side stacks' in the report.  A segment that does not decode on a copy is left out of it.

Extra gates used here: **motion** (frames 10-12 of the motion segment): combing index (decoded luma
line-to-line over same-field roughness, t2d and newdecode) not > 1.10 x baseline, and newdecode
ghosting (projection of newdecode - t2d onto the neighbours' mean - t2d, 0 = none, 1 = averaged with the
neighbours) not > baseline + 0.05; **audio**: the production squelch_pcm.py + mono_pcm.py on the stacked
.pcm: seconds of programme (correlated L/R, rms > 200) muted, and seconds mono-combined although L/R are
uncorrelated, not more than baseline.  City_Financial1 (one copy) has no stack (`no_stack`: the capture is
the stack), so only the decoder checks apply.

## Held-run truth from the raw captures (2026-10-28, `build_truth.py`)

The first truth was the mean of the reference (baseline) stack's unscored run frames, so it carried
the baseline stacker's own choices: its least-squares level fit compresses contrast (raw captures sit at
1.03-1.08 LF luma gain against it), and on Countryside it blends the bad capture ds3 at 0.5.
`build_truth.py <panel> [--full-index ...]` rebuilds it from the RAW captures, per held run and PAL phase,
from up to 5 unscored pictures per phase:
- each capture's field is registered to the old run mean (shift and chroma phase only), and its luma
  gain / offset and chroma gain against it are measured; every capture is then rescaled to the MEDIAN of
  the captures' gains / offsets, so neither a single capture nor a stacker sets the contrast;
- a gross outlier is dropped: a capture whose own random error (the master_cov diagonal, frame to
  frame, all bands) is > 4x the median capture's (logged in `build_truth.log` and panel_v2.json "truth":
  Countryside ds3 at 26x; every other capture of the suite is within 0.5-1.5x).  This does not drop a
  capture just for being another master (a test against the other captures' median frame would: a lone
  alt master's frame-to-frame residual is 5-10x in variance because its disc noise is its own);
- the frame's truth = mean over masters of each master's mean (captures equal within a master, master_cov
  clusters, split captures as one; a capture's dropout rows are left out of its master's mean);
- the phase truth = mean over its frames, with its own noise (frame-to-frame scatter / n) as the floor;
- `truth0` (largest master only) is scored as a sensitivity check ("error vs master-0-only truth", info).
Pictures the mini captures do not hold are read from the full captures (North: `--full-index
north_index.json`).  Written to `ref/truth_*.npy` / `ref/truth0_*.npy` and `panel_v2.json` (the panel
description plus "truth"); use it with `quickcheck.py --panel-file panel_v2.json` (or QC_PANEL_FILE)
and `quickcheck_suite.py --panel-file panel_v2.json`.  The error-at-dropout-pixels metric uses the same
truth.  master_cov still registers to the old run mean (its frame differencing cancels the reference).

## Edge sharpness without noise bias (2026-10-28, qcmap)

The first rise measure (monotone walk from the steepest point of one row) depends on noise: noise
stops the walk early, so a NOISIER picture reads as a SHARPER edge (NationalA bar run: single frames
1.96-2.66, the 9-frame mean 3.52, the mean plus noise 2.16).  Replaced (`rise`; the walk is kept as
`rise_walk`, info) by:
- edges chosen once on the reference (map / graphic picture): strong gradient peaks that run along
  >= 2/3 of 13 rows and whose averaged profile is one clean step (erf fit residual < 0.1, so thin lines,
  corners and clutter are not used); the sub-pixel edge centre of every row is stored;
- on a candidate frame each edge's profile is averaged along the edge at those REFERENCE centres (the
  candidate's noise cannot move the alignment), oriented, and normalised with plateaux at fixed
  -8..-5 / +5..+8 samples; all edges of the picture are averaged into ONE profile (thousands of samples)
  and an erf step is fitted; rise = its 10-90 % width (1.8124 s).
`test_rise_noise.py` (17 rows per edge, up to 200 / 400 edges per picture): on clean held-run truths
(north2 graph paper, test card, bars, text, map, aerial runs) with white noise at 0.5x / 1x / 2x a
typical stack residual (420), the composite rise moves by <= 0.2 % (0.5-1x) and <= 0.8 % (2x); synthetic
blurred steps < 0.8 %.  After transform2d it moves by 0.1-0.7 % at 0.5x but 1.3-3.6 % at 1x (always
wider): transform2d's output really is softer on noisier input, so this is the decoder, not the
measure; the decoded map gates therefore use 1.03x on the mean and 1.05x per picture (stack: 1.01x /
1.03x).  The old walk moved by up to 3.9 % on the same edges and reads noisier pictures as sharper.
The text-area 2-5 MHz detail (text blocks minus paper blocks, i.e. above the measured noise floor)
moves by < 0.6 % at 0.5-1x (2.9 % at 2x on one run): already noise-corrected.
Bars and test cards have their own "bar / test card edge rise" (gate 1.03x); map gates use map pictures
only (panel "map_pics" when set) and are also gated per picture (worst picture rise <= 1.03x, text detail
>= 0.95x); the number of map pictures measured is reported.

## Score stamps and flat-noise confounders (2026-10-28)

- Every results.json carries `stamp`: md5 of the scoring code (qcmap.py, qcmetrics.py and quickcheck's
  scoring functions: Scorer, summarise, score_decoded, decoded_summary, nd_summary, dropout_error,
  weights_check, ...) and of the panel file used.  A baseline whose stamp differs from the running code /
  panel is re-scored in place from its kept stack before any comparison (`rescore.log` in its dir); if it
  has no kept stack the run stops, unless `--allow-stale`.  Why: the per-picture rise of north2 52200 read
  3.12 stored and 7.50 re-scored: edge geometry found with 13 rows was being measured with the 17-row
  offsets of a later STEP_ROWS (fixed: the row span is taken from the stored geometry), and nothing
  showed that the stored score came from other code.  Edge selection is deterministic: chosen on the
  reference picture with a generator seeded by the picture number.
- Flat noise (composite flat stills, decoded flat grain, newdecode flat grain) leaves out near-black and
  low-APL frames (mean level < 22000; they are reported as "noise black frames", info), and each frame's
  noise is divided by the output's LF (< 1 MHz) luma gain against the median of the raw captures' frames
  of that picture, so a stack that loses contrast does not read as cleaner.  (EcoDisc's t2d flat-grain
  FAIL came entirely from black pictures 53924 / 53927 squashed by the old LS fit.)

## Running the final suite (panel_v3)

    cd /mnt/e/domsday_scratch/quickcheck
    Q=/mnt/e/domsday_scratch/quickcheck
    P="$Q/north2 $Q/south_s3000 $Q/gen/NationalA_v3 $Q/gen/NationalB_v3 $Q/gen/EcoDisc_S1 $Q/gen/Countryside_S1_v3 \
       $Q/gen/BGB_S1_v2 $Q/gen/UN_v3 $Q/gen/City_Culture1-Side1_v3 $Q/gen/City_Financial1-Side1_v3"
    /mnt/s/domsday/ldenv/bin/python /mnt/s/domsday/domesday-aiv-restore/quality/quickcheck/quickcheck_suite.py \
        --stacker <candidate stack.py> --decoder <newdecode dir, e.g. /mnt/s/domsday/paldec_dev/v6dh_cand8> \
        --baseline-stacker /mnt/s/domsday/prefix_backup_2026-10-26/stack_m.py \
        --baseline-decoder /mnt/s/domsday/paldec_dev/v6_0 \
        --panel-file panel_v3.json --baseline-suffix _v3 --panels $P --par 1 --jobs 6 --out $Q/suite/<name>

Baselines (`<panel>/results/base_3dda464d_471bc290_v3`) are reused when their stamp (measure code, panel
file, truth files) matches, and re-scored from their kept stack otherwise; newdecode runs under
aiv-gpu.lock, so expect 1-3 h for the ten panels.

**Reading suite.txt.** The first lines name the candidate and baseline (stacker md5, frozen decoder dir)
and give one line per panel: PASS, FAIL with the failed checks, SKIP (newdecode stage not run) or ERROR
(see the panel's `<results>.log`).  Below that, every panel's full table: check name, candidate value,
baseline value, ratio and verdict (`info` rows are reported, not gated).  The checks that matter most:
master weights (absolute, vs the measured per-master noise; "not gated" when measured on one held run),
error vs held-run truth per band, error at dropout pixels, flat-still noise and decoded flat grain
(low-APL frames left out, gain-normalised), bars grain per bar, flicker, motion combing / ghosting,
and for maps the text / paper area error vs truth (primary, 1.02x), then edge rise (mean and worst
picture with >= 10 edges; the edge counts per picture are listed in an info row).  Contact sheets are
`<out>/<panel>.png` (candidate | baseline | difference).
