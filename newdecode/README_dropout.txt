Dropout handling in newdecode (v5 = v4 + dropout mask, 2026-10-19), for single/two-copy sides (City Financial1-Side1 = DS2 only).
newdecode.py --dropout_mask 1 (or PALDEC_DOMASK=1): the samples listed as dropOuts in <input>.tbc.json
(fieldLine is 1-based; dilated +-PALDEC_DOPAD=6 samples) get zero data weight in every solver
(optcore: per-pixel Dm on the residual before the row spectrum; solve_held also ignores them in the
outlier test), so the spatial/temporal priors fill them - a held run from its other frames.
The input is still the ld-dropout-corrected TBC (its line copies are only the starting point).
Video: only runs of >= PALDEC_DO_VMIN=3 interlaced frames are masked (a lone interlaced frame's
temporal priors would fill from a different picture); a saturation-guard fallback frame keeps
transform3d but takes the masked solve's LUMA in its dropout pixels.
Mask off: identical to v4 (0.02 8-bit levels = GPU noise).

Test (E:/domsday_scratch/domask_test, Financial1 Sep TBC, ~12 synthetic dropouts/field scored against
the clean-clip decode, RMS 8-bit inside the dropouts, v4 -> v5):
  held stills (12840+80)      Y 14.95 -> 3.11   C 1.49 -> 0.55
  film runs                   Y 14.11 -> 11.11  C 2.31 -> 1.67
  single stills               Y  8.46 -> 6.43   C 1.75 -> 0.90
  video run (6760+80): solved Y 3.98 -> 2.21; fallback frames Y 10.57 -> 4.44 (C unchanged 1.68 -> 1.70)
  lone interlaced frames      unchanged (not masked)
Real dropouts (held clip): v4's dark line-copy streaks through text are gone.
Chain: ND_EXTRA="--dropout_mask 1" pipeline/nd_chunked.sh ...

2026-10-24 v5 + dropout interpolation (PALDEC_DOI, default replace; off = previous behaviour).  Outside held
stills the masked solve's dropout pixels (flagged span +-PALDEC_DOI_PAD=2) are replaced by a vertical
interpolation of the decoded picture between the nearest clean rows above/below (frame rows +-1..4; interlaced
video: same-field rows +-2..8).  Held stills keep the solve.  fq dofill test (quality/dofill,
benchS3 + syn_ds3 dropout shapes, RMS 8-bit inside dropouts, floor without dropouts 4.48/4.03):
  line copy, no mask 11.36/5.90 | production mask 8.96/4.39 | + DOI replace 8.03/4.35 | mix 8.07/4.31
  composite-domain above/below interpolation (phase-matched lines 3-8 picture lines away) is worse: 15.47 / 10.81.
2026-10-25 + temporal dropout fill (PALDEC_DOT=1.5 default, 0 = off; backup newdecode.py.pre-dot): after the
vertical interpolation a dropout pixel takes the mean of the previous / next frame where that neighbour is
clean and locally static (13x33 box mean |dY| and |dC| < 1.5 levels).  benchS3: 8.03/4.35 -> 7.98/4.29,
never worse per segment; held 4.45 -> 4.19, bars 4.57 -> 4.33, motion patterns 8.29 -> 8.20.  The stills
segments gain nothing: every CAV still frame is a different photograph.  Looser thresholds (4-6) cost chroma.

v6 (2026-10-26) = v5 + review fixes.  Each is switchable; the v5 behaviour in brackets.
Bench: fq benchS5 (make_bench now applies the disc channel smear H, the inverse of deghost's G (FQ_SMEAR,
default 0.16,9), and writes dropouts at row fieldLine-1); before, the bench input was too sharp (noise-free
decode luma gain 1-2 MHz 1.16 = 1/|H|) and every bench tuning was biased.
 1. PALDEC_DEGHOST=0 skips the deghost inverse (newdecode + vidsplice; part of the dg_cache stamp).
 2. Noise spectrum PALDEC_NOISE=active [vbi]: noisepsd.noise_psd_active - same-phase field differences
    (k, k+8) after deghost+deherring, 16x232 blocks, per 0.54 MHz band the lowest-energy (static) mode;
    guarded by the blank-line estimate (noisepsd.combine).  Blank lines see only the primary master:
    South v8 stack: blank lines sigma 939 vs picture 330; bench Test Card: true 380-395, estimate 364.
 3. noise_psd (blank lines) is a per-bin median and skips fields with dropouts on those lines.
 4. DC / Nyquist rfft bins weighted once (PALDEC_NW_EDGES=0: twice); noise_sigma likewise.
    + PALDEC_NW_NORM=inband [mean]: data weights scaled to the quietest 0-5 MHz band, not to the mean over all
    bins, which the near-empty >7.5 MHz bins dominated (bench in-band weights all at the floor, real South
    clips mostly at the cap: a 4x data/prior difference between bench and discs).
    PALDEC_KSCALE / PALDEC_KSCALE_C: prior scale knobs (1 = best on benchS4; 2.5 worse).
 5. DOT threshold PALDEC_DOT x sigma / PALDEC_DOT_REF(600) [absolute: PALDEC_DOT_ABS=1].
 6. solve_held avg mode weights each frame by its dropout mask (PALDEC_HELD_DOW=0: v5).
 7. Windows end at run boundaries (extend to <= PALDEC_WIN_MAX=96 frames, else end where the run
    starts; PALDEC_WIN_MAX=64: v5).
 8. doi_mask from the unclipped flagged span (v5 lost up to 4 flagged columns at the picture edges).
 9. A guard-fallback frame keeps the masked solve's luma in its holes through DOI (PALDEC_DOI_FB=interp: v5).
10. vidsplice takes the levels from the TBC json (palmodel.set_levels, shared with newdecode).
11. Held-run outlier test (optcore.held_outliers) against T x max(global median, the block's own median over
    the run), T 5 (PALDEC_HELD_LOCN=0: v5, T 3).  Saturated / edge blocks have a larger residual in every
    frame (player chroma errors, timing jitter) and were outliers in a fraction of the frames, which then kept
    their own unaveraged solution: grain in saturated colour (South bars 3092).
