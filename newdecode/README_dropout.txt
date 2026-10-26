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
