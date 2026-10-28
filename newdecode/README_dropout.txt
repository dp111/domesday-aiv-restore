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

Test (Financial1 Sep TBC, ~12 synthetic dropouts/field scored against
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
the fq benchmark + syn_ds3 dropout shapes, RMS 8-bit inside dropouts, floor without dropouts 4.48/4.03):
  line copy, no mask 11.36/5.90 | production mask 8.96/4.39 | + DOI replace 8.03/4.35 | mix 8.07/4.31
  composite-domain above/below interpolation (phase-matched lines 3-8 picture lines away) is worse: 15.47 / 10.81.
2026-10-25 + temporal dropout fill (PALDEC_DOT=1.5 default, 0 = off; backup newdecode.py.pre-dot): after the
vertical interpolation a dropout pixel takes the mean of the previous / next frame where that neighbour is
clean and locally static (13x33 box mean |dY| and |dC| < 1.5 levels).  the fq benchmark: 8.03/4.35 -> 7.98/4.29,
never worse per segment; held 4.45 -> 4.19, bars 4.57 -> 4.33, motion patterns 8.29 -> 8.20.  The stills
segments gain nothing: every CAV still frame is a different photograph.  Looser thresholds (4-6) cost chroma.

v6 (2026-10-26, review fixes 10-27) = v5 + review fixes.  Each is switchable; the v5 behaviour
is in brackets.
Bench: the fq benchmark (make_bench applies the disc channel smear H, the inverse of deghost's G (FQ_SMEAR,
default 0.16,9), and writes dropouts at row fieldLine-1); before, the bench input was too sharp (noise-free
decode luma gain 1-2 MHz 1.16 = 1/|H|) and every bench tuning was biased.
 1. Deghost inverse switchable: PALDEC_DEGHOST=1 [1; 0 skips it] (newdecode + vidsplice; part of the
    dg_cache stamp and of vidsplice's prep.done).
 2. Noise spectrum PALDEC_NOISE=active [vbi]: noisepsd.noise_psd_active - same-phase field differences
    (k, k+8) after deghost+deherring, 16x232 blocks, per 0.54 MHz band the lowest-energy (static) mode;
    each band bounded below by 0.7 x the same selection on single fields (spatial: a stack that smooths a
    held run over time makes the temporal estimate read low), blocks with no 5.5-8 MHz energy (an older
    stacker's flattened black: North panel sigma 60 -> 491) left out; guarded by the blank-line estimate
    (noisepsd.combine: q = active / blank lines capped at 1.75 x its 5.5-8 MHz value, except where the
    temporal and spatial estimates agree within 30 %: North v8 side 505 / 514, the cap alone gave 223).  Blank lines see only the primary master: South v8 stack blank lines sigma 939 vs
    picture 330.  the fq benchmark (true = doc - clean in flat areas): Test Card 363 / 380-394, 16256 364 / 361,
    50112 682 / 671-692, patterns 684 / 743-782; segments without static content (CAV stills) are high:
    08000 711 / 523-633, 29696 771 / 647-669 (blank lines 1070).  The stack's noise really differs between
    segments (5.5-9 MHz, no picture there: 409-781).  Either estimator failing (too few fields or blocks)
    falls back to the other, with a log line (noisepsd.estimate).
 3. noise_psd (blank lines) is a per-bin median and skips fields with dropouts on those lines (all of them
    flagged: uses them anyway).
 4. DC / Nyquist rfft bins weighted once [PALDEC_NW_EDGES=0: twice]; noise_sigma likewise.
    Data weights PALDEC_NW_NORM=inband [mean] (PALDEC_NW=rec only; prod / cap always use the mean): scaled
    to the quietest 0-5 MHz band (PALDEC_NW_REF=min, the default; pNN = the NN-th percentile: p75 gives
    sharper maps but +15 % grain in flat colour and raised map-text error against held-run truth), not to
    the mean over all bins, which the near-empty >7.5 MHz bins
    dominated (bench in-band weights all at the floor, real South clips mostly at the cap: a 4x data/prior
    difference between bench and discs).  PALDEC_KSCALE / PALDEC_KSCALE_C: prior scale knobs (1; bench
    Test Card / 16256: 0.7 and 1.4 both worse in luma).
 5. DOT threshold noise-relative: PALDEC_DOT x sigma / PALDEC_DOT_REF [absolute PALDEC_DOT: PALDEC_DOT_ABS=1;
    always absolute with PALDEC_NOISE=vbi].
 6. solve_held avg mode weights each frame by its dropout mask [PALDEC_HELD_DOW=0].
 7. Windows end at run boundaries (extend to <= PALDEC_WIN_MAX=80 frames, else end where the run starts;
    no piece shorter than 16) [PALDEC_WIN_MAX=64].  A 96-frame held window peaked at 5.3 GB GPU, 64: 3.2.
 8. doi_mask from the unclipped flagged span (v5 lost up to 4 flagged columns at the picture edges).
 9. A guard-fallback frame keeps the masked solve's luma in its holes through DOI [PALDEC_DOI_FB=interp].
10. vidsplice takes the levels from the TBC json (palmodel.set_levels, shared with newdecode).
11. Held-run outlier test (optcore.held_outliers): T x max(n0, min(q, PALDEC_HELD_LOCK x n0)), n0 the global
    median block energy, q the block's lower quartile over the run, LOCK 10, T 5 [PALDEC_HELD_LOCN=0:
    T x n0, T 3].  Saturated / edge blocks have a larger residual in every frame (player chroma errors,
    timing jitter) and were outliers in a fraction of the frames, which then kept their own unaveraged
    solution: grain in saturated colour (South bars 3092 luma rms 10-bit cyan / green / magenta
    20.9 / 16.1 / 8.2 -> 4.3 / 4.3 / 3.1).  The quartile (not the median) and the cap keep a change that
    covers half the run or more flagged.
12. Still / interlaced decision (combscore.classify_interlaced; PALDEC_CLASSIFY=comb: the old comb_score > 0.58
    rule alone).  Static saturated colour, menus and text pages score 0.58-0.67 on comb_score, next to the
    threshold, and anything that lowered the noise tipped them to "interlaced", which loses the held-run
    averaging.  A frame with comb_score > 0.58 is interlaced if its match_score to a same-scene neighbour (luma
    difference on the edge pixels over the same-field prediction error: repeated stills < 0.3, motion >> 1) is
    >= 0.6, or it lies within 4 frames of a strong one (comb_score > 0.70, or > 0.58 with match_score >= 0.7);
    with no same-scene neighbour the old rule stands.  A frame that still looks combed but whose
    low-saturation edges (comb_lowsat, PALDEC_COMB_LOWSAT 0.58, one median per scene) are clean is a still:
    the combing is the colour's.  Static graphics whose two fields really differ stay interlaced.
13. ld-chroma-decoder runs with 4 threads (PALDEC_CHROMA_T; 0 = its default, one per CPU): identical output
    to -t 1 and -t 8, peak RAM per decoder ~5-14 GB -> 1.8 GB.  The default 32 threads differ on long
    inputs (transform3d, North 300 frames: 147 frames, mean 0.6 / max 24 levels), so -t 4 is the reference.

v6dh (2026-10-28, herringbone integration) = v6 + the following (all switchable, defaults as listed).
14. FM-beat herringbone canceller (deherring.py, PALDEC_DEHERRING=model [tiles | both | 0]).  The disc's beat is a
    tone at f_c - f_sc whose phase is that of exp(i(Phi(t) - phi_sc(t))), Phi the FM carrier phase, so it can
    be predicted from the field itself; one complex amplitude per line is fitted over flat parts of the
    picture only (weights fall off at luma / chroma-envelope edges and where 2.2-3.9 MHz band power is high),
    gated on its significance (|c| / standard error), bounded (PALDEC_DH_CMAX 450) and subtracted.  Fields
    without flat saturated colour are returned unchanged.  Carrier at black PALDEC_DH_F0 (7.088 MHz;
    deherring.calibrate() finds it per master).  cancel_tbc_field() is the TBC-domain entry the stacker
    uses on each capture, per master, where the beats of several masters have not yet been mixed.
15. Noise estimate (noisepsd.estimate; PALDEC_NOISE_EST=v60 keeps v6_0's): the temporal estimate is bounded
    below by 0.7 x the spatial one, blocks with no 5.5-8 MHz energy are excluded, the blank-line cap is 1.75
    (PALDEC_NOISE_QCAP) except where temporal and spatial agree within 30 %; either estimator failing falls
    back to the other.  The video and film temporal priors are capped at the capped estimate's strength
    (PALDEC_VID_TCAP, PALDEC_FILM_TCAP): a higher sigma otherwise means more neighbour mixing (motion
    ghosting 0.42 -> 0.51).
16. Held runs: block outlier test against the lower quartile with a cap (PALDEC_HELD_Q=1); a run is not held
    when its vertical flow is not measurable (picture with no vertical detail, e.g. bars: only the horizontal
    component is tested, PALDEC_HELD_APERTURE 0.1), when it drifts more than PALDEC_HELD_DRIFT (0.5 px) over
    the run, or when more than PALDEC_HELD_MOVFRAC (2 %) of its 32 px blocks moved.  A run with local motion on
    a still background (inset video, talking heads) is solved per field as video and its still background is
    then averaged from the film solutions (PALDEC_HELD_LOCAL_VIDEO, PALDEC_LOCAL_HELD, PALDEC_LOCAL_HELD_T).
17. Film runs: a 2-frame run of two different pictures (match_score > PALDEC_FILM2_SPLIT 3.0) is split into two
    stills; the temporal prior is weakened where a frame differs from the motion-compensated previous one by
    far more than noise (optcore.film_tmask, PALDEC_FILM_TM_LO/HI/MIN), so a changing cursor or highlight is not
    averaged with its neighbour.
18. Video guards: the saturation guard falls back to transform3d below init-chroma ratio PALDEC_SATGUARD 0.6
    (ZIP_SATGUARD; was 0.88, which sent most low-chroma real video back to transform3d), testing init chroma
    above PALDEC_SATGUARD_C (6); the ghost guard also falls back to transform3d when the solve's luma leans
    toward its neighbours by more than PALDEC_GHOST_MAX (0.3), unless transform3d would comb more than
    PALDEC_GHOST_GAIN (10 %) worse (PALDEC_GHOST_MAX=0 off).
19. Region-adaptive data weight (PALDEC_NW_ADAPT=1) is OFF by default: the p75 detail weight raised map-text
    error against held-run truth.  The code stays for experiments (PALDEC_NW_ADAPT_REF, PALDEC_NW_FILM_LINEART).
20. stillweave.py --classes <classes file> (the ZIP_DUMP class log nd_chunked.sh writes beside the master): a
    frame the decoder solved as one instant (single / film / held) is woven; only 'inter' frames use the comb
    score.  --expect-frames and the parallel scoring (STILLWEAVE_THREADS, 6) are unchanged.
