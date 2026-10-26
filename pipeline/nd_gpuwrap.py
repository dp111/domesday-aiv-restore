#!/usr/bin/env python3
"""Run newdecode/newdecode.py with a GPU-memory workaround (2026-10-18); no change to its maths.

  nd_gpuwrap.py /path/newdecode.py <newdecode args...>

North died on 'cuFFT error: CUFFT_INTERNAL_ERROR' in optcore.estimate_flow_gpu (held_static on
the window from frame 832; reproducible in a fresh 960-frame run): PyTorch's caching allocator held
9.0 GB (1 GB in use), leaving ~2 GB on the 12 GB card, and cuFFT could not build the plan for the
large batched FFT outside that cache.  Here estimate_flow_gpu first returns the unused cache to the
driver (torch.cuda.empty_cache), and on a cuFFT error clears the plan cache and retries once.
ND_MEMLOG=1 logs GPU memory every 20 s to stderr.
ND_FORCE_T3=1 always starts the transform3d decoder: newdecode starts it only when the run has
video frames, so a chunk with a lone interlaced frame but no video left it unsolved (2D decode);
a full-side run always has video somewhere, so this gives the full-run result.
"""
import os, sys, threading, time
import torch

script = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(script))
import optcore

_flow = optcore.estimate_flow_gpu
STATS = {'retries': 0}


def estimate_flow_gpu(*a, **k):
    torch.cuda.empty_cache()
    try:
        return _flow(*a, **k)
    except RuntimeError as e:
        if 'cuFFT' not in str(e):
            raise
        STATS['retries'] += 1
        print('[gpuwrap] %s - clearing caches, retry %d' % (e, STATS['retries']), file=sys.stderr, flush=True)
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        return _flow(*a, **k)


optcore.estimate_flow_gpu = estimate_flow_gpu

if os.environ.get('ND_MEMLOG'):
    def _mon():
        while True:
            time.sleep(20)
            try:
                f, _ = torch.cuda.mem_get_info()
                print('[mem] alloc %.2f reserved %.2f free %.2f GB fftplans %d' % (
                    torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9, f / 1e9,
                    torch.backends.cuda.cufft_plan_cache.size), file=sys.stderr, flush=True)
            except Exception as e:
                print('[mem] err', e, file=sys.stderr, flush=True)
    threading.Thread(target=_mon, daemon=True).start()

sys.argv = [script] + sys.argv[2:]
src = open(script).read()
if os.environ.get('ND_FORCE_T3'):
    old = "p3 = Chroma(['-f', 'transform3d']) if video.any() else None"
    assert src.count(old) == 1, 'newdecode.py changed: p3 line not found'
    src = src.replace(old, "p3 = Chroma(['-f', 'transform3d'])")
import types
m = types.ModuleType('__main__'); m.__file__ = script
sys.modules['__main__'] = m          # the process pool pickles functions by __main__.<name>
exec(compile(src, script, 'exec'), m.__dict__)
