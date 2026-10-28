# SPDX-License-Identifier: GPL-3.0-or-later
"""Picture index of each capture, exactly as the stacker builds it (its own TBCFrameSource:
picture-number decode + sequence sieve), and the stacker's per-side analysis sample.

  index_caps.py <out.json> <stacker stack.py> <capture base>...

out.json: {"caps": {name: {"base", "keys": [...], "fi": [...]}}, "analysis": [24 keys], "span": n}
"""
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcpkg import make_pkg   # noqa: E402

SAMPLE = 24


def main():
    out, stacker, bases = sys.argv[1], sys.argv[2], sys.argv[3:]
    top, mod, _ = make_pkg(stacker, os.path.join(os.path.dirname(os.path.abspath(out)), "pkg"))
    sys.path.insert(0, top)
    stk = importlib.import_module(mod)
    caps = {}
    for b in bases:
        t0 = time.time()
        s = stk.TBCFrameSource(b)
        s._build_index()
        ks = sorted(s._idx)
        caps[s.name] = {"base": b, "keys": ks, "fi": [s._idx[k] for k in ks],
                        "nframes": len(s.fields) // 2}
        s.close()
        print(f"{s.name}: {len(ks)} pictures {ks[0]}-{ks[-1]} ({time.time() - t0:.0f}s)", flush=True)
    keysets = [set(c["keys"]) for c in caps.values()]
    # same pick as stack(): evenly over the pictures every capture carries
    span = sorted(set.intersection(*keysets)) or sorted(set.union(*keysets))
    pick = span[::max(1, len(span) // SAMPLE)][:SAMPLE]
    with open(out, "w") as f:
        json.dump({"caps": caps, "analysis": pick, "span": len(span)}, f)
    print("analysis sample:", pick)


if __name__ == "__main__":
    main()
