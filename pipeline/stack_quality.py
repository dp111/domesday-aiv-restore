#!/usr/bin/env python3
"""Predicted noise of a stack from its stack.py log.

stack.py measures each capture's noise as its deviation from the cross-capture
median in flat picture areas, so noise recorded on the disc itself (common to
every capture - Community South's 4-7 MHz master noise correlates 0.85-0.94
between independent captures) cancels out and only the noise that stacking
removes is counted.  With inverse-variance weights the stack's noise is
(sum 1/sigma_i^2)^-1/2 over the PRIMARY captures.  On the Oct 2026 Community
trial this predicted 152.4 -> 103.5 for South 4 -> 9 captures, against a
measured split-half 147.6 -> 104.2.

2026-10-26: the noise is in 16-bit TBC units, so stacks decoded with different
levels (e.g. --vits_white_level) do not compare directly.  Given the stack's
.tbc.json (LOG=JSON), the noise is scaled to the nominal white-black range
(54016 - 16384) using the json's white16bIre - black16bIre.  Two stacks are
only normalised when both jsons are readable.

  stack_quality.py LOG[=JSON] [LOG[=JSON] ...]   print the predicted noise per log
  stack_quality.py --gate OLD[=JSON] NEW[=JSON]  exit 0 only if NEW is >=5% quieter
"""
import json
import re
import sys

NOMINAL = 54016 - 16384


def level_range(js):
    """white - black of a .tbc.json, or None."""
    try:
        vp = json.load(open(js))["videoParameters"]
        r = int(vp["white16bIre"]) - int(vp["black16bIre"])
        return r if r > 0 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def predicted(log, js=None):
    """(noise, captures); noise normalised to the nominal range if js gives the stack's levels."""
    sig = [float(m.group(1)) for m in
           re.finditer(r"noise=([\d.]+) master=\d+ PRIMARY", open(log).read())]
    if not sig:
        raise SystemExit(f"no PRIMARY captures in {log}")
    v = sum(s ** -2 for s in sig) ** -0.5
    r = level_range(js) if js else None
    return (v * NOMINAL / r if r else v), len(sig)


def split(arg):
    log, _, js = arg.partition("=")
    return log, js or None


def main(argv):
    if argv[:1] == ["--gate"]:
        (ol, oj), (nl, nj) = split(argv[1]), split(argv[2])
        norm = bool(oj and nj and level_range(oj) and level_range(nj))
        (o, no), (n, nn) = predicted(ol, oj if norm else None), predicted(nl, nj if norm else None)
        ok = n < 0.95 * o
        print(f"old {o:.1f} ({no} captures)  new {n:.1f} ({nn} captures)  "
              f"change {100 * (n / o - 1):+.0f}%  [{'level-normalised' if norm else 'NOT level-normalised'}]"
              f"  -> {'NEW BETTER' if ok else 'NEW NOT BETTER'}")
        return 0 if ok else 1
    args = [split(a) for a in argv]
    norm = bool(args) and all(j and level_range(j) for _, j in args)
    for log, js in args:
        v, k = predicted(log, js if norm else None)
        r = level_range(js) if js else None
        print(f"{v:7.1f}  ({k} captures)  {log}" + (f"  [levels {r}, normalised]" if norm else
              ("  [NOT level-normalised: " + ("no json given" if not js else "json unreadable" if not r
                                               else "another json unreadable") + "]") if any(j for _, j in args) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
