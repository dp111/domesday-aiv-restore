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

  stack_quality.py LOG [LOG ...]       print the predicted noise per log
  stack_quality.py --gate OLD NEW      exit 0 only if NEW is >=5% quieter
"""
import re
import sys


def predicted(log):
    sig = [float(m.group(1)) for m in
           re.finditer(r"noise=([\d.]+) master=\d+ PRIMARY", open(log).read())]
    if not sig:
        raise SystemExit(f"no PRIMARY captures in {log}")
    return sum(s ** -2 for s in sig) ** -0.5, len(sig)


def main(argv):
    if argv[:1] == ["--gate"]:
        (o, no), (n, nn) = predicted(argv[1]), predicted(argv[2])
        ok = n < 0.95 * o
        print(f"old {o:.1f} ({no} captures)  new {n:.1f} ({nn} captures)  "
              f"change {100 * (n / o - 1):+.0f}%  -> {'NEW BETTER' if ok else 'NEW NOT BETTER'}")
        return 0 if ok else 1
    for p in argv:
        v, k = predicted(p)
        print(f"{v:7.1f}  ({k} captures)  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
