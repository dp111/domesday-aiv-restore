# SPDX-License-Identifier: GPL-3.0-or-later
"""Run any stack.py as a package without touching the production ones.

make_pkg(stacker) copies the candidate stack.py (a frozen copy: the file under test may be
edited while a check runs) into <root>/qcstk_<md5>/qcstk/ beside symlinks to the other lddecode
modules, so it runs as `python -m qcstk.stack`.  The package name is deliberately neither
lddecode_xm3 nor ld-decode: the production scheduler (v7_prio2.sh) SIGSTOPs those.
`stacker` may also be a package directory holding stack.py (its siblings are used).
"""
import hashlib
import os
import shutil

LDBASE = "/mnt/s/domsday/ld-decode/lddecode"     # production modules (tbcmeta, efm_pll, ...)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def make_pkg(stacker, root, ldbase=LDBASE):
    """-> (PYTHONPATH dir, module name, md5 of the stack.py used)."""
    if os.path.isdir(stacker):
        ldbase, stacker = stacker, os.path.join(stacker, "stack.py")
    h = md5(stacker)
    top = os.path.join(root, f"qcstk_{h[:8]}")
    pkg = os.path.join(top, "qcstk")
    if not os.path.exists(os.path.join(pkg, "stack.py")):
        tmp = pkg + ".tmp%d" % os.getpid()
        os.makedirs(tmp, exist_ok=True)
        for n in os.listdir(ldbase):
            if n.endswith(".py") and n != "stack.py":
                os.symlink(os.path.join(os.path.realpath(ldbase), n), os.path.join(tmp, n))
        if not os.path.exists(os.path.join(tmp, "__init__.py")):
            open(os.path.join(tmp, "__init__.py"), "w").close()
        shutil.copy2(stacker, os.path.join(tmp, "stack.py"))
        try:
            os.rename(tmp, pkg)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    return top, "qcstk.stack", h
