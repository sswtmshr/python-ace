#!/usr/bin/env python
"""T11 (patched pyace.bfgs): the unsplit BFGS resumes from a checkpoint written by the split BFGS.
12 split iterations with a checkpoint (R ranks) -> 28 unsplit iterations from it == 40 unsplit uninterrupted
(bitwise at R=1; summation-order noise otherwise: max|dx| <= 1e-8, f to 1e-12). usage: [srun -n R] python harness.py THIS DIR"""
import os
import sys

import numpy as np
from pyace import bfgs as base
from pyace import mpi

d = os.path.abspath(sys.argv[1])
R, root = mpi.world_size(), mpi.is_root()
if root:
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
mpi.barrier()
N = 1001
w = np.random.default_rng(0).uniform(0.5, 50, size=N)
quart = lambda x: (float(np.sum(w * x ** 2) + 0.25 * np.sum(x ** 4) - np.sum(x)), 2 * w * x + x ** 3 - 1)
o = {"gtol": 1e-12, "norm": 2}
ck = d + "/ck.npz"
base.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=12, checkpoint=ck, checkpoint_min_s=0, bfgs_split=True))
split_written = "H" not in np.load(ck).files
full = base.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=40, bfgs_split=False))
res = base.minimize_bfgs(quart, np.load(ck)["x"], options=dict(o, maxiter=28, checkpoint=ck, bfgs_split=False))
dx = float(np.abs(res.x - full.x).max())
df = abs(res.fun - full.fun) / max(1, abs(full.fun))
ok = split_written and res.nit == 28 and dx <= 1e-8 and df <= 1e-12 and (R > 1 or np.array_equal(res.x, full.x))
if root:
    print("%s  T11 unsplit resumes from a split checkpoint, R=%d: split format %s, nit %d, max|dx| %.1e, rel df %.1e%s"
          % ("PASS" if ok else "FAIL", R, split_written, res.nit, dx, df, "  bitwise" if np.array_equal(res.x, full.x) else ""),
          flush=True)
