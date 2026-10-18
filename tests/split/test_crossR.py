#!/usr/bin/env python
"""T10: split checkpoint written at R ranks, resumed at R' ranks (moving a fit to a different number of GPUs).
  write  (at R):  12 iterations of the N=1001 quartic with a checkpoint -> DIR/ck.npz + R part files;
                  also the uninterrupted 40-iteration run -> DIR/x40_R<R>.npy
  resume (at R'): (a) rows read from the R part files == the gathered H written at R, bitwise;
                  (b) 28 more iterations from the checkpoint == the uninterrupted 40 within summation-order noise
                      (the run switches block shape at iteration 12, so not bitwise: max|dx| <= 1e-8, f to 1e-12)
usage: PACEMAKER_MPI=1 srun -n R python test_crossR.py write DIR ; PACEMAKER_MPI=1 srun -n R2 python test_crossR.py resume DIR R"""
import os
import sys

import numpy as np


from pyace import mpi   # noqa: E402
from pyace import bfgs_split as S  # noqa: E402


def partfile(ck, r, R):
    """part file of rank r as named by the checkpoint index (versioned set)"""
    z = np.load(ck)
    return S._part(ck, r, R, int(z["kset"]) if "kset" in z.files else None)

mode, d = sys.argv[1], os.path.abspath(sys.argv[2])
R, root = mpi.world_size(), mpi.is_root()
N = 1001
w = np.random.default_rng(0).uniform(0.5, 50, size=N)
quart = lambda x: (float(np.sum(w * x ** 2) + 0.25 * np.sum(x ** 4) - np.sum(x)), 2 * w * x + x ** 3 - 1)
o = {"gtol": 1e-12, "norm": 2}
ck = d + "/ck.npz"
if mode == "write":
    if root:
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    mpi.barrier()
    S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=12, checkpoint=ck, checkpoint_min_s=0))
    full = S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=40))
    if root:
        np.save(d + "/x40_R%d.npy" % R, full.x)
        print("T10 write R=%d: parts %s" % (R, sorted(f for f in os.listdir(d) if ".part" in f)), flush=True)
else:
    R0 = int(sys.argv[3])
    cm = S._Comm()
    x12 = np.load(ck)["x"]
    rows, k, _ = S._load(ck, N, np.nan, x12, cm)
    H = S._SplitInvHessian(N, None, cm, rows)
    g = H.full_on_root()
    ok_a = True
    if root:
        Hw = np.vstack([np.load(partfile(ck, r, R0))["H"] for r in range(R0)])
        ok_a = bool(np.array_equal(g, Hw)) and k == 12
    ok_a = bool(mpi.bcast(ok_a))
    # resume reads a copy of the R0 checkpoint (its end-of-run save rewrites parts for R ranks)
    ckc = "%s/ck_at%d.npz" % (d, R)
    if root:
        import shutil
        shutil.copy(ck, ckc)
        for r in range(R0):
            shutil.copy(partfile(ck, r, R0), S._part(ckc, r, R0, int(np.load(ck)["kset"])))
    mpi.barrier()
    res = S.minimize_bfgs(quart, x12, options=dict(o, maxiter=28, checkpoint=ckc))
    x40 = np.load(d + "/x40_R%d.npy" % R0)
    full40 = quart(x40)[0]
    dx = float(np.abs(res.x - x40).max())
    df = abs(res.fun - full40) / max(1, abs(full40))
    ok_b = res.nit == 28 and dx <= 1e-8 and df <= 1e-12
    if root:
        print("T10 resume R=%d from R=%d: %s rows read == H written at R=%d bitwise (k=%d)"
              % (R, R0, "PASS" if ok_a else "FAIL", R0, k), flush=True)
        print("T10 resume R=%d from R=%d: %s 12 + 28 resumed vs 40 uninterrupted: nit %d  max|dx| %.1e  rel df %.1e"
              % (R, R0, "PASS" if ok_b else "FAIL", res.nit, dx, df), flush=True)
mpi.barrier()
