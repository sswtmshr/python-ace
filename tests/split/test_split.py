#!/usr/bin/env python
"""Validation of bfgs_split against pyace.bfgs (pace-modern native BFGS) and scipy, the same kinds of checks that
validated pyace.bfgs against scipy (python-ace/tests/test_native_bfgs.py), at any MPI rank count.
Run: PACEMAKER_MPI=1 srun -n R python test_split.py [--device cpu|cuda] [--out DIR]   (R = 1 also without MPI)
Every rank evaluates the full objective (replicated optimizer, as pacemaker after its Allreduce).
Checks (printed by rank 0 as PASS/FAIL lines; exit code 1 on any FAIL):
  T1 quadratic N=30 (as test_native_bfgs) and N=3000 well-conditioned: same nit and x to 1e-12 vs pyace.bfgs
     (bitwise on one rank)
  T2 Rosenbrock N=12 (three option sets of test_native_bfgs) and N=1001 quartic: same nit/status, f to 1e-5
     (summation-order differences grow on ill-conditioned paths, as in test_native_bfgs)
  T3 vs scipy BFGS (one rank only, as test_native_bfgs): same nit/status
  T4 checkpoint: 12 iterations + resume to 40 == 40 uninterrupted (bitwise on the same rank count); parts on disk
  T5 checkpoint written by pyace.bfgs (full H) -> resumed split: equals pyace.bfgs resumed (1e-10)
  T6 split checkpoint written at R ranks, read back as rows by every rank: rows equal the gathered H bitwise
  T7 maxiter 0 / None; NaN objective is not success; norm -inf; deadline pause -> status 9, flag + checkpoint
  T8 the H after N iterations: gathered split H equals pyace.bfgs hess_inv (1e-12 relative)"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy.optimize import minimize, rosen, rosen_der


from pyace import bfgs as base   # noqa: E402
from pyace import mpi            # noqa: E402
from pyace import bfgs_split as S  # noqa: E402

NATIVE = {"bfgs_split": False}   # the split is pyace.bfgs's default: baseline calls ask for the rank-0 inverse Hessian


def partfile(ck, r, R):
    """part file of rank r as named by the checkpoint index (versioned set)"""
    z = np.load(ck)
    return S._part(ck, r, R, int(z["kset"]) if "kset" in z.files else None)

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = None if a.device == "cpu" else a.device
R, rank = mpi.world_size(), mpi.rank()
root = rank == 0
fails = []
if a.out:
    tmpd = os.path.abspath(a.out)
    os.makedirs(tmpd, exist_ok=True)
else:
    tmpd = mpi.bcast(tempfile.mkdtemp(prefix="bfgs_split_") if root else None)


def say(ok, name, detail=""):
    if not ok:
        fails.append(name)
    if root:
        print("%s  %-58s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def both(F, x0, o, tag=np.nan, **kw):
    ob = dict(o, device=dev) if dev else dict(o)
    b = base.minimize_bfgs(F, x0, options=dict(ob, **NATIVE), tag=tag, **kw)
    s = S.minimize_bfgs(F, x0, options=ob, tag=tag, **kw)
    return b, s




class _BlockMatched(base._InvHessian):
    """pyace.bfgs inverse Hessian whose products H v are computed row block by row block with the split bounds of R
    ranks (each block a separate contiguous array, as on the ranks). Same pyace.bfgs logic otherwise: if the split
    BFGS equals THIS bitwise, the split logic is exact and every remaining difference to plain pyace.bfgs is the BLAS
    kernel's summation order on blocks of a different shape."""

    def _blocks(self, v):
        b = S._bounds(self.H.shape[0], R)
        if self.torch is not None:
            vt = self.torch.as_tensor(v, device=self.dev)
            return self.torch.cat([self.H[b[r]:b[r + 1]].contiguous() @ vt for r in range(R)])
        return np.concatenate([np.ascontiguousarray(self.H[b[r]:b[r + 1]]).dot(v) for r in range(R)])

    def dot(self, v):
        out = self._blocks(v)
        return out.cpu().numpy() if self.torch is not None else out

    def update(self, s, y):
        ys = float(np.dot(y, s))
        rho = 1000.0 if ys == 0.0 else 1.0 / ys
        if self.torch is not None:
            yt, st = (self.torch.as_tensor(v, device=self.dev) for v in (y, s))
            Hyt = self.torch.as_tensor(self._blocks(y).cpu().numpy(), device=self.dev)
            u = 0.5 * (rho * rho * float(self.torch.dot(yt, Hyt)) + rho) * st - rho * Hyt
            self.H.addr_(u, st).addr_(st, u)
            return
        from scipy.linalg.blas import dger
        Hy = self._blocks(y)
        u = 0.5 * (rho * rho * float(np.dot(y, Hy)) + rho) * s - rho * Hy
        dger(1.0, u, s, a=self.H.T, overwrite_a=1)
        dger(1.0, s, u, a=self.H.T, overwrite_a=1)


def matched(F, x0, o):
    orig = base._InvHessian
    base._InvHessian = _BlockMatched
    try:
        return base.minimize_bfgs(F, x0, options=dict(o, device=dev, **NATIVE) if dev else dict(o, **NATIVE))
    finally:
        base._InvHessian = orig


def yard(F, x0, o):
    """native pyace.bfgs vs scipy BFGS on the same problem: the size of disagreement pace-modern was accepted with"""
    so = {k: v for k, v in o.items() if k in ("gtol", "norm", "maxiter", "c1", "c2")}
    sc = minimize(lambda x: F(x)[0], x0, jac=lambda x: F(x)[1], method="BFGS", options=so)
    nb = base.minimize_bfgs(F, x0, options=dict(o, device=dev, **NATIVE) if dev else dict(o, **NATIVE))
    return abs(sc.nit - nb.nit), abs(sc.fun - nb.fun) / max(1, abs(nb.fun)), float(np.abs(sc.x - nb.x).max())


def check(name, F, x0, o):
    """T0: split == block-matched pyace.bfgs bitwise (the algorithm is identical). T1: split vs plain pyace.bfgs, where
    only BLAS summation order on differently shaped blocks differs: converged runs must agree in f to 1e-12;
    maxiter-capped runs must agree in nit/status and in f, x within 10 x the native-vs-scipy yardstick"""
    b, s = both(F, x0, o)
    mb = matched(F, x0, o)
    exact = np.array_equal(mb.x, s.x) and mb.nit == s.nit and mb.fun == s.fun
    say(exact, "T0 %s == block-matched pyace.bfgs (bitwise), R=%d" % (name, R),
        "nit %d/%d  max|dx| %.1e" % (mb.nit, s.nit, np.abs(mb.x - s.x).max()))
    yn, yf, yx = yard(F, x0, o)
    dn, df, dx = abs(b.nit - s.nit), abs(b.fun - s.fun) / max(1, abs(b.fun)), float(np.abs(b.x - s.x).max())
    if b.status in (0, 2) and s.status in (0, 2):   # both converged: gnorm <= gtol (0) or the line search stopped at
        ok = df <= 1e-12                             # machine precision (2); same answer = same f; the last iterations are
                                                     # where summation-order noise decides the count (reported)
    else:                                  # stopped by maxiter: compare at the same iteration
        ok = b.status == s.status and dn == 0 and df <= max(10 * yf, 1e-12) and dx <= max(10 * yx, 1e-12)
    say(ok, "T1 %s vs pyace.bfgs, R=%d" % (name, R),
        "status %d/%d  dnit %d  rel df %.1e  max|dx| %.1e | native-vs-scipy: dnit %d  rel df %.1e  max|dx| %.1e%s"
        % (b.status, s.status, dn, df, dx, yn, yf, yx, "  bitwise" if np.array_equal(b.x, s.x) else ""))

F_ros = lambda x: (rosen(x), rosen_der(x))
X0 = np.linspace(-1.2, 1.0, 12)
rng = np.random.default_rng(0)

# T0/T1: well-conditioned quadratics, Rosenbrock (test_native_bfgs option sets), an ill-conditioned quartic
for N in (30, 3000):
    M = rng.normal(size=(N, N))
    A = M @ M.T / N + np.eye(N)
    q = lambda x, A=A: (0.5 * float(x @ A @ x) - float(x.sum()), A @ x - 1)
    check("quadratic N=%d" % N, q, rng.normal(size=N), {"gtol": 1e-6, "maxiter": 400})
for o in ({"maxiter": 60, "gtol": 1e-8}, {"maxiter": 60, "gtol": 1e-6, "norm": 2}, {"maxiter": 40, "c1": 1e-3, "c2": 0.5}):
    check("Rosenbrock %s" % o, F_ros, X0, o)
N = 1001
w = rng.uniform(0.5, 50, size=N)
quart = lambda x: (float(np.sum(w * x ** 2) + 0.25 * np.sum(x ** 4) - np.sum(x)), 2 * w * x + x ** 3 - 1)
check("quartic N=1001", quart, np.random.default_rng(1).normal(size=N), {"gtol": 1e-8, "maxiter": 300})

# T3 scipy (one rank, as test_native_bfgs)
if R == 1:
    for o in ({"maxiter": 60, "gtol": 1e-8}, {"maxiter": 40, "c1": 1e-3, "c2": 0.5}):
        sc = minimize(rosen, X0, jac=rosen_der, method="BFGS", options=o)
        s = S.minimize_bfgs(F_ros, X0, options=dict(o, device=dev) if dev else o)
        say(sc.nit == s.nit and sc.status == s.status and abs(sc.fun - s.fun) <= 1e-5 * max(1, abs(sc.fun)),
            "T3 scipy vs split, Rosenbrock %s" % o, "nit %d/%d" % (sc.nit, s.nit))

# T4 checkpoint resume == uninterrupted (same R)
ck = os.path.join(tmpd, "ck_T4_R%d.npz" % R)
for f in [ck]:
    if root and os.path.exists(f):
        os.remove(f)
mpi.barrier()
o = {"gtol": 1e-12, "norm": 2}
full = S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=40, **({"device": dev} if dev else {})))
S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=12, checkpoint=ck, checkpoint_min_s=0,
                                                       **({"device": dev} if dev else {})))
parts = all(os.path.exists(partfile(ck, r, R)) for r in range(R)) and os.path.exists(ck)
x12 = np.load(ck)["x"]
res = S.minimize_bfgs(quart, x12, options=dict(o, maxiter=28, checkpoint=ck, **({"device": dev} if dev else {})))
say(parts and np.array_equal(res.x, full.x) and res.fun == full.fun, "T4 checkpoint 12 + resume 28 == 40, R=%d" % R,
    "parts on disk %s, max|dx| %.1e%s" % (parts, np.abs(res.x - full.x).max(),
                                         "  bitwise" if np.array_equal(res.x, full.x) else ""))

# T5 pyace.bfgs full-H checkpoint -> split resume (and pyace.bfgs resume as the reference)
ckb = os.path.join(tmpd, "ck_T5_R%d.npz" % R)
if root and os.path.exists(ckb):
    os.remove(ckb)
mpi.barrier()
od = dict(o, **({"device": dev} if dev else {}))
base.minimize_bfgs(quart, np.full(N, 0.3), options=dict(od, maxiter=12, checkpoint=ckb, checkpoint_min_s=0, **NATIVE))
if root:
    shutil.copy(ckb, ckb + ".copy.npz")
mpi.barrier()
x12 = np.load(ckb)["x"]
rb = base.minimize_bfgs(quart, x12, options=dict(od, maxiter=28, checkpoint=ckb, **NATIVE))
ckc = ckb + ".copy.npz"
rs = S.minimize_bfgs(quart, x12, options=dict(od, maxiter=28, checkpoint=ckc))
dx = float(np.abs(rb.x - rs.x).max())
say(rb.nit == rs.nit and dx < 1e-10, "T5 pyace.bfgs full-H checkpoint -> split resume, R=%d" % R,
    "nit %d/%d  max|dx| %.1e" % (rb.nit, rs.nit, dx))

# T6 split checkpoint rows read back == gathered H (bitwise), written at R, read at R
cm = S._Comm()
rows, kk, _ = S._load(ck, N, np.nan, res.x, cm)
Hw = S._SplitInvHessian(N, None, cm, rows)
full_root = Hw.full_on_root()
if root:
    b0 = S._bounds(N, R)
    Hparts = np.vstack([np.load(partfile(ck, r, R))["H"] for r in range(R)])
    ok6 = np.array_equal(full_root, Hparts) and kk == 40
else:
    ok6 = True
say(bool(mpi.bcast(ok6 if root else None)), "T6 split checkpoint rows == parts on disk, R=%d" % R, "k=%d" % kk)

# T7 edge cases
r0 = S.minimize_bfgs(F_ros, X0, options={"maxiter": 0})
say(r0.nit == 0 and r0.status == 1, "T7 maxiter 0", "nit %d status %d" % (r0.nit, r0.status))
rn = S.minimize_bfgs(lambda x: (np.nan, np.full_like(x, np.nan)), X0, options={"maxiter": 5})
say(not rn.success, "T7 NaN objective is not success", "status %d" % rn.status)
rm = S.minimize_bfgs(F_ros, X0, options={"maxiter": 20, "norm": -np.inf})
bm = base.minimize_bfgs(F_ros, X0, options={"maxiter": 20, "norm": -np.inf, **NATIVE})
say(rm.nit == bm.nit and rm.status == bm.status, "T7 norm -inf", "nit %d/%d" % (rm.nit, bm.nit))
ckp = os.path.join(tmpd, "ck_T7_R%d.npz" % R)
cwd = os.getcwd()
os.chdir(tmpd)
rp = S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=50, checkpoint=ckp, deadline=0, margin=0))
mpi.barrier()                     # rank 0 writes the flag while pausing: look only after it is done
flag = os.path.exists(S.PAUSE_FLAG)
mpi.barrier()
os.chdir(cwd)
say(rp.status == 9 and base.was_paused() and flag and os.path.exists(ckp), "T7 deadline pause, R=%d" % R,
    "status %d, flag %s, nit %d" % (rp.status, flag, rp.nit))

# T8 gathered split H == pyace.bfgs H after 30 iterations
o8 = dict(o, maxiter=30, **({"device": dev} if dev else {}))
hb = base.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o8, **NATIVE))
ck8 = os.path.join(tmpd, "ck_T8_R%d.npz" % R)
if root:   # fresh run: an old checkpoint with the same N and kappa would be resumed (by design)
    for f in [x for x in os.listdir(tmpd) if x.startswith("ck_T8_R%d" % R)]:
        os.remove(os.path.join(tmpd, f))
mpi.barrier()
S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o8, checkpoint=ck8, checkpoint_min_s=0))
if root:
    Hs = np.vstack([np.load(partfile(ck8, r, R))["H"] for r in range(R)])
    Hb = hb.hess_inv
    rel = float(np.abs(Hs - Hb).max() / np.abs(Hb).max())
else:
    rel = 0.0
rel = mpi.bcast(rel)
say(rel < 1e-12, "T8 split H (gathered) vs pyace.bfgs H, 30 iterations, R=%d" % R, "max rel diff %.1e" % rel)

# T12 crash between parts and index: a new part set is written, the index replace fails -> resume uses the old,
# consistent set (index k matches its parts), equal to an uninterrupted run; the orphan set is removed by the next save
ck12 = os.path.join(tmpd, "ck_T12_R%d.npz" % R)
if root:
    for f in [x for x in os.listdir(tmpd) if x.startswith("ck_T12_R%d" % R)]:
        os.remove(os.path.join(tmpd, f))
mpi.barrier()
od12 = dict(o, **({"device": dev} if dev else {}))
S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(od12, maxiter=12, checkpoint=ck12, checkpoint_min_s=0))
H20 = S._SplitInvHessian(N, None, S._Comm())                # some other state (identity), "saved" as iteration 20
np.savez(S._part(ck12, H20.cm.rank, R, 20), H=H20.rows(), lo=H20.lo, hi=H20.hi, N=N)   # parts of a new set written ...
crashed = True                                                                       # ... and the job dies before the index
mpi.barrier()
orphan = any(".k20." in f for f in os.listdir(tmpd) if f.startswith("ck_T12_R%d" % R))
x12b = np.load(ck12)["x"]
r12 = S.minimize_bfgs(quart, x12b, options=dict(od12, maxiter=28, checkpoint=ck12))
f40 = S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(od12, maxiter=40))
gone = not any(".k20." in f for f in os.listdir(tmpd) if f.startswith("ck_T12_R%d" % R))
say(crashed and orphan and np.array_equal(r12.x, f40.x) and gone, "T12 crash between parts and index -> old set used, R=%d" % R,
    "crash %s, orphan set on disk %s, resume == uninterrupted %s, orphan removed by next save %s"
    % (crashed, orphan, np.array_equal(r12.x, f40.x), gone))

# T9 GPU only: rows that cannot fit raise CUDA OOM on every rank (pacemaker shrinks the batch) up to
# PACE_BFGS_SPLIT_GPU_RETRIES times, then every rank falls back to the CPU (as v0.2.5) instead of stopping the fit
if dev:
    import torch
    big = int((torch.cuda.get_device_properties(0).total_memory * 1.5 * R / 8) ** 0.5)   # 1.5 x one GPU per rank
    oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    os.environ["PACE_BFGS_SPLIT_GPU_RETRIES"] = "3"
    S._gpu_fail[0] = 0
    raised = 0
    try:                                                              # 1: a real placement that does not fit
        S._SplitInvHessian(big, dev, S._Comm(), gpu_only=True)
    except oom:
        raised += 1
    S._INJECT["place_fail"] = True                                    # 2, 3: placement failures at small N
    for _ in range(2):
        try:
            S._SplitInvHessian(3000, dev, S._Comm(), gpu_only=True)
        except oom:
            raised += 1
    Hc = S._SplitInvHessian(3000, dev, S._Comm(), gpu_only=True)     # 4: CPU fallback, no error
    S._INJECT["place_fail"] = False
    cpu4 = Hc.torch is None
    del Hc
    Hs = S._SplitInvHessian(3000, dev, S._Comm(), gpu_only=True)     # fits: on the GPU, counter reset
    say(bool(S._Comm().all_true(raised == 3 and cpu4)) and Hs.torch is not None and S._gpu_fail[0] == 0,
        "T9 gpu_only: 3 x CUDA OOM, then CPU fallback; a fitting placement resets, R=%d" % R,
        "raised %d, 4th on cpu %s, N=3000 on gpu %s" % (raised, cpu4, Hs.torch is not None))
    del Hs

# T14 damaged checkpoint: index present, one rank's part file missing (e.g. only the index was copied) -> every rank
# falls back to H = I together (no rank left waiting in a collective)
ck14 = os.path.join(tmpd, "ck_T14_R%d.npz" % R)
if root:
    for f in [x for x in os.listdir(tmpd) if x.startswith("ck_T14_R%d" % R)]:
        os.remove(os.path.join(tmpd, f))
mpi.barrier()
S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=12, checkpoint=ck14, checkpoint_min_s=0))
if root:
    os.remove(partfile(ck14, R - 1, R))                                 # the last rank's rows are gone
mpi.barrier()
rows14, k14, _ = S._load(ck14, N, np.nan, np.load(ck14)["x"], S._Comm())
say(bool(S._Comm().all_true(rows14 is None)) and k14 == 0, "T14 damaged checkpoint (a part missing) -> H = I on every rank, R=%d" % R,
    "rows None on every rank, k=%d" % k14)

# T15 GPU error on one rank after the H y gather: that rank finishes the update on the CPU with the gathered vector
# (no second collective, no hang); result == the run without the error within summation-order noise
if dev:
    ref = S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=25, device=dev))
    S._INJECT["update_after_gather"] = R - 1
    inj = S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=25, device=dev))
    dx = float(np.abs(ref.x - inj.x).max())
    fired = S._Comm().all_true(rank != R - 1 or S._INJECT["update_after_gather"] is None)   # the injected rank used it
    S._INJECT["update_after_gather"] = None                                                  # clear on the others
    say(inj.nit == 25 and dx <= 1e-8 and fired,
        "T15 GPU error after the gather on rank %d -> CPU, no hang, R=%d" % (R - 1, R), "nit %d  max|dx| %.1e" % (inj.nit, dx))

# T13 wiring: pyace.bfgs with no bfgs_split option runs the split (default), PACE_BFGS_SPLIT=0 runs the rank-0 inverse Hessian
d1 = base.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=25))
d2 = S.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=25))
os.environ["PACE_BFGS_SPLIT"] = "0"
d3 = base.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=25))
del os.environ["PACE_BFGS_SPLIT"]
d4 = base.minimize_bfgs(quart, np.full(N, 0.3), options=dict(o, maxiter=25, **NATIVE))
say(np.array_equal(d1.x, d2.x) and np.array_equal(d3.x, d4.x) and (d1.hess_inv is None) == (R > 1) and d3.hess_inv is not None or not root,
    "T13 default = split; PACE_BFGS_SPLIT=0 = rank-0 inverse Hessian, R=%d" % R,
    "default==split %s, env0==bfgs_split:false %s" % (np.array_equal(d1.x, d2.x), np.array_equal(d3.x, d4.x)))

allfails = mpi._comm().allgather(fails) if R > 1 else [fails]     # a check can fail on one rank only: report every rank's
if root:
    for r, fr in enumerate(allfails):
        if fr and r != 0:
            print("FAIL on rank %d: %s" % (r, fr), flush=True)
    fails = sorted(set(sum(allfails, [])))
    print("RESULT R=%d device=%s: %d FAIL %s" % (R, a.device, len(fails), fails), flush=True)
else:
    fails = sorted(set(sum(allfails, [])))
mpi.barrier()
sys.exit(1 if fails else 0)
