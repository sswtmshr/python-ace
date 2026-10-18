"""Row-split inverse Hessian for pyace.bfgs (pace-modern native BFGS), for fits whose N x N inverse Hessian does not fit
on one GPU (or should not crowd rank 0's GPU).

pyace.bfgs keeps H (N x N, float64) on rank 0 alone and broadcasts the search direction. Here every MPI rank r keeps the
contiguous row block H[lo_r:hi_r, :] (np.array_split bounds) on its own device, so H costs N^2 * 8 / R bytes per rank and
GPU 0 no longer limits every rank's batch size. Per BFGS iteration:
  p = -H g      each rank computes its rows; MPI Allgatherv concatenates them (no reduction: every rank holds the same
                bytes, so the replicated line search stays identical on every rank)
  H y           same; then rho, y^T H y and u = (rho^2 y^T H y + rho) s / 2 - rho H y are computed from full vectors,
                identically on every rank
  H += u s^T + s u^T   each rank updates its own rows (the same element-wise operations as pyace.bfgs, row by row)
Communication: two Allgatherv of N doubles per iteration (1.3 MB at N = 162k). One node or several: same code.

Identical to pyace.bfgs except:
  * the row-block storage above (single rank: one block = the whole H, the same operations as pyace.bfgs);
  * device choice is collective: every rank tries its block on its GPU; if any rank fails, all use the CPU
    (a per-rank CPU fallback later, on an error inside an update, only changes that rank's own rows' arithmetic);
  * checkpoint: root writes PATH (x, f, f_old, k, kappa, t, split=R, N) and each rank writes its rows to
    PATH.part<r>of<R>.npz (atomic tmp + rename, barrier before root's file). Resume works for any rank count:
    each rank reads the rows it needs from whichever part files cover them; a pyace.bfgs full-H checkpoint is read on
    root and its rows scattered;
  * OptimizeResult.hess_inv is None when the H is split over more than one rank (pacemaker does not use it).
Default on (pyace.bfgs.minimize_bfgs dispatches here); off: `fit: options: {bfgs_split: false}` or PACE_BFGS_SPLIT=0. GPU only:
default on (`bfgs_split_gpu_only: false` or PACE_BFGS_SPLIT_GPU_ONLY=0 = CPU fallback): if the rows do not fit, a
collective CUDA OOM is raised so pacemaker's batch-size reduction makes room, instead of a silent CPU fallback.
Checkpoint (opt-in, as pyace.bfgs: fit: bfgs_checkpoint / PACE_BFGS_CKPT): saved every checkpoint_every iterations once
checkpoint_min_s (default 900 s) have passed, and at the deadline pause. This module
pyace.bfgs.minimize_bfgs dispatches here."""
import glob
import logging
import os
import time

import numpy as np
from scipy.optimize import OptimizeResult

from pyace import bfgs as _base
from pyace import mpi as _mpi

log = logging.getLogger("pyace.bfgs")
PAUSE_FLAG = _base.PAUSE_FLAG
_line_search_wolfe12, _LineSearchError = _base._line_search_wolfe12, _base._LineSearchError
_Memo, _vecnorm = _base._Memo, _base._vecnorm


_INJECT = {"update_after_gather": None,   # tests only: rank that raises a GPU error after the H y gather
           "place_fail": False}           # tests only: GPU placement fails on every rank
_gpu_fail = [0]                            # consecutive collective GPU placement failures (GPU-only retries)


def _bounds(N, R):
    """row bounds of rank r: np.array_split convention (first N % R blocks one row longer)"""
    q, m = divmod(N, R)
    lo = [r * q + min(r, m) for r in range(R + 1)]
    return lo


class _Comm:
    """the few collectives this needs; identity on a single process"""

    def __init__(self):
        self.c = _mpi._comm() if _mpi.is_distributed() else None
        self.rank = self.c.Get_rank() if self.c else 0
        self.size = self.c.Get_size() if self.c else 1

    def allgatherv(self, local, counts):
        if self.c is None:
            return np.ascontiguousarray(local, dtype=np.float64)
        out = np.empty(sum(counts), dtype=np.float64)
        displ = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        t0 = time.time()
        self.c.Allgatherv(np.ascontiguousarray(local, dtype=np.float64), [out, counts, displ, _mpi._mpi().DOUBLE])
        _mpi.COMM_SECONDS += time.time() - t0
        return out

    def all_true(self, flag):
        if self.c is None:
            return bool(flag)
        return bool(self.c.allreduce(int(bool(flag)), op=_mpi._mpi().MIN))

    def bcast(self, obj):
        return obj if self.c is None else self.c.bcast(obj, root=0)

    def barrier(self):
        if self.c is not None:
            self.c.Barrier()


class _SplitInvHessian:
    """rows lo:hi of the N x N inverse Hessian on this rank: torch on its GPU, or numpy + BLAS dger on the CPU"""

    def __init__(self, N, device, comm, rows=None, gpu_only=False):
        self.N, self.cm = N, comm
        b = _bounds(N, comm.size)
        self.lo, self.hi = b[comm.rank], b[comm.rank + 1]
        self.counts = [b[r + 1] - b[r] for r in range(comm.size)]
        n = self.hi - self.lo
        self.torch = None
        ok = False
        if device is not None and str(device) != "cpu":
            try:
                import torch
                self.torch, self.dev = torch, torch.device(device)
                torch.cuda.empty_cache()                    # cached blocks of the last evaluation count as free
                if _INJECT["place_fail"]:
                    raise RuntimeError("injected GPU placement failure")
                if rows is not None:
                    self.H = torch.tensor(rows, dtype=torch.float64, device=self.dev)
                else:
                    self.H = torch.zeros((n, N), dtype=torch.float64, device=self.dev)
                    self.H[torch.arange(n, device=self.dev), torch.arange(self.lo, self.hi, device=self.dev)] = 1.0
                ok = True
            except Exception as e:  # noqa: BLE001
                log.info("BFGS split: rows %d:%d not on %s (%s)", self.lo, self.hi, device, e)
                self.H = None
        if device is not None and str(device) != "cpu":
            if not comm.all_true(ok):
                self.torch, self.H = None, None
                retries = int(os.environ.get("PACE_BFGS_SPLIT_GPU_RETRIES", 3))
                if gpu_only and _gpu_fail[0] < retries:   # every rank raises the same CUDA OOM: pacemaker shrinks the
                    _gpu_fail[0] += 1                      # batch and retries with the rows still meant for the GPUs
                    import torch
                    torch.cuda.empty_cache()
                    oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
                    raise oom("BFGS split (gpu_only, attempt %d of %d): inverse Hessian rows %d:%d (%.1f GB) do not fit on "
                              "the GPUs next to the batches" % (_gpu_fail[0], retries, self.lo, self.hi, n * N * 8 / 1e9))
                if gpu_only and comm.rank == 0:
                    log.warning("BFGS split: rows still do not fit on the GPUs after %d batch-size reductions: CPU "
                                "(as pyace.bfgs v0.2.5)", _gpu_fail[0])
                elif ok:
                    log.info("BFGS split: another rank could not place its rows on its GPU: every rank uses the CPU")
                _gpu_fail[0] = 0
                ok = False
            else:
                _gpu_fail[0] = 0
        if not ok:
            self.torch = None
            if rows is not None:
                self.H = np.array(rows, dtype=np.float64, order="C")
            else:
                self.H = np.zeros((n, N), dtype=np.float64)
                self.H[np.arange(n), np.arange(self.lo, self.hi)] = 1.0
        log.info("BFGS split: inverse Hessian rows %d:%d of %d on %s (%.2f GB on this rank, %d ranks)", self.lo, self.hi,
                 N, self.dev if self.torch is not None else "cpu", n * N * 8 / 1e9, comm.size)

    def _to_cpu(self, e):
        log.warning("BFGS split: rows %d:%d moved to the CPU (%s)", self.lo, self.hi, e)
        self.H = self.H.cpu().numpy()
        self.torch = None

    def _local_dot(self, v):
        if self.torch is not None:
            try:
                return (self.H @ self.torch.as_tensor(v, device=self.dev)).cpu().numpy()
            except RuntimeError as e:
                self._to_cpu(e)
        return self.H.dot(v)

    def dot(self, v):
        return self.cm.allgatherv(self._local_dot(v), self.counts)

    def finite(self):
        if self.torch is not None:
            f = bool(self.torch.isfinite(self.H).all())
        else:
            f = bool(np.all(np.isfinite(self.H)))
        return self.cm.all_true(f)

    def update(self, s, y):
        """same formula and element operations as pyace.bfgs._InvHessian.update, on this rank's rows"""
        ys = float(np.dot(y, s))
        rho = 1000.0 if ys == 0.0 else 1.0 / ys
        lo, hi = self.lo, self.hi
        from scipy.linalg.blas import dger
        if self.torch is not None:
            try:
                yt = self.torch.as_tensor(y, device=self.dev)
                local = (self.H @ yt).cpu().numpy()
            except RuntimeError as e:
                self._to_cpu(e)
                local = self.H.dot(y)
            Hy = self.cm.allgatherv(local, self.counts)      # the only collective of the update, on every path
            if self.torch is not None:
                try:   # GPU: u on the device and the rank-2 update in place, as pyace.bfgs
                    if _INJECT["update_after_gather"] == self.cm.rank:      # test hook (tests/split): GPU error here
                        _INJECT["update_after_gather"] = None
                        raise RuntimeError("injected GPU error after the H y gather")
                    Hyt, st = self.torch.as_tensor(Hy, device=self.dev), self.torch.as_tensor(s, device=self.dev)
                    u = 0.5 * (rho * rho * float(self.torch.dot(yt, Hyt)) + rho) * st - rho * Hyt
                    self.H.addr_(u[lo:hi], st).addr_(st[lo:hi], u)    # rows of H += u s^T, then += s u^T
                    return
                except RuntimeError as e:   # rows unchanged unless addr_ failed midway (as pyace.bfgs): CPU, same Hy
                    self._to_cpu(e)
        else:
            Hy = self.dot(y)                                # full vector, identical on every rank (collective)
        if hi == lo:                                        # this rank owns no rows (N < ranks): nothing to update
            return
        u = 0.5 * (rho * rho * float(np.dot(y, Hy)) + rho) * s - rho * Hy
        Ht = self.H.T                                       # Fortran view of the C-ordered rows: dger updates in place
        dger(1.0, u, s[lo:hi], a=Ht, overwrite_a=1)         # rows += s u^T   (pyace.bfgs order: this one first)
        dger(1.0, s, u[lo:hi], a=Ht, overwrite_a=1)         # rows += u s^T

    def rows(self):
        return self.H.cpu().numpy() if self.torch is not None else self.H

    def full_on_root(self):
        """whole H on root (single rank: the block itself); None elsewhere"""
        if self.cm.size == 1:
            return self.rows()
        parts = self.cm.c.gather(self.rows(), root=0)
        return np.vstack(parts) if self.cm.rank == 0 else None


def _part(path, r, R, k=None):
    """rank r's rows; k = total iterations of the save (versioned sets, the index names its own; k None = unversioned)"""
    return "%s.part%dof%d.npz" % (path, r, R) if k is None else "%s.k%d.part%dof%d.npz" % (path, k, r, R)


def _save(path, H, x, f, f_old, k_total, tag):
    """parts first (a new versioned set, k_total in the names), then the index that names that set (atomic rename);
    older sets are deleted only after the new index is in place, so the index always matches the parts it names"""
    cm = H.cm
    t0 = time.time()
    final = _part(path, cm.rank, cm.size, k_total)
    np.savez(final + ".tmp.npz", H=H.rows(), lo=H.lo, hi=H.hi, N=H.N)
    os.replace(final + ".tmp.npz", final)
    cm.barrier()                                            # every part of this set on disk before the index names it
    if cm.rank == 0:
        tmp = path + ".tmp.npz"
        np.savez(tmp, x=x, f=f, f_old=f_old, k=k_total, kappa=tag, t=time.time(), split=cm.size, N=H.N, kset=k_total)
        os.replace(tmp, path)
        keep = set(_part(path, r, cm.size, k_total) for r in range(cm.size))
        for old in glob.glob(path + ".part*of*.npz") + glob.glob(path + ".k*.part*of*.npz"):
            if old not in keep:
                os.remove(old)
    cm.barrier()
    if cm.rank == 0:
        log.info("BFGS split: checkpoint saved at iteration %d: %d part files of %.2f GB in %.1f s", k_total, cm.size,
                 (H.hi - H.lo) * H.N * 8 / 1e9, time.time() - t0)


def _load(path, N, tag, x0, cm):
    """-> (rows of this rank or None, k_total, f_old_ck); collective"""
    meta = None
    if cm.rank == 0 and os.path.exists(path):
        try:
            ck = np.load(path)
            ck_tag = float(ck["kappa"])
            same = (np.isnan(tag) and np.isnan(ck_tag)) or abs(ck_tag - tag) < 1e-12
            if "H" in ck.files:
                kind, n_ck = "full", ck["H"].shape[0]
            else:
                kind, n_ck = "split", int(ck["N"])
            dx = float(np.max(np.abs(ck["x"] - x0))) if ck["x"].shape == x0.shape else np.inf
            if n_ck == N and same:
                meta = dict(kind=kind, R=int(ck["split"]) if kind == "split" else 1, k=int(ck["k"]),
                            kset=int(ck["kset"]) if "kset" in ck.files else None,
                            f_old=float(ck["f_old"]) if dx < 1e-10 * (1 + np.max(np.abs(x0))) else None, dx=dx)
            else:
                log.info("BFGS: %s ignored (N %d vs %d, tag %s vs %s): H = I", path, n_ck, N, ck_tag, tag)
        except Exception as e:  # noqa: BLE001
            log.warning("BFGS: could not read %s (%s): H = I", path, e)
    meta = cm.bcast(meta)
    if meta is None:
        return None, 0, None
    b = _bounds(N, cm.size)
    lo, hi = b[cm.rank], b[cm.rank + 1]
    rows = None
    if meta["kind"] == "full":                              # pyace.bfgs checkpoint: root reads, scatters rows
        Hf = None
        if cm.rank == 0:
            try:
                Hf = np.load(path)["H"]
            except Exception as e:  # noqa: BLE001
                log.warning("BFGS: could not read H from %s (%s)", path, e)
        if not cm.all_true(Hf is not None or cm.rank != 0):  # root's outcome decides for everyone (no rank left waiting)
            log.warning("BFGS: %s unreadable: H = I", path) if cm.rank == 0 else None
            return None, 0, None
        if cm.size == 1:
            rows = np.array(Hf)
        else:
            rows = cm.c.scatter([Hf[b[r]:b[r + 1]] for r in range(cm.size)] if cm.rank == 0 else None, root=0)
        del Hf
    else:                                                   # split checkpoint, any rank count: read the parts we overlap
        R0 = meta["R"]
        b0 = _bounds(N, R0)
        try:
            rows = np.empty((hi - lo, N), dtype=np.float64)
            for r0 in range(R0):
                a, z = max(lo, b0[r0]), min(hi, b0[r0 + 1])
                if a < z:
                    part = np.load(_part(path, r0, R0, meta["kset"]))
                    rows[a - lo:z - lo] = part["H"][a - b0[r0]:z - b0[r0]]
        except Exception as e:  # noqa: BLE001 - missing/damaged part file (e.g. only the index was copied)
            log.warning("BFGS: rank %d could not read its rows of %s (%s)", cm.rank, path, e)
            rows = None
        if not cm.all_true(rows is not None):               # every rank votes: any failure -> H = I everywhere
            if cm.rank == 0:
                log.warning("BFGS: %s incomplete or damaged: H = I", path)
            return None, 0, None
    ok = cm.all_true(np.all(np.isfinite(rows)))
    if not ok:
        log.info("BFGS: %s not finite: H = I", path)
        return None, 0, None
    if cm.rank == 0:
        log.info("BFGS: inverse Hessian loaded from %s (%s, N=%d, %d iterations so far, max|x - x_ckpt| = %.2e)",
                 path, "full" if meta["kind"] == "full" else "split over %d" % meta["R"], N, meta["k"], meta["dx"])
    return rows, meta["k"], meta["f_old"]


def minimize_bfgs(fun, x0, args=(), callback=None, options=None, tag=np.nan):
    """pyace.bfgs.minimize_bfgs with the row-split inverse Hessian (see module doc); same options, statuses, pause"""
    o = dict(options or {})
    o.pop("bfgs_split", None)
    gpu_only = o.pop("bfgs_split_gpu_only", None)           # default on (only matters when the fit has a GPU)
    if gpu_only is None:
        gpu_only = os.environ.get("PACE_BFGS_SPLIT_GPU_ONLY", "1").lower() not in ("0", "false", "no", "off")
    gpu_only = bool(gpu_only)
    gtol = float(o.get("gtol", 1e-5))
    norm = o.get("norm", np.inf)
    xrtol = float(o.get("xrtol", 0))
    c1, c2 = float(o.get("c1", 1e-4)), float(o.get("c2", 0.9))
    disp = bool(o.get("disp", False))
    path = o.get("checkpoint", os.environ.get("PACE_BFGS_CKPT"))
    every = max(1, int(o.get("checkpoint_every", os.environ.get("PACE_BFGS_EVERY", 10)) or 1))
    min_s = float(o.get("checkpoint_min_s", os.environ.get("PACE_BFGS_MIN_S", 900)))
    device = o.get("device")
    deadline = float(o.get("deadline", os.environ.get("PACE_DEADLINE", "inf")))
    margin = float(o.get("margin", os.environ.get("PACE_MARGIN", 240)))
    cm = _Comm()
    root = cm.rank == 0
    _base._paused[0] = False
    if np.isfinite(deadline) and not path:
        log.warning("BFGS: PACE_DEADLINE is set but there is no checkpoint (fit: bfgs_checkpoint): the fit will not "
                    "pause before the deadline")
        deadline = float("inf")

    x0 = np.asarray(x0, dtype=np.float64).flatten()
    N = len(x0)
    maxiter = o.get("maxiter")
    maxiter = N * 200 if maxiter is None else int(maxiter)
    m = _Memo(fun, args)
    old_fval = m.f(x0)
    gfk = m.fprime(x0)
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2
    rows, k_total, f_old_ck = (None, 0, None)
    if path:
        rows, k_total, f_old_ck = _load(path, N, tag, x0, cm)
    if f_old_ck is not None:
        old_old_fval = f_old_ck
    H = _SplitInvHessian(N, device, cm, rows, gpu_only=gpu_only)
    del rows
    last_save = time.time()
    if root and os.path.exists(PAUSE_FLAG):
        os.remove(PAUSE_FLAG)

    def save():
        nonlocal last_save
        if path and np.isfinite(old_fval) and np.all(np.isfinite(xk)) and H.finite():
            _save(path, H, xk, old_fval, old_old_fval, k_total + k, tag)
            last_save = time.time()

    xk, k, warnflag, paused = x0, 0, 0, False
    gnorm = _vecnorm(gfk, ord=norm)
    try:
        while gnorm > gtol and k < maxiter:
            pk = -H.dot(gfk)
            try:
                alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(
                    m.f, m.fprime, xk, pk, gfk, old_fval, old_old_fval, amin=1e-100, amax=1e100, c1=c1, c2=c2)
            except _LineSearchError:
                warnflag = 2
                break
            sk = alpha_k * pk
            xk = xk + sk
            if gfkp1 is None:
                gfkp1 = m.fprime(xk)
            yk = gfkp1 - gfk
            gfk = gfkp1
            k += 1
            gnorm = _vecnorm(gfk, ord=norm)
            converged = gnorm <= gtol or alpha_k * _vecnorm(pk) <= xrtol * (xrtol + _vecnorm(xk))
            finite = np.isfinite(old_fval) and np.all(np.isfinite(gfk))
            if finite and not converged:
                H.update(sk, yk)                            # collective (Allgatherv of H y)
            if callback is not None:
                callback(xk)
            if converged:
                break
            if not finite:
                warnflag = 2
                break
            stop = bool(cm.bcast(time.time() > deadline - margin if root else None))
            if stop or (k % every == 0 and bool(cm.bcast(time.time() - last_save >= min_s if root else None))):
                save()
            if stop:
                paused = True
                if root:
                    with open(PAUSE_FLAG, "w") as fh:
                        fh.write("paused at iteration %d of this piece (%d total): deadline %.0f - margin %.0f s\n"
                                 % (k, k_total + k, deadline, margin))
                log.info("BFGS: pausing before the allocation ends (iteration %d, %d total); state saved", k,
                         k_total + k)
                break
    except StopIteration:
        save()
        raise

    if paused:
        warnflag, msg = 9, "paused for allocation end"
    elif warnflag == 2:
        msg = "Desired error not necessarily achieved due to precision loss."
    elif k >= maxiter:
        warnflag, msg = 1, "Maximum number of iterations has been exceeded."
    elif np.isnan(gnorm) or np.isnan(old_fval) or np.isnan(xk).any():
        warnflag, msg = 3, "NaN result encountered."
    else:
        msg = "Optimization terminated successfully."
    _base._paused[0] = paused
    if not paused:
        save()
    if disp and root:
        print(("         " if warnflag == 0 else "Warning: ") + msg)
        print("         Current function value: %f\n         Iterations: %d\n         Function evaluations: %d"
              % (old_fval, k, m.nfev))
    hinv = H.full_on_root() if cm.size == 1 else None
    return OptimizeResult(fun=old_fval, jac=gfk, hess_inv=hinv, nfev=m.nfev, njev=m.nfev,
                          status=warnflag, success=(warnflag == 0), message=msg, x=xk, nit=k)
