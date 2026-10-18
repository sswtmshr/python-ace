"""BFGS for pacemaker fits: scipy's algorithm (same Wolfe line search, same stopping rules) with
  * the O(N^2) rank-2 inverse-Hessian update  H+ = H - rho (s (Hy)^T + (Hy) s^T) + (rho^2 y^T H y + rho) s s^T
    instead of scipy's two dense N x N products (O(N^3)) per iteration;
  * optional checkpointing of (H, x, f, f_old) so a fit cut into short allocations, or restarted at the same loss,
    continues with its curvature instead of H = I;
  * optional pause before an allocation deadline, decided on rank 0 and broadcast (replicated optimisers under MPI
    must stop on the same iteration); a BFGS_PAUSED file is left for driver scripts.

Native BFGS with a checkpoint and a deadline pause (environment variables:
PACE_BFGS_CKPT, PACE_BFGS_EVERY, PACE_DEADLINE, PACE_MARGIN), framework-agnostic (numpy + pyace.mpi).

The inverse Hessian lives on rank 0 only (other ranks receive the search direction, so every rank takes the identical
step) and is updated in place: on the fit's GPU when there is one (torch; N = 9640: 3 ms per iteration), otherwise
with BLAS dger on the CPU (0.14 s; the old numpy outer-product update took 0.52 s and 4 GB of temporaries).

Options (fit: options: {...} in input.yaml, or environment):
  checkpoint:        path of the .npz checkpoint (env PACE_BFGS_CKPT); unset = no checkpoint
  checkpoint_every:  iterations between saves (env PACE_BFGS_EVERY, default 10), and at least
  checkpoint_min_s:  seconds between saves (env PACE_BFGS_MIN_S, default 900; ~0.66 GB/s per writing rank)
  device:            torch device for the inverse Hessian (set by the torch fitter); None = CPU
  bfgs_split:        inverse Hessian rows spread over all MPI ranks, each on its own GPU (default on; false or env
                     PACE_BFGS_SPLIT=0 = the inverse Hessian on rank 0 alone, as before), see bfgs_split.py
  bfgs_split_gpu_only: with bfgs_split, rows that do not fit on the GPUs raise CUDA OOM on every rank (pacemaker then
                     shrinks the batch) instead of falling back to the CPU (default on; false or PACE_BFGS_SPLIT_GPU_ONLY=0 off)
  deadline:          unix time of the allocation end (env PACE_DEADLINE); margin: seconds (env PACE_MARGIN, 240)
  gtol, norm, maxiter: as scipy
"""
import logging
import os
import time

import numpy as np
from scipy.optimize import OptimizeResult

try:  # scipy's combined MINPACK/pure-python Wolfe search (what method="BFGS" uses)
    from scipy.optimize._optimize import _LineSearchError, _line_search_wolfe12
except ImportError:  # pragma: no cover - private API moved: fall back to the public strong-Wolfe search
    from scipy.optimize import line_search as _ls

    class _LineSearchError(RuntimeError):
        pass

    def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kw):
        r = _ls(f, fprime, xk, pk, gfk, old_fval, old_old_fval, amin=kw.get("amin", 1e-100),
                amax=kw.get("amax", 1e100), c1=kw.get("c1", 1e-4), c2=kw.get("c2", 0.9))
        if r[0] is None:
            raise _LineSearchError()
        return r

from pyace import mpi as _mpi

log = logging.getLogger(__name__)
PAUSE_FLAG = "BFGS_PAUSED"


class _Memo:
    """fun(x) -> (f, g); caches the last point (the line search asks for f and g separately)"""

    def __init__(self, fun, args):
        self.fun, self.args = fun, args
        self.x = None
        self.nfev = 0

    def _eval(self, x):
        if self.x is None or not np.array_equal(x, self.x):
            self.x = np.array(x, copy=True)
            f, g = self.fun(self.x, *self.args)
            self.fv, self.gv = float(f), np.asarray(g, dtype=np.float64)
            self.nfev += 1

    def f(self, x):
        self._eval(x)
        return self.fv

    def fprime(self, x):
        self._eval(x)
        return self.gv


def _save(path, H, x, f, f_old, k_total, tag):
    tmp = path + ".tmp.npz"
    np.savez(tmp, H=H.numpy(), x=x, f=f, f_old=f_old, k=k_total, kappa=tag, t=time.time())
    os.replace(tmp, path)


class _InvHessian:
    """N x N inverse Hessian with in-place O(N^2) operations: torch on a GPU, or numpy + BLAS dger on the CPU."""

    def __init__(self, H, N, device=None):
        self.torch = None
        if device is not None and str(device) != "cpu":
            try:
                import torch
                self.torch = torch
                self.dev = torch.device(device)
                self.H = (torch.tensor(H, dtype=torch.float64, device=self.dev) if H is not None
                          else torch.eye(N, dtype=torch.float64, device=self.dev))
                return
            except Exception as e:  # noqa: BLE001 - no torch / no room on the GPU: CPU update
                log.info("BFGS: inverse Hessian on the CPU (%s)", e)
                self.torch = None
        self.H = np.array(H, dtype=np.float64, order="C") if H is not None else np.eye(N)

    def _to_cpu(self, e):
        """GPU out of memory (or any GPU error) on rank 0 alone must not reach fitadapter's collective OOM handling:
        continue on the CPU"""
        log.warning("BFGS: inverse Hessian moved to the CPU (%s)", e)
        self.H = self.H.cpu().numpy()
        self.torch = None

    def dot(self, v):
        if self.torch is not None:
            try:
                return (self.H @ self.torch.as_tensor(v, device=self.dev)).cpu().numpy()
            except RuntimeError as e:
                self._to_cpu(e)
        return self.H.dot(v)

    def finite(self):
        if self.torch is not None:
            return bool(self.torch.isfinite(self.H).all())
        return bool(np.all(np.isfinite(self.H)))

    def update(self, s, y):
        """BFGS: H+ = H - rho (s (Hy)^T + (Hy) s^T) + (rho^2 y^T H y + rho) s s^T  =  H + u s^T + s u^T with
        u = (rho^2 y^T H y + rho) s / 2 - rho H y"""
        ys = float(np.dot(y, s))
        rho = 1000.0 if ys == 0.0 else 1.0 / ys
        if self.torch is not None:
            try:
                st, yt = (self.torch.as_tensor(v, device=self.dev) for v in (s, y))
                Hy = self.H @ yt
                u = 0.5 * (rho * rho * float(self.torch.dot(yt, Hy)) + rho) * st - rho * Hy
                self.H.addr_(u, st).addr_(st, u)
                return
            except RuntimeError as e:   # nothing was modified unless addr_ itself failed midway: recompute on CPU
                self._to_cpu(e)
        from scipy.linalg.blas import dger
        Hy = self.H.dot(y)
        u = 0.5 * (rho * rho * float(np.dot(y, Hy)) + rho) * s - rho * Hy
        Ht = self.H.T                       # Fortran-ordered view of the C-ordered H: dger updates it in place
        dger(1.0, u, s, a=Ht, overwrite_a=1)
        dger(1.0, s, u, a=Ht, overwrite_a=1)

    def numpy(self):
        return self.H.cpu().numpy() if self.torch is not None else self.H


# scipy BFGS options this implementation honours (plus its own); anything else -> use_scipy() says why
_OWN = {"checkpoint", "checkpoint_every", "checkpoint_min_s", "deadline", "margin", "device", "bfgs_split",
        "bfgs_split_gpu_only"}
_SAME_AS_SCIPY = {"gtol", "norm", "maxiter", "disp", "xrtol", "c1", "c2",
                  "eps", "finite_diff_rel_step", "workers"}   # eps/finite_diff/workers: unused, the gradient is given
_paused = [False]


def use_scipy(options):
    """Options of an input written for stock pacemaker (scipy BFGS) that only scipy implements (hess_inv0,
    return_all, ...): the fitters then call scipy.optimize.minimize instead. Returns the offending keys."""
    return sorted(set(options or {}) - _OWN - _SAME_AS_SCIPY)


def was_paused():
    """True if the last minimize_bfgs paused for the allocation deadline (the fit is not finished)"""
    return _paused[0]


def _vecnorm(x, ord=2):   # scipy.optimize._optimize.vecnorm
    if ord == np.inf:
        return np.amax(np.abs(x))
    if ord == -np.inf:
        return np.amin(np.abs(x))
    return np.sum(np.abs(x) ** ord, axis=0) ** (1.0 / ord)


def minimize_bfgs(fun, x0, args=(), callback=None, options=None, tag=np.nan):
    """fun(x, *args) -> (loss, gradient). `tag` identifies the loss (pacemaker: kappa); a checkpoint written for a
    different tag or size is ignored. Returns a scipy OptimizeResult; status/message as scipy's BFGS (0 success,
    1 maxiter, 2 precision loss, 3 NaN) plus 9 = paused for the allocation deadline (was_paused())."""
    o = dict(options or {})
    split = o.pop("bfgs_split", None)                       # default on; bfgs_split: false or PACE_BFGS_SPLIT=0 turns it off
    if split is None:
        split = os.environ.get("PACE_BFGS_SPLIT", "1").lower() not in ("0", "false", "no", "off")
    if bool(split):
        from pyace import bfgs_split   # inverse Hessian rows spread over all MPI ranks (pyace/bfgs_split.py)
        return bfgs_split.minimize_bfgs(fun, x0, args=args, callback=callback, options=o, tag=tag)
    o.pop("bfgs_split_gpu_only", None)
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
    dist = _mpi.is_distributed()
    root = _mpi.is_root()
    _paused[0] = False
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
    H, k_total, f_old_ck = None, 0, None
    # rank 0 alone reads the checkpoint (and holds H)
    if path and root and os.path.exists(path):
        try:
            ck = np.load(path)
            ck_tag = float(ck["kappa"])
            same = (np.isnan(tag) and np.isnan(ck_tag)) or abs(ck_tag - tag) < 1e-12
            if "H" in ck.files:
                Hck = ck["H"]
            else:   # written by the row-split BFGS: index + one part file per rank
                R0 = int(ck["split"])
                ks = ".k%d" % int(ck["kset"]) if "kset" in ck.files else ""
                Hck = np.vstack([np.load("%s%s.part%dof%d.npz" % (path, ks, r, R0))["H"] for r in range(R0)])
            if Hck.shape == (N, N) and same and np.all(np.isfinite(Hck)):
                H = np.array(Hck)
                k_total = int(ck["k"])
                dx = float(np.max(np.abs(ck["x"] - x0)))
                if dx < 1e-10 * (1 + np.max(np.abs(x0))):
                    f_old_ck = float(ck["f_old"])
                log.info("BFGS: inverse Hessian loaded from %s (N=%d, %d iterations so far, max|x - x_ckpt| = %.2e)",
                         path, N, k_total, dx)
            else:
                log.info("BFGS: %s ignored (N %s vs %d, tag %s vs %s, or not finite): H = I", path, Hck.shape, N,
                         ck_tag, tag)
        except Exception as e:  # noqa: BLE001 - unreadable checkpoint: plain start
            log.warning("BFGS: could not read %s (%s): H = I", path, e)
    if dist:
        k_total, f_old_ck = _mpi.bcast((k_total, f_old_ck) if root else None)
    if f_old_ck is not None:
        old_old_fval = f_old_ck
    # rank 0 alone holds and updates H; the other ranks get the search direction from it (identical steps everywhere)
    H = _InvHessian(H, N, device) if root else None
    last_save = time.time()
    if root and os.path.exists(PAUSE_FLAG):
        os.remove(PAUSE_FLAG)

    def save():
        nonlocal last_save
        if path and root and np.isfinite(old_fval) and np.all(np.isfinite(xk)) and H.finite():
            _save(path, H, xk, old_fval, old_old_fval, k_total + k, tag)
            last_save = time.time()

    xk, k, warnflag, paused = x0, 0, 0, False
    gnorm = _vecnorm(gfk, ord=norm)
    try:
        while gnorm > gtol and k < maxiter:
            pk = -H.dot(gfk) if root else None
            if dist:
                pk = _mpi.bcast(pk)
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
            # H update before the callback (scipy order: after it): a checkpoint written on early stopping then holds
            # the curvature of every accepted step; same iterates
            if root and finite and not converged:
                H.update(sk, yk)
            if callback is not None:
                callback(xk)
            if converged:
                break
            if not finite:
                warnflag = 2
                break
            stop = time.time() > deadline - margin
            if dist:
                stop = bool(_mpi.bcast(stop if root else None))
            if stop or (k % every == 0 and time.time() - last_save >= min_s):
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
    except StopIteration:   # early stopping in a callback: keep the curvature, then let the fitter handle it
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
    _paused[0] = paused
    if not paused:
        save()
    if disp and root:
        print(("         " if warnflag == 0 else "Warning: ") + msg)
        print("         Current function value: %f\n         Iterations: %d\n         Function evaluations: %d"
              % (old_fval, k, m.nfev))
    return OptimizeResult(fun=old_fval, jac=gfk, hess_inv=H.numpy() if root else None, nfev=m.nfev, njev=m.nfev,
                          status=warnflag, success=(warnflag == 0), message=msg, x=xk, nit=k)
