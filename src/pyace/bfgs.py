"""BFGS for pacemaker fits: scipy's algorithm (same Wolfe line search, same stopping rules) with
  * the O(N^2) rank-2 inverse-Hessian update  H+ = H - rho (s (Hy)^T + (Hy) s^T) + (rho^2 y^T H y + rho) s s^T
    instead of scipy's two dense N x N products (O(N^3)) per iteration;
  * optional checkpointing of (H, x, f, f_old) so a fit cut into short allocations, or restarted at the same loss,
    continues with its curvature instead of H = I;
  * optional pause before an allocation deadline, decided on rank 0 and broadcast (replicated optimisers under MPI
    must stop on the same iteration); a BFGS_PAUSED file is left for driver scripts.

Native BFGS with a checkpoint and a deadline pause (environment variables:
PACE_BFGS_CKPT, PACE_BFGS_EVERY, PACE_DEADLINE, PACE_MARGIN), framework-agnostic (numpy + pyace.mpi).

Options (fit: options: {...} in input.yaml, or environment):
  checkpoint:        path of the .npz checkpoint (env PACE_BFGS_CKPT); unset = no checkpoint
  checkpoint_every:  iterations between saves (env PACE_BFGS_EVERY, default 10)
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
                amax=kw.get("amax", 1e100))
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
    np.savez(tmp, H=H, x=x, f=f, f_old=f_old, k=k_total, kappa=tag, t=time.time())
    os.replace(tmp, path)


def minimize_bfgs(fun, x0, args=(), callback=None, options=None, tag=np.nan):
    """fun(x, *args) -> (loss, gradient). `tag` identifies the loss (pacemaker: kappa); a checkpoint written for a
    different tag or size is ignored. Returns a scipy OptimizeResult."""
    o = dict(options or {})
    gtol = float(o.get("gtol", 1e-5))
    norm = o.get("norm", np.inf)
    path = o.get("checkpoint", os.environ.get("PACE_BFGS_CKPT"))
    every = int(o.get("checkpoint_every", os.environ.get("PACE_BFGS_EVERY", 10)))
    deadline = float(o.get("deadline", os.environ.get("PACE_DEADLINE", "inf")))
    margin = float(o.get("margin", os.environ.get("PACE_MARGIN", 240)))
    dist = _mpi.is_distributed()
    root = _mpi.is_root()

    x0 = np.asarray(x0, dtype=np.float64).flatten()
    N = len(x0)
    maxiter = int(o.get("maxiter") or N * 200)
    m = _Memo(fun, args)
    old_fval = m.f(x0)
    gfk = m.fprime(x0)
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2
    H, k_total, f_old_ck = None, 0, None
    # rank 0 alone reads the checkpoint and broadcasts it: replicated optimisers must start bit-identical
    if path and root and os.path.exists(path):
        try:
            ck = np.load(path)
            ck_tag = float(ck["kappa"])
            same = (np.isnan(tag) and np.isnan(ck_tag)) or abs(ck_tag - tag) < 1e-12
            if ck["H"].shape == (N, N) and same:
                H = np.array(ck["H"])
                k_total = int(ck["k"])
                dx = float(np.max(np.abs(ck["x"] - x0)))
                if dx < 1e-10 * (1 + np.max(np.abs(x0))):
                    f_old_ck = float(ck["f_old"])
                log.info("BFGS: inverse Hessian loaded from %s (N=%d, %d iterations so far, max|x - x_ckpt| = %.2e)",
                         path, N, k_total, dx)
            else:
                log.info("BFGS: %s ignored (N %s vs %d, tag %s vs %s): H = I", path, ck["H"].shape, N, ck_tag, tag)
        except Exception as e:  # noqa: BLE001 - unreadable checkpoint: plain start
            log.warning("BFGS: could not read %s (%s): H = I", path, e)
    if dist:
        H, k_total, f_old_ck = _mpi.bcast((H, k_total, f_old_ck) if root else None)
    if f_old_ck is not None:
        old_old_fval = f_old_ck
    if H is None:
        H = np.eye(N)
    if root and os.path.exists(PAUSE_FLAG):
        os.remove(PAUSE_FLAG)

    def vnorm(v):
        return np.amax(np.abs(v)) if norm == np.inf else np.sum(np.abs(v) ** norm) ** (1.0 / norm)

    xk, k, warnflag, paused = x0, 0, 0, False
    gnorm = vnorm(gfk)
    try:
        while gnorm > gtol and k < maxiter:
            pk = -H.dot(gfk)
            try:
                alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(
                    m.f, m.fprime, xk, pk, gfk, old_fval, old_old_fval, amin=1e-100, amax=1e100)
            except _LineSearchError:
                warnflag = 2
                break
            sk = alpha_k * pk
            xk = xk + sk
            if gfkp1 is None:
                gfkp1 = m.fprime(xk)
            yk = gfkp1 - gfk
            gfk = gfkp1
            if callback is not None:
                callback(xk)
            k += 1
            gnorm = vnorm(gfk)
            if gnorm <= gtol or not np.isfinite(old_fval):
                warnflag = 0 if gnorm <= gtol else 2
                break
            ys = float(np.dot(yk, sk))
            rho = 1000.0 if ys == 0.0 else 1.0 / ys
            Hy = H.dot(yk)
            H += (rho * rho * float(np.dot(yk, Hy)) + rho) * np.outer(sk, sk) - rho * (np.outer(sk, Hy) +
                                                                                        np.outer(Hy, sk))
            stop = time.time() > deadline - margin
            if dist:
                stop = bool(_mpi.bcast(stop if root else None))
            if path and root and (stop or k % every == 0):
                _save(path, H, xk, old_fval, old_old_fval, k_total + k, tag)
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
        if path and root:
            _save(path, H, xk, old_fval, old_old_fval, k_total + k, tag)
        raise

    if paused:
        msg = "paused for allocation end"
    elif warnflag == 2:
        msg = "Desired error not necessarily achieved due to precision loss."
    elif k >= maxiter:
        warnflag, msg = 1, "Maximum number of iterations has been exceeded."
    else:
        msg = "Optimization terminated successfully."
    if path and root and not paused:
        _save(path, H, xk, old_fval, old_old_fval, k_total + k, tag)
    return OptimizeResult(fun=old_fval, jac=gfk, hess_inv=H, nfev=m.nfev, njev=m.nfev, status=warnflag,
                          success=(warnflag == 0), message=msg, x=xk, nit=k)
