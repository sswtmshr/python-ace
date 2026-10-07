"""pyace.bfgs (native BFGS) against scipy's BFGS: same iterates and statuses; checkpoint resume; edge cases.
CPU only, seconds:  python tests/test_native_bfgs.py   (or pytest)"""
import os
import tempfile

import numpy as np
from scipy.optimize import minimize, rosen, rosen_der

from pyace.bfgs import minimize_bfgs, use_scipy, was_paused

F = lambda x: (rosen(x), rosen_der(x))
X0 = np.linspace(-1.2, 1.0, 12)


def _scipy(fun, x0, **o):
    return minimize(fun, x0, jac=True, method="BFGS", options=o)


def test_matches_scipy():
    for o in ({"maxiter": 60, "gtol": 1e-8}, {"maxiter": 60, "gtol": 1e-6, "norm": 2}, {"maxiter": 40, "c1": 1e-3,
                                                                                          "c2": 0.5}):
        a, b = _scipy(F, X0, **o), minimize_bfgs(F, X0, options=o)
        assert a.nit == b.nit and a.status == b.status, (o, a.nit, b.nit, a.status, b.status)
        # Rosenbrock is ill-conditioned: summation-order differences grow along 40-60 iterations
        assert abs(a.fun - b.fun) <= 1e-5 * max(1, abs(a.fun)), (o, a.fun, b.fun)
    # well-conditioned quadratic: tight agreement
    rng = np.random.default_rng(0)
    M = rng.normal(size=(30, 30))
    A = M @ M.T / 30 + np.eye(30)
    q = lambda x: (0.5 * float(x @ A @ x) - float(x.sum()), A @ x - 1)
    x0 = rng.normal(size=30)
    a, b = _scipy(q, x0, gtol=1e-6), minimize_bfgs(q, x0, options={"gtol": 1e-6})
    assert a.nit == b.nit and np.abs(a.x - b.x).max() < 1e-12, (a.nit, b.nit, np.abs(a.x - b.x).max())


def test_maxiter_zero_and_none():
    a, b = _scipy(F, X0, maxiter=0), minimize_bfgs(F, X0, options={"maxiter": 0})
    assert b.nit == 0 and b.nfev == 1 and a.status == b.status == 1, (a.status, b.status, b.nit, b.nfev)
    assert minimize_bfgs(F, X0, options={"maxiter": None, "gtol": 1e-3}).nit > 0


def test_norm_minus_inf():
    g = lambda x: (float(np.sum(x ** 2)), 2 * x)
    x0 = np.array([1e-3 / 2, 1.0, 2.0])
    a, b = _scipy(g, x0, norm=-np.inf, gtol=1e-2), minimize_bfgs(g, x0, options={"norm": -np.inf, "gtol": 1e-2})
    assert a.nit == b.nit == 0, (a.nit, b.nit)


def test_nan_is_not_success():
    bad = lambda x: (float("nan"), np.full_like(x, np.nan))
    a, b = _scipy(bad, X0), minimize_bfgs(bad, X0)
    assert not b.success and b.status == a.status == 3, (a.status, b.status, b.message)


def test_checkpoint_resume_bitwise_and_no_nan_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "ck.npz")
        o = {"gtol": 1e-10, "checkpoint": ck, "checkpoint_min_s": 0}
        full = minimize_bfgs(F, X0, options=dict(o, maxiter=30, checkpoint=os.path.join(d, "full.npz")))
        a = minimize_bfgs(F, X0, options=dict(o, maxiter=12))
        b = minimize_bfgs(F, a.x, options=dict(o, maxiter=18))
        assert np.array_equal(b.x, full.x), np.abs(b.x - full.x).max()
        # early stop: the checkpoint holds the last accepted step and its curvature
        n = [0]

        def cb(x):
            n[0] += 1
            if n[0] == 5:
                raise StopIteration("early stop")
        os.remove(ck)
        try:
            minimize_bfgs(F, X0, callback=cb, options=dict(o, maxiter=30))
        except StopIteration:
            pass
        assert int(np.load(ck)["k"]) == 5
        # NaN at the start: nothing written
        os.remove(ck)
        minimize_bfgs(lambda x: (float("nan"), np.full_like(x, np.nan)), X0, options=dict(o, maxiter=5))
        assert not os.path.exists(ck)


def test_scipy_only_options_and_pause_without_checkpoint():
    assert use_scipy({"gtol": 1e-8, "maxiter": 5, "disp": True, "c1": 1e-4, "xrtol": 0}) == []
    assert use_scipy({"hess_inv0": None, "return_all": True}) == ["hess_inv0", "return_all"]
    r = minimize_bfgs(F, X0, options={"maxiter": 5, "deadline": 0.0})     # deadline passed, no checkpoint: no pause
    assert not was_paused() and r.nit == 5
    with tempfile.TemporaryDirectory() as d:
        r = minimize_bfgs(F, X0, options={"maxiter": 50, "deadline": 0.0, "margin": 0,
                                          "checkpoint": os.path.join(d, "c.npz")})
        assert was_paused() and r.status == 9 and not r.success and r.nit == 1


if __name__ == "__main__":
    for t in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        t()
    print("ok")
