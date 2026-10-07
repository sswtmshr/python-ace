"""Radial functions, cutoffs, spherical harmonics and embeddings of TensorPotential 0.2 (tensorpotential/functions,
potentials/ace.py), ported 1:1 to torch (float64). Same formulas, same branch conventions, so a fit gives the same
potential as the TF backend."""
import math

import numpy as np
import torch


# ----------------------------------------------------------------------------------------------- radial basis
def chebvander(x, deg):
    """x: [P] -> [P, max(deg, 2)] Chebyshev T_0, T_1, ... exactly as TF radial_functions.chebvander (which returns
    two columns for deg=1)."""
    v = [torch.ones_like(x)]
    if deg > 0:
        x2 = 2 * x
        v.append(x)
        for i in range(2, deg):
            v.append(v[i - 1] * x2 - v[i - 2])
    return torch.stack(v, dim=1)


def cutoff_func_cos(x, rcut):
    return torch.where(x < rcut, (1. + torch.cos(math.pi * x / rcut)) * 0.5, torch.zeros_like(x))


def cutoff_func_poly(r, rin, delta):
    x = 1 - 2 * (1 + (r - rin) / (delta + 1e-8))
    f = 7.5 * (x / 4 - x ** 3 / 6 + x ** 5 / 20)
    return torch.where(rin < r, torch.zeros_like(r), torch.where(r < rin - delta, torch.ones_like(r), 0.5 * (1 + f)))


def _sinc(x):
    return torch.where(x != 0, torch.sin(x) / x, torch.ones_like(x))


def _fn(x, rc, n):
    return ((-1.) ** n) * math.sqrt(2.) * math.pi / rc ** 1.5 * (n + 1) * (n + 2) / math.sqrt(
        (n + 1) ** 2 + (n + 2) ** 2) * (_sinc(x * (n + 1) * math.pi / rc) + _sinc(x * (n + 2) * math.pi / rc))


def simplified_bessel(x, rc, deg):
    sbf = [_fn(x, rc, 0.)]
    d = [1.]
    for i in range(1, deg):
        n = float(i)
        en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = 1 - en / d[i - 1]
        d.append(dn)
        sbf.append(1 / math.sqrt(d[i]) * (_fn(x, rc, n) + math.sqrt(en / d[i - 1]) * sbf[i - 1]))
    return torch.stack(sbf, dim=1)


def radial_function(d, nfunc, ftype, cutoff, lmbda):
    """d: [P] -> [P, nfunc]"""
    if ftype == "ChebExpCos":
        d_ch = 1. - 2. * ((torch.exp(-lmbda * (d / cutoff - 1.)) - 1.) / (math.exp(lmbda) - 1.))
        cheb = chebvander(d_ch, nfunc)
        cos_cut = cutoff_func_cos(d, cutoff).unsqueeze(1)
        gk = torch.where(cheb == 1., cheb * cos_cut, 0.5 * (1 - cheb) * cos_cut)
        return gk[:, :-1] if nfunc == 1 else gk
    if ftype == "ChebPow":
        d_ch = 2.0 * (1.0 - torch.abs(1.0 - d / cutoff) ** lmbda) - 1.0
        cheb = 0.5 - 0.5 * chebvander(d_ch, nfunc + 1)[:, 1:]
        return torch.where((d < cutoff).unsqueeze(1), cheb, torch.zeros_like(cheb))
    if ftype in ("SBessel", "TEST_SBessel"):
        y = simplified_bessel(d, cutoff, nfunc)
        return torch.where((d < cutoff).unsqueeze(1), y, torch.zeros_like(y))
    if ftype == "TEST_LegendrePow":
        x = 2.0 * (1.0 - torch.abs(1.0 - d / cutoff) ** lmbda) - 1.0
        leg = [torch.ones_like(x), x]
        for i in range(2, nfunc + 1):
            leg.append(((2 * i - 1) * leg[i - 1] * x - (i - 1) * leg[i - 2]) / i)
        leg = (torch.stack(leg[:nfunc + 1], dim=1)[:, 1:] - 1) / 2
        return torch.where((d < cutoff).unsqueeze(1), leg, torch.zeros_like(leg))
    raise ValueError("Unknown radial function type {}".format(ftype))


# ----------------------------------------------------------------------------------------- spherical harmonics
class SphericalHarmonics:
    """Complex Y_lm (real, imag) for l <= lmax, flattened index l*(l+1)+m, m=-l..l, times sqrt(4 pi) as in the TF
    ACE (same associated-Legendre recursion and constants as tensorpotential.functions.spherical_harmonics)."""

    def __init__(self, lmax):
        self.lmax = lmax
        alm, blm = [0.], [0.]
        for i in range(1, lmax + 1):
            l = float(i)
            lsq, ld = l * l, 2 * l
            l1, l2 = 4 * lsq - 1, lsq - ld + 1
            for j in range(0, i + 1):
                m = float(j)
                msq = m * m
                if i == j:
                    alm.append(-math.sqrt(1.0 + 0.5 / m))
                    blm.append(0.)
                else:
                    alm.append(math.sqrt(l1 / (lsq - msq)))
                    blm.append(-math.sqrt((l2 - msq) / (4 * l2 - 1)))
        self.alm, self.blm = alm, blm
        self.factor4pi = math.sqrt(4 * math.pi)

    @staticmethod
    def lm1d(l, m):
        return m + l * (l + 1) // 2

    def legendre(self, x):
        y00 = math.sqrt(1. / (4. * math.pi))
        plm = [x * 0 + y00]
        if self.lmax > 0:
            plm.append(math.sqrt(3. / (4. * math.pi)) * x)
            plm.append(x * 0 - math.sqrt(3. / (8. * math.pi)))
            for l in range(2, self.lmax + 1):
                for m in range(0, l + 1):
                    if m == l - 1:
                        plm.append(x * math.sqrt(2. * m + 3.) * plm[self.lm1d(l - 1, l - 1)])
                    elif m == l:
                        plm.append(self.alm[self.lm1d(l, l)] * plm[self.lm1d(l - 1, l - 1)])
                    else:
                        plm.append(self.alm[self.lm1d(l, m)] * (x * plm[self.lm1d(l - 1, m)]
                                                                + self.blm[self.lm1d(l, m)] * plm[self.lm1d(l - 2, m)]))
        return plm

    def compute(self, rhat):
        """rhat [P,3] -> (yr, yi) [P, (lmax+1)^2]"""
        x, y, z = rhat[:, 0], rhat[:, 1], rhat[:, 2]
        plm = self.legendre(z)
        yr, yi = {}, {}
        for l in range(self.lmax + 1):
            yr[(l, 0)] = plm[self.lm1d(l, 0)]
            yi[(l, 0)] = torch.zeros_like(x)
        pr, pi = x, y
        for m in range(1, self.lmax + 1):
            if m > 1:
                pr, pi = pr * x - pi * y, pr * y + pi * x
            for l in range(m, self.lmax + 1):
                yr[(l, m)] = pr * plm[self.lm1d(l, m)]
                yi[(l, m)] = pi * plm[self.lm1d(l, m)]
        out_r, out_i = [], []
        for l in range(self.lmax + 1):
            for m in range(-l, l + 1):
                if m < 0:
                    ph = 1. if abs(m) % 2 == 0 else -1.
                    out_r.append(yr[(l, -m)] * ph)
                    out_i.append(-yi[(l, -m)] * ph)
                else:
                    out_r.append(yr[(l, m)])
                    out_i.append(yi[(l, m)])
        return torch.stack(out_r, 1) * self.factor4pi, torch.stack(out_i, 1) * self.factor4pi


# ----------------------------------------------------------------------------------------------- embeddings
def embedding(rho, mexp, ftype):
    if ftype == "FinnisSinclairShiftedScaled":
        if abs(mexp - 1.) < 1e-10:
            return rho
        arho = torch.abs(rho)
        exprho = torch.exp(-arho)
        nx = 1. / mexp
        xoff = nx ** (nx / (1.0 - nx)) * exprho
        yoff = nx ** (1 / (1.0 - nx)) * exprho
        return torch.sign(rho) * ((xoff + arho) ** mexp - yoff)
    if ftype == "FinnisSinclair":
        w = 10.
        y1 = w * rho ** 2
        g = torch.where(30. < y1, 0. * rho, torch.exp(-y1))
        a = torch.abs(rho)
        f = torch.sign(rho) * ((((1. - g) * a + 1e-20) ** mexp) + mexp * g * a)
        return torch.where(a < 1e-10, mexp * rho, f)
    raise ValueError("Unknown embedding {}".format(ftype))
