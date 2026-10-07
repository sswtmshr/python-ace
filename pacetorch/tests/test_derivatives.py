"""Checks for the hand-written derivatives and the evaluation paths (CPU, seconds).

    python -m pytest tests            (or: python tests/test_derivatives.py)

1. _SegOuter / _GatherContract: first and second derivatives against finite differences (gradcheck, gradgradcheck)
   and a third-order quantity against plain autograd.
2. ACE.products (prefix tree + GEMM) against the direct per-ms-combination product, with first and second
   derivatives, for every element of a random two-element basis.
3. Whole model on random SiC structures: the A-basis (outer products, hand-written derivatives) against the
   plain-autograd reference formulation: same energies, forces, stress, loss and loss gradient."""
import numpy as np
import torch

import pacetorch.ace as A


def test_segouter_derivatives():
    torch.manual_seed(0)
    P, n, m, nat = 13, 3, 5, 4
    u = torch.randn(P, n, dtype=torch.float64, requires_grad=True)
    v = torch.randn(P, m, dtype=torch.float64, requires_grad=True)
    idx = torch.sort(torch.randint(0, nat, (P,))).values
    seg = (idx, torch.searchsorted(idx, torch.arange(nat + 1)))
    f = lambda u, v: A._SegOuter.apply(seg, nat, u, v)
    assert torch.autograd.gradcheck(f, (u, v))
    assert torch.autograd.gradgradcheck(f, (u, v))

    def plain(u, v):
        return torch.zeros(nat, n * m, dtype=u.dtype).index_add(0, idx, (u[:, :, None] * v[:, None, :]).reshape(P, -1)
                                                                ).view(nat, n, m)
    res = []
    for F in (f, plain):
        E = (F(u, v) ** 3).sum()
        gu, = torch.autograd.grad(E, u, create_graph=True)
        res.append([E] + list(torch.autograd.grad((gu ** 2).sum(), (u, v))))
    for a, b in zip(*res):
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-12)


def test_triton_kernels_match_torch():
    """GPU only: the fused kernels against the torch formulation, all derivative orders (via the full model)."""
    if not torch.cuda.is_available() or not A._kernels.available(torch.zeros(1, dtype=torch.float64, device="cuda")):
        print("test_triton_kernels_match_torch: skipped (no CUDA / Triton)")
        return
    from pacetorch.potential import TorchPotential
    bconf, b = _model_and_batch()
    specs = {"loss_force_factor": 0.3, "loss_energy_factor": 0.7, "loss_stress_factor": 0.5}
    out = {}
    saved = A._TRITON
    try:
        for tri in (False, True):
            A._TRITON = tri
            ace = A.ACE(bconf, device="cuda")
            c0 = ace.get_coefs()
            ace.set_coefs(c0 + np.random.default_rng(2).normal(scale=0.3, size=c0.shape))
            tp = TorchPotential(ace, specs)
            out[tri] = [x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
                        for x in tp.loss_and_grad(tp.device_batch(b))]
    finally:
        A._TRITON = saved
    for name, a, r in zip(("loss", "grad", "E", "F", "stress"), out[True], out[False]):
        err = np.max(np.abs(a - r)) / max(np.max(np.abs(r)), 1e-300)
        assert err < 1e-11, "{}: relative difference {:.2e}".format(name, err)


def test_radial_and_embedding_types_match_cpp():
    """Every radial basis and embedding type against the C++ evaluator (what LAMMPS runs). Catches training formulas
    that differ from C++ (TF had ChebExpCos without the dcut taper and a different FinnisSinclair)."""
    from ase.build import bulk
    from ase import Atoms
    from pyace import PyACECalculator
    from pyace.multispecies_basisextension import create_multispecies_basis_config
    from pyace.atomicenvironment import generate_tp_atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    import pandas as pd
    from pacetorch.data import batching_data
    from pacetorch.potential import TorchPotential
    rng = np.random.default_rng(0)
    at = bulk("Si", "diamond", a=5.3, cubic=True)
    at = Atoms(rng.choice(["C", "Si"], len(at)), positions=at.positions, cell=at.cell, pbc=True)
    at.rattle(0.15, seed=1)
    for radbase, par in (("SBessel", 5.25), ("ChebExpCos", 5.25), ("ChebPow", 2.0)):
        for npot, fs in (("FinnisSinclairShiftedScaled", [1, 1, 1, 0.5]), ("FinnisSinclair", [1, 1, 1, 0.5])):
            b = create_multispecies_basis_config({
                "elements": ["C", "Si"], "embeddings": {"ALL": {"npot": npot, "fs_parameters": fs, "ndensity": 2}},
                "bonds": {"ALL": {"radbase": radbase, "radparameters": [par], "rcut": 4.5, "dcut": 0.01,
                                  "NameOfCutoffFunction": "cos", "nradbase": 6}},
                "functions": {"ALL": {"nradmax_by_orders": [6, 3, 2], "lmax_by_orders": [0, 2, 1]}}})
            ace = A.ACE(b)
            c0 = ace.get_coefs()
            ace.set_coefs(c0 + np.random.default_rng(5).normal(scale=0.2, size=c0.shape))
            b = ace.get_updated_config()
            b.deltaSplineBins = 2e-5          # C++ spline interpolation error well below the tolerance
            x = at.copy(); x.calc = PyACECalculator(b)
            e_ref, f_ref = x.get_potential_energy(), x.get_forces()
            y = at.copy(); y.calc = SinglePointCalculator(y, energy=0., forces=np.zeros((len(y), 3)))
            df = pd.DataFrame(dict(tp_atoms=[generate_tp_atoms(y, cutoff=4.5)], w_energy=[1.], w_forces=[np.ones(len(y))]))
            tp = TorchPotential(A.ACE(b))
            e, f, _ = tp.evaluate(tp.device_batch(batching_data(df, 1)[0]))
            de, dF = abs(float(e[0, 0]) - e_ref) / len(at), float(np.max(np.abs(f.numpy() - f_ref)))
            assert de < 1e-9 and dF < 1e-6, "{} + {}: dE/at {:.1e} eV, dF {:.1e} eV/A".format(radbase, npot, de, dF)


def _model_and_batch():
    from ase.build import bulk
    from pyace.atomicenvironment import generate_tp_atoms
    from pyace.multispecies_basisextension import create_multispecies_basis_config
    import pandas as pd
    from pacetorch.data import batching_data
    from ase.calculators.singlepoint import SinglePointCalculator
    bconf = create_multispecies_basis_config({
        "elements": ["Si", "C"],
        "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1, 1, 1, 0.5], "ndensity": 2}},
        "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.25], "rcut": 4.5, "dcut": 0.01,
                          "NameOfCutoffFunction": "cos", "nradbase": 6}},
        "functions": {"ALL": {"nradmax_by_orders": [6, 3, 2, 1], "lmax_by_orders": [0, 3, 2, 1]}}})
    rng = np.random.default_rng(1)
    rows, nat = [], []
    for k in range(3):
        at = bulk("SiC", "zincblende", a=4.36, cubic=True)
        at.rattle(0.1, seed=k)
        at.set_cell(at.cell * (1 + 0.02 * k), scale_atoms=True)
        at.calc = SinglePointCalculator(at, energy=-7.0 * len(at), forces=rng.normal(size=(len(at), 3)))
        rows.append(generate_tp_atoms(at, cutoff=4.5))
        nat.append(len(at))
    df = pd.DataFrame(dict(tp_atoms=rows, w_energy=[1 / 3] * 3, w_forces=[np.full(n, 1 / sum(nat)) for n in nat],
                           stress=[rng.normal(scale=1e-3, size=6) for _ in rows]))
    df["w_stress"] = 1 / 3
    return bconf, batching_data(df, 3)[0]


def _products_reference(E, Ar, Ai, bcoefs):
    """rho_d = sum_f c_fd sum_ms gcg Re prod_k A_k, one ms-combination at a time (the formula, no sharing)"""
    out = []
    for R in E["ranks"]:
        ar, ai = Ar[:, R["idx"][0]], Ai[:, R["idx"][0]]
        for k in range(1, R["rank"]):
            br, bi = Ar[:, R["idx"][k]], Ai[:, R["idx"][k]]
            ar, ai = ar * br - ai * bi, ai * br + ar * bi
        out.append(torch.zeros(Ar.shape[0], R["nfunc"], dtype=Ar.dtype).index_add(1, R["msum"], ar * R["gcg"]))
    return torch.cat(out, 1) @ bcoefs[E["coefs_idx"]].reshape(-1, E["nd"])


def test_products_match_direct_formula():
    bconf, _ = _model_and_batch()
    ace = A.ACE(bconf)
    torch.manual_seed(3)
    bc = torch.randn(ace.ncoef - ace.total_num_crad, dtype=torch.float64, requires_grad=True)
    for E in ace.elem:
        if not E["ranks"]:
            continue
        Ar = torch.randn(5, E["ncols"], dtype=torch.float64, requires_grad=True)
        Ai = torch.randn(5, E["ncols"], dtype=torch.float64, requires_grad=True)
        res = []
        for F in (A.ACE.products, _products_reference):
            rho = F(E, Ar, Ai, bc)
            gA = torch.autograd.grad((rho ** 2).sum(), (Ar, Ai), create_graph=True)
            g2 = torch.autograd.grad(sum((x ** 2).sum() for x in gA), (bc, Ar))
            res.append([rho, *gA, *g2])
        for a, b in zip(*res):
            assert torch.allclose(a, b, rtol=1e-11, atol=1e-11 * float(b.abs().max())), float((a - b).abs().max())


def test_abasis_paths_agree():
    from pacetorch.potential import TorchPotential
    bconf, b = _model_and_batch()
    specs = {"loss_force_factor": 0.3, "loss_energy_factor": 0.7, "loss_stress_factor": 0.5}
    out = {}
    saved = A._REFERENCE_ABASIS
    try:
        for ab in ("scatter", "outer"):
            A._REFERENCE_ABASIS = ab == "scatter"
            ace = A.ACE(bconf)
            c0 = ace.get_coefs()        # random coefficients (radial ones are shared by (a,b) and (b,a))
            ace.set_coefs(c0 + np.random.default_rng(2).normal(scale=0.3, size=c0.shape))
            tp = TorchPotential(ace, specs)
            loss, g, e, f, s = tp.loss_and_grad(tp.device_batch(b))
            out[ab] = [x.numpy() if torch.is_tensor(x) else np.asarray(x) for x in (loss, g, e, f, s)]
    finally:
        A._REFERENCE_ABASIS = saved
    for name, a, r in zip(("loss", "grad", "E", "F", "stress"), out["outer"], out["scatter"]):
        err = np.max(np.abs(a - r)) / max(np.max(np.abs(r)), 1e-300)
        assert err < 1e-11, "{}: relative difference {:.2e}".format(name, err)


if __name__ == "__main__":
    test_segouter_derivatives()
    test_products_match_direct_formula()
    test_abasis_paths_agree()
    test_radial_and_embedding_types_match_cpp()
    if torch.cuda.is_available():
        test_triton_kernels_match_torch()
    print("ok")
