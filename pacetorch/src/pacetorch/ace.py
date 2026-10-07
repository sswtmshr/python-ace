"""Finnis-Sinclair B-basis ACE (pacemaker / TensorPotential 0.2 model) in torch.

Same model, same coefficient vector layout as tensorpotential.potentials.ace.ACE (radial coefficients of the unique
bonds first, then the basis coefficients in BBasisSet order), so get_updated_config() returns the same
BBasisConfiguration and the .yaml/.yace written by pacemaker are unchanged.

Differences from the TF implementation are in how the same numbers are computed, not in what is computed:
  * all index bookkeeping (bond-type partition, atom->element, unique central atoms) is done once per batch on the
    host (prepare_batch), instead of dynamic_partition / tf.unique / tf.where inside every evaluation;
  * the atomic base A is built only for the (mu_j, n, l, m) channels the product basis actually uses, with no padding
    to nradmax/lmax, and pairs are scatter-added straight into it;
  * the last factor of every product basis function only needs the real part.
"""
from itertools import combinations_with_replacement

import numpy as np
import torch

from pacetorch.functions import SphericalHarmonics, cutoff_func_poly, embedding, radial_function

_Z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12,
      'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23,
      'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
      'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
      'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
      'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67,
      'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
      'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89,
      'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94}


class ACE:
    def __init__(self, bbasisconfig, compute_smoothness=False, compute_orthogonality=False, device="cpu"):
        from pyace.basis import ACEBBasisSet, BBasisConfiguration
        assert isinstance(bbasisconfig, BBasisConfiguration), "expected a pyace BBasisConfiguration"
        if compute_smoothness:
            raise NotImplementedError("radial smoothness regularisation (w0_rad/w1_rad/w2_rad) is not ported to the "
                                      "torch backend; use evaluator: tensorpot")
        self.compute_orthogonality = compute_orthogonality
        self.device = torch.device(device)
        self.bbasisset = ACEBBasisSet(bbasisconfig)
        b = self.bbasisset
        self.nelements = b.nelements
        self.elem_names = b.elements_name
        self.elems_Z = [_Z[s] for s in self.elem_names]
        self.bond_specs = b.map_bond_specifications
        self.embed_spec = b.map_embedding_specifications
        self.bond_combs = list(self.bond_specs.keys())
        self.ndensity = b.ndensitymax
        self.rcut = b.cutoffmax

        coefs = np.array(b.all_coeffs, dtype=np.float64)
        coefs[coefs == 0] += 1e-32          # as in TF (keeps |c| differentiable-looking for L1)
        self.ncoef = len(coefs)

        # ---- radial coefficient slices (shared by (a,b) and (b,a))
        self.total_num_crad = 0
        uniq = {}
        for mus in combinations_with_replacement(range(self.nelements), 2):
            if mus not in self.bond_specs:
                continue
            n_crad = int(np.prod(np.shape(self.bond_specs[mus].radcoefficients)))
            uniq[mus] = (self.total_num_crad, self.total_num_crad + n_crad)
            self.total_num_crad += n_crad
        self.bond_to_slice = {k: uniq[tuple(sorted(k))] for k in self.bond_combs}

        # ---- basis coefficient indices (relative to basis_coefs = coefs[total_num_crad:])
        coefs_r1 = {c: [] for c in self.bond_combs}
        self.coefs_rk_part = {c: [] for c in self.bond_combs}
        coefs_part = {}
        count = 0
        self.ranksmax = []
        funcs_by_elem = []
        for ne in range(self.nelements):
            coefs_part[ne] = []
            rank = 0
            for f in list(b.basis_rank1[ne]) + list(b.basis[ne]):
                for _ in f.coeffs:
                    key = (f.mu0, f.mus[0])
                    if f.rank == 1:
                        coefs_r1[key].append(count)
                    else:
                        coefs_part[ne].append(count)
                    self.coefs_rk_part[key].append(count + self.total_num_crad)
                    count += 1
                rank = max(rank, f.rank)
            self.ranksmax.append(rank)
            funcs_by_elem.append([f for f in b.basis[ne]])
            rk = [f.rank for f in b.basis[ne]]
            # the coefficient order (BBasisSet order) must equal the product-basis column order (grouped by rank)
            assert rk == sorted(rk), "basis functions of element {} are not sorted by rank".format(ne)
        self.rankmax = max(self.ranksmax)

        # ---- per element: product-basis index tables on a compact A layout
        # A_e columns: the used (mu_j, n, lm) channels of central element e, grouped by neighbour element mu_j
        self.elem = []
        for ne in range(self.nelements):
            nd = self.embed_spec[ne].ndensity
            funcs = funcs_by_elem[ne]
            used = sorted({(f.mus[k], f.ns[k] - 1, f.ls[k] * (f.ls[k] + 1) + int(ms[k]))
                           for f in funcs for ms in f.ms_combs for k in range(f.rank)})
            col = {u: i for i, u in enumerate(used)}
            # per neighbour element: which (n, l, lm) channels to build from the pair contributions
            chan = {}
            for (mu, n, lm) in used:
                l = int(np.floor(np.sqrt(lm)))
                chan.setdefault(mu, []).append((n, l, lm))
            ranks = []
            for r in range(2, self.ranksmax[ne] + 1):
                fr = [f for f in funcs if f.rank == r]
                if not fr:
                    continue
                idx = [[] for _ in range(r)]
                gcg, msum = [], []
                for fi, f in enumerate(fr):
                    for ms, cg in zip(f.ms_combs, f.gen_cgs):
                        for k in range(r):
                            idx[k].append(col[(f.mus[k], f.ns[k] - 1, f.ls[k] * (f.ls[k] + 1) + int(ms[k]))])
                        gcg.append(cg)
                        msum.append(fi)
                ranks.append(dict(rank=r, nfunc=len(fr), idx=[torch.tensor(i, dtype=torch.long) for i in idx],
                                  gcg=torch.tensor(gcg, dtype=torch.float64),
                                  msum=torch.tensor(msum, dtype=torch.long)))
            self.elem.append(dict(nd=nd, ncols=len(used), chan=chan, ranks=ranks,
                                  coefs_idx=torch.tensor(coefs_part[ne], dtype=torch.long),
                                  fs=list(self.embed_spec[ne].FS_parameters), npot=self.embed_spec[ne].npoti,
                                  rho_cut=float(self.embed_spec[ne].rho_core_cutoff),
                                  drho_cut=float(self.embed_spec[ne].drho_core_cutoff)))
        # column offset of each neighbour element's block inside A_e (columns are sorted by mu first)
        for ne, E in enumerate(self.elem):
            off, blocks = 0, {}
            for mu in sorted(E["chan"]):
                blocks[mu] = (off, off + len(E["chan"][mu]))
                off += len(E["chan"][mu])
            E["blocks"] = blocks

        # ---- per bond combination: radial/Ylm channel tables
        self.bonds = []
        for c in self.bond_combs:
            bs = self.bond_specs[c]
            e, mu = c
            ch = self.elem[e]["chan"].get(mu, [])
            nl_idx = [n * (bs.lmax + 1) + l for (n, l, lm) in ch]
            lm_idx = [lm for (n, l, lm) in ch]
            lmax_used = max([l for (n, l, lm) in ch], default=0)
            self.bonds.append(dict(
                c=c, nradmax=bs.nradmax, lmax=bs.lmax, nradbase=bs.nradbasemax, rcut=float(bs.rcut),
                radbase=bs.radbasename, lmbda=float(bs.radparameters[0]),
                rcut_in=float(bs.rcut_in), dcut_in=float(bs.dcut_in),
                prehc=float(bs.prehc), lambdahc=float(bs.lambdahc),
                crad=self.bond_to_slice[c], nd=self.embed_spec[e].ndensity,
                r1_idx=torch.tensor(coefs_r1[c], dtype=torch.long),
                nl_idx=torch.tensor(nl_idx, dtype=torch.long), lm_idx=torch.tensor(lm_idx, dtype=torch.long),
                sh=SphericalHarmonics(lmax_used) if ch else None, has_A=bool(ch)))

        self.coefs = torch.tensor(coefs, dtype=torch.float64)
        self.to(device)
        self.reg_l1 = self.reg_l2 = None
        self.aux = None

    # ------------------------------------------------------------------------------------------------ plumbing
    def to(self, device):
        self.device = torch.device(device)
        mv = lambda t: t.to(self.device)
        for E in self.elem:
            E["coefs_idx"] = mv(E["coefs_idx"])
            for R in E["ranks"]:
                R["idx"] = [mv(i) for i in R["idx"]]
                R["gcg"], R["msum"] = mv(R["gcg"]), mv(R["msum"])
        for B in self.bonds:
            for k in ("r1_idx", "nl_idx", "lm_idx"):
                B[k] = mv(B[k])
        self.coefs = self.coefs.to(self.device)
        return self

    def set_coefs(self, coefs):
        with torch.no_grad():
            self.coefs.copy_(torch.as_tensor(np.asarray(coefs, dtype=np.float64)))

    def get_coefs(self):
        return self.coefs.detach().cpu().numpy().copy()

    def get_updated_config(self, updating_coefs=None, prefix=None):
        if updating_coefs is not None:
            self.set_coefs(updating_coefs)
        self.bbasisset.all_coeffs = self.get_coefs()
        return self.bbasisset.to_BBasisConfiguration()

    def selective_fitting(self, list_of_bonds, basis_factor=1., rad_coefs_factor=1.):
        factor = np.zeros(self.ncoef)
        for bond in list_of_bonds:
            assert bond in self.bond_combs, "Bond type {} is not in the potential".format(bond)
            factor[self.coefs_rk_part[bond]] = basis_factor
            factor[slice(*self.bond_to_slice[bond])] = rad_coefs_factor
        return factor

    # ------------------------------------------------------------------------------------------- batch indices
    def prepare_batch(self, b):
        """Host-side, once per batch: element of every atom, element-local atom index, per-bond-type pair lists."""
        lut = np.full(256, -1, dtype=np.int64)          # atomic number -> element index
        lut[self.elems_Z] = np.arange(len(self.elems_Z))
        ind_i = np.asarray(b["ind_i"], dtype=np.int64)
        ind_j = np.asarray(b["ind_j"], dtype=np.int64)
        nat = len(b["positions"])
        ei = lut[np.asarray(b["mu_i"], dtype=np.int64)]
        ej = lut[np.asarray(b["mu_j"], dtype=np.int64)]
        if (ei < 0).any() or (ej < 0).any():
            raise ValueError("structure contains an element that is not in the potential")
        atom_e = np.full(nat, -1, dtype=np.int64)
        atom_e[ind_i] = ei
        if (atom_e < 0).any():
            raise ValueError("atom without neighbours inside a batch")
        local = np.zeros(nat, dtype=np.int64)
        atoms_e = []
        for e in range(self.nelements):
            a = np.nonzero(atom_e == e)[0]
            local[a] = np.arange(len(a))
            atoms_e.append(a)
        pairs = []
        present = np.zeros((nat, len(self.bond_combs)), dtype=bool)
        key = ei * self.nelements + ej
        order = np.argsort(key, kind="stable")
        bounds = np.searchsorted(key[order], np.arange(self.nelements ** 2 + 1))
        for ci, (e, mu) in enumerate(self.bond_combs):
            k = e * self.nelements + mu
            p = order[bounds[k]:bounds[k + 1]]
            pairs.append(dict(p=p, central=ind_i[p], central_local=local[ind_i[p]]))
            present[ind_i[p], ci] = True
        return dict(nat=nat, atoms_e=atoms_e, pairs=pairs, present=present)

    # ------------------------------------------------------------------------------------------------ forward
    def basis_coefs(self):
        return self.coefs[self.total_num_crad:]

    def compute_regularization(self):
        bc = self.basis_coefs()
        self.reg_l1 = torch.sum(torch.abs(bc))
        self.reg_l2 = torch.sum(bc ** 2)
        if self.compute_orthogonality:
            dsor = torch.zeros((), dtype=torch.float64, device=self.device)
            if self.rankmax > 1:
                for B in self.bonds:
                    crad = self.coefs[slice(*B["crad"])].reshape(B["nradmax"], B["lmax"] + 1, B["nradbase"])
                    c_lnk = crad.permute(1, 0, 2)
                    wwt = c_lnk @ c_lnk.transpose(1, 2)
                    wtw = c_lnk.transpose(1, 2) @ c_lnk
                    i1 = torch.eye(B["nradmax"], dtype=torch.float64, device=self.device)
                    i2 = torch.eye(B["nradbase"], dtype=torch.float64, device=self.device)
                    dsor = dsor + torch.mean((wwt - i1) ** 2) + torch.mean((wtw - i2) ** 2)
            self.aux = [dsor.reshape(1, 1)]
        else:
            self.aux = None

    def atomic_energy(self, r_ij, idx):
        """r_ij [P,3] -> per-atom energy [nat] (float64); idx = device-side prepare_batch() output."""
        nat = idx["nat"]
        dev = r_ij.device
        coefs = self.coefs
        bcoefs = coefs[self.total_num_crad:]
        d_all = torch.linalg.norm(r_ij, dim=1)
        nd_max = self.ndensity
        rho1 = [torch.zeros(len(a), nd_max, dtype=torch.float64, device=dev) for a in idx["atoms_e_cpu"]]
        blk_r = [dict() for _ in self.elem]
        blk_i = [dict() for _ in self.elem]
        ncomb = len(self.bonds)
        zero_nat = torch.zeros(nat, dtype=torch.float64, device=dev)
        phi_cols = [zero_nat] * ncomb

        for ci, B in enumerate(self.bonds):
            P = idx["pairs"][ci]
            if P["n"] == 0:
                continue
            e, mu = B["c"]
            d = d_all.index_select(0, P["p"])
            rij = r_ij.index_select(0, P["p"])
            # core repulsion (summed per central atom and bond type, as in TF)
            fcp = cutoff_func_poly(d, B["rcut_in"], B["dcut_in"])
            phi = B["prehc"] * torch.exp(-B["lambdahc"] * d ** 2) / d * fcp
            phi_cols[ci] = zero_nat.index_add(0, P["central"], phi)
            # radial basis g_k(r), damped inside the inner cutoff
            g = radial_function(d, B["nradbase"], B["radbase"], B["rcut"], B["lmbda"]) * (1 - fcp).unsqueeze(1)
            # rank 1: sum_j g_k, then rho += G @ c_r1
            if len(B["r1_idx"]):
                G = torch.zeros(len(idx["atoms_e_cpu"][e]), g.shape[1], dtype=torch.float64,
                                device=dev).index_add(0, P["central_local"], g)
                c1 = bcoefs.index_select(0, B["r1_idx"]).reshape(-1, B["nd"])
                r1 = G @ c1
                if B["nd"] < nd_max:
                    r1 = torch.nn.functional.pad(r1, (0, nd_max - B["nd"]))
                rho1[e] = rho1[e] + r1
            # A_{mu n l m}: R_nl(r) Y_lm(rhat), only the used channels
            if B["has_A"]:
                crad = coefs[slice(*B["crad"])].reshape(B["nradmax"], B["lmax"] + 1, B["nradbase"])
                R = torch.einsum("pk,nlk->pnl", g, crad).reshape(len(d), -1).index_select(1, B["nl_idx"])
                yr, yi = B["sh"].compute(rij / d.unsqueeze(1))
                ne_ = len(idx["atoms_e_cpu"][e])
                blk_r[e][mu] = torch.zeros(ne_, len(B["lm_idx"]), dtype=torch.float64, device=dev).index_add(
                    0, P["central_local"], R * yr.index_select(1, B["lm_idx"]))
                blk_i[e][mu] = torch.zeros(ne_, len(B["lm_idx"]), dtype=torch.float64, device=dev).index_add(
                    0, P["central_local"], R * yi.index_select(1, B["lm_idx"]))

        phi_ac = torch.stack(phi_cols, 1)
        phi_core = phi_ac.sum(1)
        e_atom = torch.zeros(nat, dtype=torch.float64, device=dev)
        for e, E in enumerate(self.elem):
            atoms = idx["atoms_e"][e]
            if len(idx["atoms_e_cpu"][e]) == 0:
                continue
            rho = rho1[e][:, :E["nd"]]
            if E["ranks"]:
                ne_ = len(idx["atoms_e_cpu"][e])
                z = lambda mu: torch.zeros(ne_, len(E["chan"][mu]), dtype=torch.float64, device=dev)
                Ar = torch.cat([blk_r[e].get(mu, None) if mu in blk_r[e] else z(mu) for mu in sorted(E["chan"])], 1)
                Ai = torch.cat([blk_i[e].get(mu, None) if mu in blk_i[e] else z(mu) for mu in sorted(E["chan"])], 1)
                Bf = []
                for R in E["ranks"]:
                    ar, ai = Ar.index_select(1, R["idx"][0]), Ai.index_select(1, R["idx"][0])
                    for k in range(1, R["rank"] - 1):
                        br, bi = Ar.index_select(1, R["idx"][k]), Ai.index_select(1, R["idx"][k])
                        ar, ai = ar * br - ai * bi, ai * br + ar * bi
                    br, bi = Ar.index_select(1, R["idx"][-1]), Ai.index_select(1, R["idx"][-1])
                    prod_r = ar * br - ai * bi            # last factor: real part only
                    Bf.append(torch.zeros(prod_r.shape[0], R["nfunc"], dtype=torch.float64, device=dev)
                              .index_add(1, R["msum"], prod_r * R["gcg"]))
                Bm = torch.cat(Bf, dim=1)
                rho = rho + Bm @ bcoefs.index_select(0, E["coefs_idx"]).reshape(-1, E["nd"])
            safe = torch.where(rho != 0., rho, rho + 1e-32)
            en = 0.
            for p in range(E["nd"]):
                en = en + E["fs"][2 * p] * embedding(safe[:, p], E["fs"][2 * p + 1], E["npot"])
            # inner cutoff on the core-repulsion density: mean over the bond types present at the atom (as in TF)
            phi_e = phi_ac.index_select(0, atoms)
            fc = []
            for ci, B in enumerate(self.bonds):
                rc, drc = E["rho_cut"], E["drho_cut"]
                x = phi_e[:, ci]
                fc.append(torch.where(rc <= x, torch.zeros_like(x),
                                      torch.where(x < rc - drc, torch.ones_like(x), cutoff_func_poly(x, rc, drc))))
            fc = torch.stack(fc, 1)
            pres = idx["present"].index_select(0, atoms)              # float 0/1 [nat_e, ncomb]
            inner = (fc * pres).sum(1) / pres.sum(1)
            e_atom = e_atom.index_add(0, atoms, en * inner)
        return e_atom + phi_core
