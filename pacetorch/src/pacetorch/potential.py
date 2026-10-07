"""Loss / gradient / prediction on top of pacetorch.ace.ACE. Same loss as TensorPotential 0.2 (per-atom energy,
forces, L1/L2, orthogonality aux) plus the stress term:

  loss = e_f * sum_s w_e,s ((E_s - E_s^ref)/N_s)^2 + f_f * sum_i w_f,i |F_i - F_i^ref|^2
       + s_f * sum_s w_s,s |sigma_s - sigma_s^ref|^2_(6 Voigt)  + l1 * sum|c| + l2 * sum c^2 (+ w_orth * dsor)

sigma = sym(h^T dE/dh) / V at fixed fractional coordinates (ASE sign, eV/A^3).
"""
import numpy as np
import torch

from pacetorch.constants import *


class DeviceBatch:
    """A numpy batch moved to the device once, with the model's index tables."""

    def __init__(self, b, model, device):
        t = lambda x, dt=torch.float64: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
        L = torch.int32      # index arrays: int32 halves the resident neighbour-list memory (index_select/add accept it)
        self.np = b
        self.nstruct = len(b["energy"])
        self.positions = t(b["positions"])
        self.offsets = t(b["offsets"])
        self.cell = t(b["cell"])
        self.energy = t(b["energy"]).reshape(-1, 1)
        self.forces = t(b["forces"])
        self.eweights = t(b["eweights"]).reshape(-1, 1)
        self.fweights = t(b["fweights"]).reshape(-1, 1)
        self.stress = t(b["stress"]).reshape(-1, 3, 3)
        self.sweights = t(b["sweights"]).reshape(-1, 1)
        self.ind_i, self.ind_j = t(b["ind_i"], L), t(b["ind_j"], L)
        self.cell_map, self.pair_map, self.e_map = t(b["cell_map"], L), t(b["pair_map"], L), t(b["e_map"], L)
        nat = np.bincount(np.asarray(b["e_map"]), minlength=self.nstruct)
        self.natoms = t(nat).reshape(-1, 1)
        self.has_stress = bool(np.any(np.asarray(b["sweights"]) > 0))
        ix = model.prepare_batch(b)
        self.idx = dict(
            nat=ix["nat"], atoms_e_cpu=ix["atoms_e"], atoms_e=[t(a, L) for a in ix["atoms_e"]],
            present=t(ix["present"].astype(np.float64)),
            pairs=[dict(n=len(P["p"]), p=t(P["p"], L), central=t(P["central"], L),
                        central_local=t(P["central_local"], L)) for P in ix["pairs"]])


class TorchPotential:
    def __init__(self, potential, loss_specs=None):
        self.potential = potential
        self.device = potential.device
        specs = dict(loss_specs or {})
        lt = specs.get(LOSS_TYPE, "per-atom")
        if lt not in ("per-atom", "per-structure"):
            raise ValueError("{} is not a supported loss type".format(lt))
        self.loss_specs = {
            LOSS_TYPE: lt,
            LOSS_FORCE_FACTOR: float(specs.get(LOSS_FORCE_FACTOR, 0.)),
            LOSS_ENERGY_FACTOR: float(specs.get(LOSS_ENERGY_FACTOR, 1.)),
            L1_REG: float(specs.get(L1_REG, 0.)),
            L2_REG: float(specs.get(L2_REG, 0.)),
            AUX_LOSS_FACTOR: [float(x) for x in np.atleast_1d(specs.get(AUX_LOSS_FACTOR, [0.]))],
            LOSS_STRESS_FACTOR: float(specs.get(LOSS_STRESS_FACTOR, 0.)),
        }
        self.fit_stress = self.loss_specs[LOSS_STRESS_FACTOR] > 0
        self.reg_components = np.zeros((2, 1))

    def device_batch(self, b):
        return DeviceBatch(b, self.potential, self.device)

    # ------------------------------------------------------------------------------------------------ physics
    def _energy(self, db, need_stress):
        cell = db.cell.detach().requires_grad_(need_stress)
        pos = torch.einsum("ai,aij->aj", db.positions, cell.index_select(0, db.cell_map))
        if not need_stress:
            pos = pos.detach().requires_grad_(True)
        off = torch.einsum("pi,pij->pj", db.offsets, cell.index_select(0, db.pair_map))
        r_ij = pos.index_select(0, db.ind_j) + off - pos.index_select(0, db.ind_i)
        e_atom = self.potential.atomic_energy(r_ij, db.idx)
        e = torch.zeros(db.nstruct, dtype=torch.float64, device=e_atom.device).index_add(0, db.e_map, e_atom)
        return e.reshape(-1, 1), pos, cell

    @staticmethod
    def _virial_stress(cell, dedh):
        # |a . (b x c)|: exact, and no cuSOLVER call (linalg.det failed on CUDA with mixed TF/torch CUDA libraries)
        vol = torch.abs(torch.sum(cell[:, 0] * torch.cross(cell[:, 1], cell[:, 2], dim=1), dim=1)).reshape(-1, 1, 1)
        sig = torch.matmul(cell.transpose(1, 2), dedh) / vol
        return 0.5 * (sig + sig.transpose(1, 2))

    def _loss(self, db, e, f, sig):
        ls = self.loss_specs
        if ls[LOSS_TYPE] == "per-atom":
            le = torch.sum(db.eweights * ((e - db.energy) / db.natoms) ** 2)
        else:
            le = torch.sum(db.eweights * (e - db.energy) ** 2)
        lf = torch.sum(db.fweights * (f - db.forces) ** 2)
        loss = ls[LOSS_ENERGY_FACTOR] * le + ls[LOSS_FORCE_FACTOR] * lf
        if sig is not None:
            ds = sig - db.stress
            dv = torch.stack([ds[:, 0, 0], ds[:, 1, 1], ds[:, 2, 2], ds[:, 1, 2], ds[:, 0, 2], ds[:, 0, 1]], 1)
            loss = loss + ls[LOSS_STRESS_FACTOR] * torch.sum(db.sweights * dv ** 2)
        pot = self.potential
        pot.compute_regularization()
        loss = loss + pot.reg_l1 * ls[L1_REG] + pot.reg_l2 * ls[L2_REG]
        regs = [pot.reg_l1.detach().reshape(1, 1), pot.reg_l2.detach().reshape(1, 1)]
        if pot.aux is not None:
            for i, a in enumerate(pot.aux):
                loss = loss + torch.squeeze(a * ls[AUX_LOSS_FACTOR][i])
                regs.append(a.detach().reshape(1, 1))
        return loss, regs

    def loss_and_grad(self, db):
        """-> loss (0-d), dloss/dcoefs, e [S,1], f [nat,3], sigma [S,3,3] or None; all detached, on device."""
        c = self.potential.coefs.detach().requires_grad_(True)
        self.potential.coefs = c
        need_s = self.fit_stress
        e, pos, cell = self._energy(db, need_s)
        if need_s:
            dpos, dcell = torch.autograd.grad(e.sum(), [pos, cell], create_graph=True)
            sig = self._virial_stress(cell, dcell)
        else:
            (dpos,) = torch.autograd.grad(e.sum(), [pos], create_graph=True)
            sig = None
        f = -dpos
        loss, regs = self._loss(db, e, f, sig)
        (g,) = torch.autograd.grad(loss, [c])
        self.potential.coefs = c.detach()
        self.reg_components = regs
        return loss.detach(), g, e.detach(), f.detach(), (sig.detach() if sig is not None else None)

    def evaluate(self, db):
        """energy [S,1], forces [nat,3], stress [S,3,3] (eV/A^3, ASE sign) without a loss."""
        self.potential.coefs = self.potential.coefs.detach()
        e, pos, cell = self._energy(db, True)
        dpos, dcell = torch.autograd.grad(e.sum(), [pos, cell])
        return e.detach(), (-dpos).detach(), self._virial_stress(cell.detach(), dcell)
