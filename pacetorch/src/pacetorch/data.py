"""DataFrame (pacemaker `tp_atoms` column + weights) -> list of numpy batches. Same batch layout as
tensorpotential.utils.utilities.supply_batch, plus stress / sweights. No framework imports."""
import logging

import numpy as np

log = logging.getLogger(__name__)


def voigt_to_tensor(s):
    """ASE stress (6 Voigt xx yy zz yz xz xy, or 3x3) -> (1, 3, 3)"""
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if s.size == 9:
        return s.reshape(1, 3, 3)
    xx, yy, zz, yz, xz, xy = s
    return np.array([[[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]])


def _entry(tp, w_e=None, w_f=None, stress=None, w_s=0.):
    n_at = len(tp["_positions"])
    e = dict(tp)
    e["_eweights"] = np.ones((1, 1)) if w_e is None else np.array([[float(w_e)]])
    e["_fweights"] = np.ones((n_at, 1)) if w_f is None else np.asarray(w_f, dtype=np.float64).reshape(-1, 1)
    if e["_fweights"].shape[0] != n_at:
        raise ValueError("w_forces has {} entries for a structure with {} atoms".format(e["_fweights"].shape[0], n_at))
    e["_sweights"] = np.array([[float(w_s)]])
    e["_stress"] = voigt_to_tensor(stress) if w_s > 0 else np.zeros((1, 3, 3))
    return e


def batching_data(df, batch_size=10):
    """Rows without tp_atoms are dropped (logged). Weights come from w_energy / w_forces / w_stress when present; a
    malformed weight raises instead of silently falling back to unit weights."""
    n0 = len(df)
    df = df[df["tp_atoms"].notna()]
    if len(df) < n0:
        log.warning("{} structures without tp_atoms (no neighbours) are skipped".format(n0 - len(df)))
    has_w = "w_energy" in df.columns and "w_forces" in df.columns
    if not has_w:
        log.warning("no w_energy/w_forces columns: unit weights are used")
    has_s = "w_stress" in df.columns
    tp_col = df["tp_atoms"].tolist()
    we = df["w_energy"].tolist() if has_w else [None] * len(df)
    wf = df["w_forces"].tolist() if has_w else [None] * len(df)
    ws = df["w_stress"].tolist() if has_s else [0.] * len(df)
    st = df["stress"].tolist() if has_s else [None] * len(df)
    entries = [_entry(t, a, b, s, w) for t, a, b, s, w in zip(tp_col, we, wf, st, ws)]
    bs = max(1, int(batch_size))
    return [supply_batch(entries[i:i + bs]) for i in range(0, len(entries), bs)]


def supply_batch(data):
    keys = ["ind_i", "ind_j", "mu_i", "mu_j", "offsets", "energy", "forces", "positions", "eweights", "fweights",
            "stress", "sweights", "cell", "cell_map", "pair_map", "e_map"]
    out = {k: [] for k in keys}
    n_atoms_tot = 0
    for j, e in enumerate(data):
        n_atoms = e["_positions"].shape[0]
        n_pairs = e["_ind_j"].shape[0]
        out["ind_i"].append(e["_ind_i"] + n_atoms_tot)
        out["ind_j"].append(e["_ind_j"] + n_atoms_tot)
        out["mu_i"].append(e["_mu_i"])
        out["mu_j"].append(e["_mu_j"])
        out["offsets"].append(e["_offsets"])
        out["energy"].append(np.asarray(e["_energy"], dtype=np.float64).reshape(-1, 1))
        out["forces"].append(e["_forces"])
        out["positions"].append(e["_positions"])
        out["eweights"].append(e["_eweights"])
        out["fweights"].append(e["_fweights"])
        out["stress"].append(e["_stress"])
        out["sweights"].append(e["_sweights"])
        out["cell"].append(e["_cell"])
        out["pair_map"].append(np.full(n_pairs, j, dtype=np.int64))
        out["cell_map"].append(np.full(n_atoms, j, dtype=np.int64))
        out["e_map"].append(np.full(n_atoms, j, dtype=np.int64))
        n_atoms_tot += n_atoms
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}
