# pacemaker with stress fitting, multi-GPU / multi-node training and a PyTorch backend

A fork of [ICAMS/python-ace](https://github.com/ICAMS/python-ace) (pacemaker), with [ICAMS/TensorPotential](https://github.com/ICAMS/TensorPotential) (pacemaker's TensorFlow backend) in
`TensorPotential/`. Same model, `input.yaml` and potential files as upstream. The original
pacemaker README follows at the end.

## Install

Install the backend you use (or both).

PyTorch backend:
```bash
conda create -n pace python=3.11 && conda activate pace
pip install torch                   # the build for your GPU
pip install mpi4py                  # for multi-GPU; built against the cluster's MPI
pip install .                       # pacemaker
pip install ./pacetorch             # the PyTorch backend
```

TensorFlow backend:
```bash
conda create -n pace-tf python=3.10 && conda activate pace-tf     # TensorPotential needs Python < 3.11
pip install tensorflow
pip install mpi4py                  # built against the cluster's MPI
pip install .
pip install ./TensorPotential
```

**Updating:** `git pull`, then the same `pip install .` / `pip install ./pacetorch` / `pip install ./TensorPotential`
lines again.

## Backends

Without a `backend:` section pacemaker uses `pyace`, upstream's C++ evaluator on the CPU. The two GPU backends:

### PyTorch: `backend: evaluator: torch`

- `pacetorch/`: the ACE model in PyTorch, on a GPU or the CPU (`backend: gpu_config: {gpu_ind: -1}`).
- Energies, forces and stress, with the same loss and regularisation as upstream; radial smoothness
  (`w0_rad`/`w1_rad`/`w2_rad`) is not available on torch yet (use `tensorpot`).

### TensorFlow: `backend: evaluator: tensorpot`

- `TensorPotential/`: upstream's TensorFlow model plus the multi-GPU / multi-node fitting and the stress term
  described here; NVIDIA GPUs.

#### TensorFlow-only options for multi-GPU

- `PACEMAKER_MPI_SHARD=inbatch`: every rank evaluates every batch, but only its share of the structures in it. GPU
  memory per rank drops by about the number of ranks, so `batch_size` can grow by that factor. It keeps up with the
  default mode only while each GPU still gets about 100 structures per call.
- `backend: gpu_config: {memory_growth: true}`: TensorFlow takes GPU memory as it needs it instead of all at once.

## New input keys

```yaml
fit:
  interim_save_every_s: 30    # interim_potential_*.yaml at most every 30 s (0 = every iteration, as upstream)
```

## Stress in the loss

```yaml
fit:
  loss: { kappa: 0.3, L1_coeffs: 1e-8, L2_coeffs: 1e-8, w_stress: 1.0 }
```

- `w_stress` > 0 adds the stress residual (6 Voigt components) to the loss; 0 or absent: upstream behaviour.
- The dataset needs a `stress` column: 3x3 or 6 Voigt components per structure, eV/A^3, ASE sign convention.
- Only fully periodic structures with a finite stress label count; their weights are uniform.
- Both GPU backends fit stress. Stress errors are printed and written with the energy and force metrics.
- An unknown key in `fit: loss:` is an error, so a typo cannot silently do nothing.

## Multi-GPU / multi-node fitting

```bash
PACEMAKER_MPI=1 srun -n 8 --gpus-per-node 4 pacemaker input.yaml                  # PyTorch
PACEMAKER_MPI=1 srun -n 8 --gpus-per-node 4 bash -c 'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID pacemaker input.yaml'   # TF
```

- One MPI rank per GPU. PyTorch: each rank takes GPU (node-local rank) modulo (visible GPUs), from MPI with any
  launcher, and warns if two ranks share a GPU. TensorFlow (its own MPI code in `TensorPotential/`): the launch
  line gives each rank its GPU.
- Every rank runs the same optimizer. Each evaluates a contiguous share of the batches; loss and gradient are summed
  over the ranks, so every rank takes the same step. Only rank 0 writes files.
- Optimizers: BFGS, L-BFGS-B, CG.
- `PACEMAKER_MPI_REPRODUCIBLE=1`: sums in a fixed rank order instead of the MPI library's (results still depend on
  the number of ranks).
- Without `PACEMAKER_MPI=1` nothing changes: one process runs the upstream code.

## BFGS: checkpoint and pause (PyTorch backend)

```yaml
fit:
  bfgs_checkpoint: bfgs_ckpt.npz   # save the BFGS state (point, inverse Hessian, ...)
  bfgs_checkpoint_every: 10        # iterations between saves
```

- BFGS runs natively: scipy's line search and stopping rules with an O(N^2) update instead of scipy's O(N^3).
  `fit: options: {native_bfgs: false}` uses scipy's.
- `PACE_DEADLINE` (unix time of the job's end) and `PACE_MARGIN` (seconds before it, default 240): the fit saves and
  stops before the deadline and leaves `BFGS_PAUSED`. The next run with the same checkpoint continues with the saved
  curvature, from the potential you give it (for example the last interim potential).

## Credits and licence

pacemaker / python-ace and TensorPotential are by Yury Lysogorskiy, Anton Bochkarev, Ralf Drautz et al. (ICAMS,
Ruhr University Bochum); please cite their papers (below) when you use this code. This fork keeps the original
**Academic Software Licence** (`LICENSE.md`): academic, non-commercial use; derivative works under the same licence.

---

*The original pacemaker README follows.*

# Description

This repository contains `pacemaker` - a tool for fitting of interatomic potentials in a general nonlinear Atomic Cluster Expansion (ACE) form.
It is based on the `pyace` (this repo) and `tensorpotential`([another repo](https://github.com/ICAMS/TensorPotential)) Python packages.

# Installation

Please follow [these](https://pacemaker.readthedocs.io/en/latest/pacemaker/install/) instructions.


# Documentation

Please see the `docs` folder for documentation or check [online documentation](https://pacemaker.readthedocs.io/)


# Citation

Please cite following papers:

* [Bochkarev, A., Lysogorskiy, Y., Menon, S., Qamar, M., Mrovec, M. and Drautz, R. Efficient parametrization of the atomic cluster expansion. Physical Review Materials 6(1) 013804 (2022)](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.6.013804)
* [Lysogorskiy, Y., Oord, C. v. d., Bochkarev, A., Menon, S., Rinaldi, M., Hammerschmidt, T., Mrovec, M., Thompson, A., Csányi, G., Ortner, C. and  Drautz, R. Performant implementation of the atomic cluster expansion (PACE) and application to copper and silicon. npj Computational Materials 7(1), 1-12 (2021)](https://www.nature.com/articles/s41524-021-00559-9)
* [Drautz, R. Atomic cluster expansion for accurate and transferable interatomic potentials. Physical Review B, 99(1), 014104 (2019)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104)

BibTeX:

```bibtex
@article{bochkarev2022efficient,
  title={Efficient parametrization of the atomic cluster expansion},
  author={Bochkarev, Anton and Lysogorskiy, Yury and Menon, Sarath and Qamar, Minaam and Mrovec, Matous and Drautz, Ralf},
  journal={Physical Review Materials},
  volume={6},
  number={1},
  pages={013804},
  year={2022},
  publisher={APS}
}

@article{lysogorskiy2021performant,
  title={Performant implementation of the atomic cluster expansion (PACE) and application to copper and silicon},
  author={Lysogorskiy, Yury and Oord, Cas van der and Bochkarev, Anton and Menon, Sarath and Rinaldi, Matteo and Hammerschmidt, Thomas and Mrovec, Matous and Thompson, Aidan and Cs{\'a}nyi, G{\'a}bor and Ortner, Christoph and others},
  journal={npj Computational Materials},
  volume={7},
  number={1},
  pages={1--12},
  year={2021},
  publisher={Nature Publishing Group}
}

@article{drautz2019atomic,
  title={Atomic cluster expansion for accurate and transferable interatomic potentials},
  author={Drautz, Ralf},
  journal={Physical Review B},
  volume={99},
  number={1},
  pages={014104},
  year={2019},
  publisher={APS}
}
```
