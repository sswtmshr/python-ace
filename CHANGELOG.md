# What changed, commit by commit

Starting point: upstream pacemaker (python-ace `d1c213a`, TensorPotential `1e44b25`). Each entry: the problem, the
change, and what was measured.

Benchmark, unless noted: the SiC set (3735 structures, 538k atoms, 5640 coefficients, batch 94); "s/eval" = one
evaluation of the loss and its gradient over the whole set, what BFGS needs per step.

## At a glance

| commit | date | one GPU: s/eval | one GPU: memory | several GPUs |
|---|---|---|---|---|
| upstream TensorFlow (start) | | 20.0 | 22.9 GiB | single GPU only |
| 1 multinode (TensorFlow) | 2026-08-31 | | | 6.3 s/eval on 4 A100s |
| 3 PyTorch backend | 2026-10-06 | 13.7* | 32.4 GiB* | 3.75 s/eval on 4 A100s |
| 4 forward-over-reverse (trial) | 2026-10-06 | 21.45 | 30.87 GiB | |
| 6 back to reverse-over-reverse, int32 indices | 2026-10-06 | | 30.75 GiB** | |

- \* same job as the upstream row (one A100); every later one-GPU row ran on another node.
- \*\* measured after 7 and 8, which do not change memory (32.39 -> 30.75 GiB: the int32 index arrays).

### 1. Multi-GPU / multi-node fitting with MPI, TensorFlow backend (2026-08-31)

- `PACEMAKER_MPI=1`: every rank runs the same optimizer and takes a contiguous share of the batches; loss and
  gradient are summed over the ranks, so every rank takes the identical step. Only rank 0 writes. BFGS, L-BFGS-B
  and CG only. `PACEMAKER_MPI_REPRODUCIBLE=1` sums in a fixed rank order.
- `PACEMAKER_MPI_SHARD=inbatch`: each rank takes a slice of every batch instead, so memory per GPU drops to ~1/ranks;
  it keeps up only while each GPU still gets ~100 structures per call (measured 2026-09-23).
- `backend: gpu_config: {memory_growth: true}`; fixes under MPI: early stopping deadlocked the other ranks, a missing
  metrics default crashed, `predict()` ran out of memory at the full batch size.
- Measured: 4 A100s 6.3 s/eval (one A100: 20.0).

### 2. Stress in the loss (2026-10-06)

- `fit: loss: w_stress` compares the 6 Voigt components (eV/A^3, ASE sign) of sigma = sym(h^T dE/dh)/V at fixed
  fractional coordinates, in one tape pass; weights uniform over the periodic structures with a stress label.
- Cost: +1-3% time. With `w_stress: 0` the behaviour is upstream's.

### 3. PyTorch backend, with MPI (2026-10-06)

- Problem: TensorFlow rebuilt index bookkeeping and batch data on every call (~2.6 s/eval of host work); upstream's
  pins (numpy, pandas, scikit-learn, Python) and `distutils`/`pkg_resources` blocked current Python.
- Change: `pacetorch`, a drop-in backend with the same coefficient layout, so every file pacemaker writes is
  unchanged. Indices are built once per batch, batch data is uploaded to the GPU once, and the A-basis keeps only the
  channels the product basis uses (62%). MPI as in 1, for the torch backend; each rank's GPU is pinned by the launch
  line (`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`).
- Effect: 20.0 -> **13.7 s/eval** (same A100, same job); 4 A100s **3.75 s/eval** vs **6.3** for TensorFlow (1).
  Agreement with TensorFlow: energies 3e-14 eV/at, forces 2e-13 eV/A, stress 4e-12 GPa, gradient 9e-14. 4 ranks
  reproduce 1 rank to 1e-14.

### 4. Loss gradient by forward-over-reverse differentiation: a trial (2026-10-06)

- The gradient of the force term is a second derivative. Forward-over-reverse (a forward-mode derivative pushed
  through the backward pass) instead of reverse-over-reverse (the backward pass differentiated again); both exact.
- Measured, one A100, batch 94: 21.45 s/eval and 30.87 GiB, against 13.73 s/eval and 32.39 GiB for
  reverse-over-reverse: 57% slower for 5% less memory (separate jobs). Undone in 6.

### 5. Native BFGS: O(N^2) update, checkpoint, deadline pause (2026-10-06)

- Problem: scipy's inverse-Hessian update is two dense N x N products (O(N^3)), and fits outlive a job.
- Change: scipy's line search and stopping rules with the update multiplied out, O(N^2); `fit: bfgs_checkpoint`;
  a pause before `PACE_DEADLINE`.
- Effect: same iterates and statuses as scipy; checkpoint resume bit-identical; the update at N = 9640: 0.52 s.

### 6. Back to reverse-over-reverse; cell volume; int32 indices; GPU per rank from MPI (2026-10-06)

- The forward-over-reverse gradient of 4 is removed.
- Cell volume by triple product (`torch.linalg.det` failed on some GPUs).
- int32 index arrays: half the neighbour-list memory, 32.39 -> 30.75 GiB.
- Each rank picks its GPU itself, from its MPI node-local rank (any launcher, no pinning in the launch line), with a
  warning when two ranks share a GPU.
- Bug fix: a numeric `data: reference_energy` was silently replaced by 0.

### 7. Per-structure stress weights (2026-10-07)

- Weights follow the weighting policy (GRACE convention); a dataset `w_stress` column wins. Fixes the first PyTorch
  version, which overwrote that column with 1/N.

### 8. MPI robustness and small upstream fixes (2026-10-07)

- An uncaught error on any rank aborts the job instead of leaving the others waiting; only rank 0 deletes old
  metrics files (restarts in a run directory killed ranks) and prints the metric tables.
- `test_size: 0` gives no test set instead of a crash; ASE `atoms.calc` replaces `set_calculator`.

### 9. Speed experiments, off by default (2026-10-07)

- A dense A-basis by batched matrix products: -38% memory, +46% time. Products in complex numbers: -3.5% time at
  first. Both exact; both removed in 11 (slower and larger than the prefix tree).
