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
