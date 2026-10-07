"""FitTorchPotential: drop-in for tensorpotential.fit.FitTensorPotential (same attributes / callbacks / metrics
dicts that pyace's FitBackendAdapter and GeneralACEFit use), on the torch backend.

Differences from the TF fitter (behaviour of the optimiser is unchanged):
  * batches are moved to the device once per fit() (DeviceBatch), not converted from numpy at every evaluation;
  * per evaluation: loss and gradient are accumulated on the device and copied to the host once;
  * StopIteration raised in a callback (pacemaker early stopping) ends the fit with the last accepted x instead of
    propagating through scipy and aborting the run;
  * final test metrics are computed for the returned coefficients; test batches are prepared once;
  * no BBasisConfiguration is rebuilt in the callback just to be discarded.

Multi-GPU / multi-node (PACEMAKER_MPI=1, one rank per GPU, as in the multinode TF fork): every rank runs the same
optimiser on the full-dataset loss; each rank owns a contiguous slice of the batch list, loss and gradient are summed
over ranks and broadcast from rank 0 (bit-identical optimiser input on all ranks), errors are gathered in global
order. Callbacks (metrics files, interim potential, early stopping) run on rank 0; a StopIteration there is
broadcast so that all ranks stop on the same iteration.
"""
import logging
import time

import numpy as np
import torch
from scipy.optimize import OptimizeResult, minimize

from pacetorch.constants import *
from pacetorch.data import batching_data
from pacetorch.fitmetrics import FitMetrics

import os

_OOM = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
# test hook "rank:n": that MPI rank raises out-of-memory in its n-th collective GPU step (0 = first batch upload), once
_INJECT_OOM = [int(v) for v in os.environ.get("PACETORCH_INJECT_OOM", "-1:-1").split(":")]
_collective_calls = [0]

try:   # framework-agnostic data-parallel layer (PACEMAKER_MPI=1); identity when disabled
    from pyace import mpi
except ImportError:  # pragma: no cover
    mpi = None

log = logging.getLogger(__name__)

NUMPY_OPTIMIZERS = ["BFGS", "L-BFGS-B", "CG"]
TORCH_OPTIMIZERS = ["Adam", "SGD"]


def _voigt_residual(s_pred, s_ref):
    ds = s_pred.reshape(-1, 3, 3) - s_ref.reshape(-1, 3, 3)
    return np.stack([ds[:, 0, 0], ds[:, 1, 1], ds[:, 2, 2], ds[:, 1, 2], ds[:, 0, 2], ds[:, 0, 1]], axis=1)


class FitTorchPotential:
    def __init__(self, tensorpot, display_step=20, batch_size=10, eager=False):
        self.tensorpot = tensorpot           # name kept: pyace reads fitter.tensorpot.potential
        self.ncoefs = self.tensorpot.potential.ncoef
        self.fit_coefs = None
        self.jacobian_factor = None
        self.loss_history = []
        self.iter_num = 0
        self.eval_count = 0
        self.last_loss = 0
        self.global_callback = None
        self.display_step = display_step
        self.data_df = None
        self.test_df = None
        self.res_opt = None
        self.batch_size = batch_size
        self.n_batches = None
        self.metrics = None
        self.test_metrics = None
        self.fit_metrics_data_dict = {}
        self.last_fit_metric_data = None
        self.last_test_metric_data = None
        self.fit_metric_callback = None
        self.test_metric_callback = None
        self.nfuncs = None
        self.eval_time = 0
        self._test_batches = None
        self._last_x = None

    # -------------------------------------------------------------------------------------------------- setup
    def prepare_batches(self, batches):
        return [self.tensorpot.device_batch(b) for b in batches]

    @staticmethod
    def _dist():
        return mpi is not None and mpi.is_distributed()

    def _local(self, batches):
        """this rank's contiguous slice of the batch list (all batches without MPI)"""
        if not self._dist():
            return batches
        part = np.array_split(np.arange(len(batches)), mpi.world_size())[mpi.rank()]
        return [batches[i] for i in part]

    def _collective(self, fn, *args):
        """Run this rank's share of GPU work. With MPI, out-of-GPU-memory is decided collectively: if fn runs out of
        memory on any rank, every rank raises it, so fitadapter's batch-size reduction restarts all ranks together
        (one rank alone restarting with a smaller batch desynchronised the collectives)."""
        if not self._dist():
            return fn(*args)
        err, out = None, None
        try:
            _collective_calls[0] += 1
            if _INJECT_OOM == [mpi.rank(), _collective_calls[0] - 1]:
                raise _OOM("injected out-of-memory on MPI rank {} (PACETORCH_INJECT_OOM)".format(mpi.rank()))
            out = fn(*args)
        except _OOM as e:
            err = e
        if mpi.allreduce_max(int(err is not None)):
            out = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise err if err is not None else _OOM("CUDA out of memory on another MPI rank")
        return out

    def _root_call(self, fn, *args, **kwargs):
        """run a callback on rank 0 only; a StopIteration it raises (early stopping) stops every rank"""
        if not self._dist():
            return fn(*args, **kwargs)
        stop = None
        if mpi.is_root():
            try:
                fn(*args, **kwargs)
            except StopIteration as e:
                stop = str(e) or "stop"
        stop = mpi.bcast(stop)
        if stop is not None:
            raise StopIteration(stop)

    def _make_metrics(self, batches):
        ls = self.tensorpot.loss_specs
        m = FitMetrics(np.vstack([b["eweights"] for b in batches]), np.vstack([b["fweights"] for b in batches]),
                       ls[LOSS_ENERGY_FACTOR], ls[LOSS_FORCE_FACTOR], self.ncoefs)
        m.s_scale = ls[LOSS_STRESS_FACTOR]
        m.nfuncs = self.nfuncs
        return m

    def fit(self, df, test_df=None, batch_size=100, optimizer="BFGS", coefs=None, niter=100, jacobian_factor=None,
            callback=None, save_coefs=False, options=None, fit_metric_callback=None, test_metric_callback=None):
        if fit_metric_callback is not None:
            self.fit_metric_callback = fit_metric_callback
        if test_metric_callback is not None:
            self.test_metric_callback = test_metric_callback
        self.batch_size = batch_size
        if self._dist():   # collective, every rank is here: warn if the launcher put several ranks on one GPU
            from pacetorch import check_shared_devices
            check_shared_devices(self.tensorpot.device, mpi)
        self.data_df = df[df["tp_atoms"].notna()].reset_index(drop=True)
        t = time.time()
        np_batches = batching_data(self.data_df, batch_size=batch_size)
        self.n_batches = len(np_batches)
        self.metrics = self._make_metrics(np_batches)
        batches = self._collective(self.prepare_batches, self._local(np_batches))
        del np_batches
        if test_df is not None:
            self.test_df = test_df[test_df["tp_atoms"].notna()].reset_index(drop=True)
            np_test = batching_data(self.test_df, batch_size=batch_size)
            self.test_metrics = self._make_metrics(np_test)
            self._test_batches = self._collective(self.prepare_batches, self._local(np_test))
        log.info("Batches on {}: {} of {} train{}{} ({:.1f} s)".format(
            self.tensorpot.device, len(batches), self.n_batches,
            " / {} test".format(len(self._test_batches)) if test_df is not None else "",
            " [MPI rank {}/{}]".format(mpi.rank(), mpi.world_size()) if self._dist() else "", time.time() - t))

        self.global_callback = callback
        if jacobian_factor is not None:
            self.jacobian_factor = np.asarray(jacobian_factor, dtype=np.float64)
            if len(self.jacobian_factor) != self.ncoefs:
                raise ValueError("Size of the Jacobian factor ({}) != number of coefficients ({})".format(
                    len(self.jacobian_factor), self.ncoefs))
        if coefs is None:
            coefs = self.tensorpot.potential.get_coefs()
        elif len(coefs) != self.ncoefs:
            raise ValueError("Number of provided coefficients {} != {} in the model".format(len(coefs), self.ncoefs))
        coefs = np.asarray(coefs, dtype=np.float64)
        self._last_x = coefs.copy()

        t0 = time.time()
        if optimizer in NUMPY_OPTIMIZERS:
            options = dict(options or {"gtol": 1e-8, "disp": True})
            options["maxiter"] = niter
            log.info("Minimizer options: {}".format(options))
            self.fit_metrics_data_dict = {}
            native = bool(options.pop("native_bfgs", True)) and optimizer == "BFGS"
            try:
                if native:   # O(N^2) update, optional checkpoint/pause (pyace.bfgs); same iterates as scipy BFGS
                    from pyace.bfgs import minimize_bfgs
                    res_opt = minimize_bfgs(self.fit_func, coefs, args=(batches,), callback=self.callback,
                                            options=options, tag=self.tensorpot.loss_specs[LOSS_FORCE_FACTOR])
                else:
                    options.pop("checkpoint", None), options.pop("checkpoint_every", None)
                    res_opt = minimize(self.fit_func, coefs, method=optimizer, jac=True, args=(batches,),
                                       options=options, callback=self.callback)
            except StopIteration as e:   # pacemaker early stopping raised inside the callback
                log.info("Optimization stopped by callback: {}".format(e))
                res_opt = OptimizeResult(x=self._last_x.copy(), success=True, status=0, message=str(e),
                                         fun=self.last_loss, nit=self.iter_num, nfev=self.eval_count)
            self.res_opt = res_opt
            self.fit_coefs = np.asarray(res_opt.x)
            self.tensorpot.potential.set_coefs(self.fit_coefs)
            self.process_test_metric()
        elif optimizer in TORCH_OPTIMIZERS:
            c = self.tensorpot.potential
            c.set_coefs(coefs)
            opt_cls = torch.optim.Adam if optimizer == "Adam" else torch.optim.SGD
            lr = float((options or {}).get("learning_rate", 1e-3))
            param = c.coefs.detach().clone().requires_grad_(True)
            opt = opt_cls([param], lr=lr)
            for _ in range(niter):
                order = np.random.permutation(len(batches))
                for i in order:
                    c.set_coefs(param.detach().cpu().numpy())
                    _, g, _, _, _ = self.tensorpot.loss_and_grad(batches[i])
                    param.grad = g.detach().clone()
                    opt.step()
                x = param.detach().cpu().numpy()
                self.fit_func(x, batches)
                self.callback(x)
            self.fit_coefs = param.detach().cpu().numpy()
            self.res_opt = OptimizeResult(x=self.fit_coefs, success=True, nit=niter, fun=self.last_loss)
            c.set_coefs(self.fit_coefs)
        else:
            raise ValueError("Unknown optimizer `{}` for the torch backend. Should be one of {}".format(
                optimizer, NUMPY_OPTIMIZERS + TORCH_OPTIMIZERS))
        if save_coefs:
            np.savetxt("Fit_coefficients_step_{0}".format(self.eval_count), self.fit_coefs)
        self.update_last_metrics(self.fit_coefs)
        log.info("Fitting took {0:6.2f} seconds".format(time.time() - t0))

    # ------------------------------------------------------------------------------------------- loss / grad
    def _run(self, x, batches):
        """One full pass: loss, gradient (host numpy), and per-structure errors."""
        tot_loss, tot_grad, outs = self._collective(self._local_pass, x, batches)
        if self._dist():
            tot_loss = mpi.allreduce_scalar_sum(tot_loss)
            tot_grad = mpi.allreduce_array_sum(tot_grad)
            # identical optimiser input on every rank, whatever the MPI reduction algorithm
            tot_loss, tot_grad = mpi.bcast((tot_loss, tot_grad) if mpi.is_root() else None)
        de, de_pa, df, nat, ds, sw = [], [], [], [], [], []
        for db, (e, f, s) in zip(batches, outs):
            b = db.np
            n = np.bincount(b["e_map"], minlength=len(b["energy"])).reshape(-1, 1)
            d_e = b["energy"].reshape(-1, 1) - e.cpu().numpy()
            de.append(d_e)
            de_pa.append(d_e / n)
            df.append(b["forces"] - f.cpu().numpy())
            nat.append(n)
            if s is not None:
                ds.append(_voigt_residual(s.cpu().numpy(), b["stress"]))
                sw.append(b["sweights"].reshape(-1, 1))
        fit_s = self.tensorpot.fit_stress
        st = lambda l, w: np.vstack(l) if l else np.zeros((0, w))
        errs = [st(de, 1), st(de_pa, 1), st(df, 3), st(nat, 1)]
        serrs = [st(ds, 6), st(sw, 1)] if fit_s else None
        if self._dist():   # global order: ranks own contiguous batch slices
            errs = [mpi.allgather_vstack(a) for a in errs]
            if serrs is not None:
                serrs = [mpi.allgather_vstack(a) for a in serrs]
        return tot_loss, tot_grad, tuple(errs), (tuple(serrs) if serrs is not None else None)

    def _local_pass(self, x, batches):
        """this rank's batches: summed loss and gradient (host), per-batch predictions"""
        tp = self.tensorpot
        tp.potential.set_coefs(x)
        tot_loss, tot_grad = None, None
        outs = []
        for db in batches:
            loss, g, e, f, s = tp.loss_and_grad(db)
            tot_loss = loss if tot_loss is None else tot_loss + loss
            tot_grad = g if tot_grad is None else tot_grad + g
            outs.append((e, f, s))
        # single device -> host transfer
        tot_loss = float(tot_loss.cpu()) if tot_loss is not None else 0.
        tot_grad = tot_grad.cpu().numpy() if tot_grad is not None else np.zeros(self.ncoefs)
        return tot_loss, tot_grad, outs

    def fit_func(self, x, batches):
        t0 = time.time()
        tot_loss, tot_jac, errs, serrs = self._run(x, batches)
        self.fit_coefs = x
        self.eval_count += 1
        self.eval_time = time.time() - t0
        self.metrics.regs = self.get_reg_components()
        self.metrics.reg_weights = self.get_reg_weights()
        self.metrics.compute_metrics(*errs, dataframe=self.data_df)
        if serrs is not None:
            self.metrics.compute_stress_metrics(*serrs)
        self.metrics.loss = tot_loss
        self.metrics.eval_time = self.eval_time
        cur = self.metrics.to_FitMetricsDict()
        cur["eval_count"] = self.eval_count
        cur["iter_num"] = self.iter_num
        self.fit_metrics_data_dict[hash(np.asarray(x).tobytes())] = cur
        self.last_fit_metric_data = cur
        if self.iter_num == 0:
            if self.fit_metric_callback is not None:
                self._root_call(self.fit_metric_callback, cur)
            elif not self._dist() or mpi.is_root():
                print_detailed_metrics(cur, title="Initial state:")
                print_extended_metrics(cur, title="INIT STATS")
            self.process_test_metric(title="TEST INIT STATS")
            self.iter_num += 1
        self.loss_history.append(tot_loss)
        self.last_loss = tot_loss
        if self.jacobian_factor is not None:
            tot_jac = tot_jac * self.jacobian_factor
        return tot_loss, tot_jac

    # -------------------------------------------------------------------------------------------- callbacks
    def callback(self, x, *args, **kwargs):
        self.metrics.record_time(self.eval_time)
        self._last_x = np.array(x, copy=True)
        data = self.update_last_metrics(x)
        # keep only the entry of the accepted point: the dict is keyed by every evaluated x
        self.fit_metrics_data_dict = {hash(np.asarray(x).tobytes()): data}
        if self.fit_metric_callback is not None:
            self._root_call(self.fit_metric_callback, data)
        elif not self._dist() or mpi.is_root():
            if self.iter_num % self.display_step == 0:
                print_extended_metrics(data)
            else:
                print_detailed_metrics(data)
        if self.iter_num % self.display_step == 0:
            self.process_test_metric(updating_coefs=x)
        self.iter_num += 1
        if self.global_callback is not None:
            self._root_call(self.global_callback, x, *args, **kwargs)

    def update_last_metrics(self, x):
        d = self.fit_metrics_data_dict[hash(np.asarray(x).tobytes())]
        d["iter_num"] = self.iter_num
        d["eval_count"] = len(self.loss_history)
        self.metrics.from_FitMetricsDict(d)
        self.last_fit_metric_data = d
        return d

    def process_test_metric(self, title="TEST_STATS", updating_coefs=None):
        if self.test_df is None:
            return
        d = self.compute_test_metric(updating_coefs=updating_coefs)
        d["eval_count"] = self.eval_count
        d["iter_num"] = self.iter_num
        self.last_test_metric_data = d
        if self.test_metric_callback is not None:
            self._root_call(self.test_metric_callback, d)
        elif not self._dist() or mpi.is_root():
            print_extended_metrics(d, title=title)

    def compute_test_metric(self, updating_coefs=None):
        x = self.tensorpot.potential.get_coefs() if updating_coefs is None else updating_coefs
        keep = self.tensorpot.potential.get_coefs()
        _, _, errs, serrs = self._run(np.asarray(x), self._test_batches)
        self.tensorpot.potential.set_coefs(keep)
        tm = self.test_metrics
        tm.regs = self.get_reg_components()
        tm.reg_weights = self.get_reg_weights()
        tm.compute_metrics(*errs, dataframe=self.test_df)
        if serrs is not None:
            tm.compute_stress_metrics(*serrs)
        d = tm.to_FitMetricsDict()
        d["loss"] = d["e_loss_contrib"] + d["f_loss_contrib"] + d["l1_reg_contrib"] + d["l2_reg_contrib"] + \
            d.get("s_loss_contrib", 0.) + sum(d["extra_regularization_contrib"])
        d["iter_num"] = self.iter_num
        return d

    # ------------------------------------------------------------------------------------------- prediction
    def predict(self, datadf=None):
        df = self.data_df if datadf is None else datadf[datadf["tp_atoms"].notna()].reset_index(drop=True)
        out = {k: [] for k in ("energy_pred", "forces_pred", "stress_pred", "de", "de_pa", "df", "total_na")}
        for b in batching_data(df, batch_size=self.batch_size):
            db = self.tensorpot.device_batch(b)
            e, f, s = self.tensorpot.evaluate(db)
            e, f, s = e.cpu().numpy(), f.cpu().numpy(), s.cpu().numpy()
            n = np.bincount(b["e_map"], minlength=len(b["energy"]))
            split = np.cumsum(n)[:-1]
            d_e = b["energy"].reshape(-1, 1) - e
            out["energy_pred"] += list(e.reshape(-1))
            out["forces_pred"] += list(np.split(f, split))
            out["stress_pred"] += list(s.reshape(-1, 3, 3))
            out["de"] += list(d_e.reshape(-1))
            out["de_pa"] += list((d_e / n.reshape(-1, 1)).reshape(-1))
            out["df"] += list(np.split(b["forces"] - f, split))
            out["total_na"].append(n.reshape(-1, 1))
        return out

    # ------------------------------------------------------------------------------------------------- misc
    def get_fitting_data(self):
        return self.data_df

    def get_reg_components(self):
        regs = self.tensorpot.reg_components
        return np.vstack([np.asarray(r.cpu() if torch.is_tensor(r) else r, dtype=np.float64).reshape(-1, 1)
                          for r in regs])

    def get_reg_weights(self):
        ls = self.tensorpot.loss_specs
        return np.vstack([np.array(ls[L1_REG]).reshape(-1, 1), np.array(ls[L2_REG]).reshape(-1, 1),
                          np.array(ls[AUX_LOSS_FACTOR]).reshape(-1, 1)]) * self.n_batches

    def get_fitted_coefficients(self):
        if self.eval_count > 0:
            return self.fit_coefs
        raise AttributeError("Coefficients have not been fitted yet")

    def print_detailed_metrics(self, title="Iteration:"):
        print_detailed_metrics(self.last_fit_metric_data, title=title)

    def print_extended_metrics(self, title="FIT_STATS"):
        print_extended_metrics(self.last_fit_metric_data, title=title)


def print_extended_metrics(fit_metrics_dict, title="FIT_STATS"):
    # (self, iter_num, total_loss, reg_comps, reg_weights, title='FIT STATS', nfuncs=None):

    iter_num = fit_metrics_dict["iter_num"]
    total_loss = fit_metrics_dict["loss"]

    str0 = '\n' + '-' * 44 + title + '-' * 44 + '\n'
    str1 = '{prefix:<11} #{iter_num:<4}'.format(prefix='Iteration:', iter_num=iter_num)
    str1 += '{prefix:<8}'.format(prefix='Loss:')
    str1 += '{prefix:>8} {tot_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Total: ', tot_loss=total_loss, fr=100)
    str1 += '\n'

    fr = fit_metrics_dict["e_loss_contrib"] / total_loss * 100 if total_loss > 0 else 0
    str2 = '{prefix:>33} {e_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Energy: ',
                                                              e_loss=fit_metrics_dict["e_loss_contrib"],
                                                              fr=fr)
    str2 += '\n'

    fr = fit_metrics_dict["f_loss_contrib"] / total_loss * 100 if total_loss > 0 else 0
    str3 = '{prefix:>33} {f_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Force: ',
                                                              f_loss=fit_metrics_dict["f_loss_contrib"],
                                                              fr=fr)
    str3 += '\n'
    if fit_metrics_dict.get("s_scale", 0):
        fr = fit_metrics_dict["s_loss_contrib"] / total_loss * 100 if total_loss > 0 else 0
        str3 += '{prefix:>33} {s_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Stress: ',
                                                                   s_loss=fit_metrics_dict["s_loss_contrib"], fr=fr)
        str3 += '  RMSE {:.2f} GPa  MAE {:.2f} GPa  MAX {:.2f} GPa\n'.format(
            fit_metrics_dict["rmse_s"], fit_metrics_dict["mae_s"], fit_metrics_dict["max_abs_s"])

    l1 = fit_metrics_dict["l1_reg_contrib"]
    l2 = fit_metrics_dict["l2_reg_contrib"]

    fr = l1 / total_loss * 100 if total_loss != 0 else 0
    str4 = '{prefix:>33} {l1:>1.4e} ({fr:3.0f}%) '.format(prefix='L1: ', l1=l1, fr=fr)
    str4 += '\n'
    fr = l2 / total_loss * 100 if total_loss != 0 else 0
    str4 += '{prefix:>33} {l2:>1.4e} ({fr:3.0f}%) '.format(prefix='L2: ', l2=l2, fr=fr)
    str4 += '\n'

    reg_comps = fit_metrics_dict["extra_regularization_contrib"]
    str5 = ''
    for i, comp in enumerate(reg_comps):
        str5 += '{prefix:>33} '.format(prefix='Smooth_w{}: '.format(i + 1))
        str5 += '{s1:>1.4e} '.format(s1=comp)
        str5 += '({fr:3.0f}%) '.format(fr=comp / total_loss * 100)
        str5 += '\n'

    nfuncs = fit_metrics_dict.get('nfuncs')
    ncoefs = fit_metrics_dict.get('ncoefs')

    if nfuncs is None:
        line = 'Number of params.: '
    else:
        line = 'Number of params./funcs: '
    str6 = '{prefix:>20}'.format(prefix=line) + '{ncoefs:>6d}'.format(ncoefs=ncoefs)
    if nfuncs is not None:
        str6 += '/{nfuncs:<6d}'.format(nfuncs=nfuncs)

    avg_t = fit_metrics_dict["eval_time"] / fit_metrics_dict["nat"]  # in sec/atom
    str6 += '{prefix:>42}'.format(prefix='Avg. time: ') + \
            '{avg_t:>10.2f} {un:<6}'.format(avg_t=avg_t * 1e6, un='mcs/at')

    str6 += '\n' + '-' * 97 + '\n'
    str_loss = str0 + str1 + str2 + str3 + str4 + str5 + str6
    ##############################
    er_str_h = '{:>9}'.format('') + \
               '{:^22}'.format('Energy/at, meV/at') + \
               '{:^22}'.format('Energy_low/at, meV/at') + \
               '{:^22}'.format('Force, meV/A') + \
               '{:^22}\n'.format('Force_low, meV/A')

    er_rmse = '{prefix:>9} '.format(prefix='RMSE: ')
    er_rmse += '{:>14.2f}'.format(fit_metrics_dict["rmse_epa"] * 1e3) + \
               '{:>21.2f}'.format(fit_metrics_dict["low_rmse_epa"] * 1e3) + \
               '{:>21.2f}'.format(fit_metrics_dict["rmse_f"] * 1e3) + \
               '{:>24.2f}\n'.format(fit_metrics_dict["low_rmse_f"] * 1e3)
    er_mae = '{prefix:>9} '.format(prefix='MAE: ')
    er_mae += '{:>14.2f}'.format(fit_metrics_dict["mae_epa"] * 1e3) + \
              '{:>21.2f}'.format(fit_metrics_dict["low_mae_epa"] * 1e3) + \
              '{:>21.2f}'.format(fit_metrics_dict["mae_f"] * 1e3) + \
              '{:>24.2f}\n'.format(fit_metrics_dict["low_mae_f"] * 1e3)
    er_max = '{prefix:>9} '.format(prefix='MAX_AE: ')
    er_max += '{:>14.2f}'.format(fit_metrics_dict["max_abs_epa"] * 1e3) + \
              '{:>21.2f}'.format(fit_metrics_dict["low_max_abs_epa"] * 1e3) + \
              '{:>21.2f}'.format(fit_metrics_dict["max_abs_f"] * 1e3) + \
              '{:>24.2f}\n'.format(fit_metrics_dict["low_max_abs_f"] * 1e3)
    er_str = er_str_h + er_rmse + er_mae + er_max + '-' * 97  # + '\n'
    log.info(str_loss + er_str)


def print_detailed_metrics(fit_metrics_dict, title='Iteration:'):
    # fit_metrics_dict
    iter_num = fit_metrics_dict["iter_num"]
    total_loss = fit_metrics_dict["loss"]
    avg_t = fit_metrics_dict["eval_time"] / fit_metrics_dict["nat"]  # in sec/atom
    log.info('{:<12}'.format(title) +
             "#{iter_num:<5}".format(iter_num=iter_num) +
             '{:<14}'.format('({numeval} evals):'.format(numeval=fit_metrics_dict["eval_count"])) +
             '{:>10}'.format('Loss: ') + "{loss: >3.6f}".format(loss=total_loss) +
             '{str1:>21}{rmse_epa:>.2f} ({low_rmse_e:>.2f}) meV/at' \
             .format(str1=" | RMSE Energy(low): ",
                     rmse_epa=1e3 * fit_metrics_dict["rmse_epa"],
                     low_rmse_e=1e3 * fit_metrics_dict["low_rmse_epa"]) +
             '{str3:>16}{rmse_f:>.2f} ({low_rmse_f:>.2f}) meV/A' \
             .format(str3=" | Forces(low): ",
                     rmse_f=1e3 * fit_metrics_dict["rmse_f"],
                     low_rmse_f=1e3 * fit_metrics_dict["low_rmse_f"]) +
             (' | Stress: {:.2f} GPa'.format(fit_metrics_dict["rmse_s"]) if fit_metrics_dict.get("s_scale", 0)
              else '') +
             ' | Time/eval: {:>6.2f} mcs/at'.format(avg_t * 1e6))
