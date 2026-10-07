import logging
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize, dual_annealing

from tensorpotential.constants import *
from tensorpotential.utils.utilities import batching_data
from tensorpotential.fitmetrics import FitMetrics
from tensorpotential import mpi

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _split_contiguous(seq):
    """This rank's contiguous chunk of `seq` (list). Whole list if MPI off/1 rank.
    Contiguous (not strided) so that a rank-ordered gather reconstructs the
    original element order with no reindexing."""
    w = mpi.world_size()
    if w == 1:
        return list(seq)
    parts = np.array_split(np.arange(len(seq)), w)
    idx = parts[mpi.rank()]
    return [seq[i] for i in idx]


def _combine_error_lists(de, de_pa, df, total_na):
    """vstack the local per-structure error lists, then (if distributed)
    allgather across ranks in rank order so every rank holds the full,
    original-order arrays -> FitMetrics.compute_metrics is then identical to
    the single-process result."""
    de = np.vstack(de) if de else np.zeros((0, 1))
    de_pa = np.vstack(de_pa) if de_pa else np.zeros((0, 1))
    df = np.vstack(df) if df else np.zeros((0, 3))
    total_na = np.vstack(total_na) if total_na else np.zeros((0, 1))
    if mpi.is_distributed():
        de = mpi.allgather_vstack(de)
        de_pa = mpi.allgather_vstack(de_pa)
        df = mpi.allgather_vstack(df)
        total_na = mpi.allgather_vstack(total_na)
    return de, de_pa, df, total_na


EV_A3_TO_GPA = 160.21766208


def stress_errors(b, s_pred):
    """per-structure stress residuals (pred - ref, 6 Voigt xx yy zz yz xz xy, eV/A^3) and loss weights of batch b"""
    ds = np.asarray(s_pred).reshape(-1, 3, 3) - b['stress'].reshape(-1, 3, 3)
    ds_v = np.stack([ds[:, 0, 0], ds[:, 1, 1], ds[:, 2, 2], ds[:, 1, 2], ds[:, 0, 2], ds[:, 0, 1]], axis=1)
    return ds_v, b['sweights'].reshape(-1, 1)


def _combine_stress_lists(ds, sw):
    ds = np.vstack(ds) if ds else np.zeros((0, 6))
    sw = np.vstack(sw) if sw else np.zeros((0, 1))
    if mpi.is_distributed():
        ds = mpi.allgather_vstack(ds)
        sw = mpi.allgather_vstack(sw)
    return ds, sw


def _my_struct_range(n_struct):
    """This rank's contiguous [lo, hi) slice of range(n_struct). Empty slice -> lo==hi."""
    w = mpi.world_size()
    if w == 1:
        return 0, n_struct
    parts = np.array_split(np.arange(n_struct), w)
    p = parts[mpi.rank()]
    if len(p) == 0:
        return 0, 0
    return int(p[0]), int(p[-1]) + 1


def slice_batch_structs(b, s_lo, s_hi):
    """Return a new super-batch dict containing only structures [s_lo, s_hi) of `b`,
    with every offset-index array (ind_i/ind_j/e_map/cell_map/pair_map) re-based to
    start at 0 -- i.e. a self-consistent smaller batch that `external_fit` accepts.

    Relies on how utilities.supply_batch() lays a batch out: structures are stored
    consecutively, so their atoms and their (intra-structure) neighbour pairs are
    each a contiguous run; e_map/cell_map are ascending atom->structure indices and
    pair_map is the ascending pair->structure index. Returns None for an empty slice.
    """
    if s_hi <= s_lo:
        return None
    e_map = b['e_map']
    pair_map = b['pair_map']
    a_lo = int(np.searchsorted(e_map, s_lo, side='left'))
    a_hi = int(np.searchsorted(e_map, s_hi, side='left'))
    p_lo = int(np.searchsorted(pair_map, s_lo, side='left'))
    p_hi = int(np.searchsorted(pair_map, s_hi, side='left'))
    return {
        # per-atom
        'positions': b['positions'][a_lo:a_hi],
        'forces':    b['forces'][a_lo:a_hi],
        'fweights':  b['fweights'][a_lo:a_hi],
        'cell_map':  b['cell_map'][a_lo:a_hi] - s_lo,
        'e_map':     b['e_map'][a_lo:a_hi] - s_lo,
        # per-structure
        'energy':    b['energy'][s_lo:s_hi],
        'eweights':  b['eweights'][s_lo:s_hi],
        'cell':      b['cell'][s_lo:s_hi],
        'stress':    b['stress'][s_lo:s_hi],
        'sweights':  b['sweights'][s_lo:s_hi],
        # per-pair (neighbour list); pairs are intra-structure so ind_* stay in-range
        'ind_i':     b['ind_i'][p_lo:p_hi] - a_lo,
        'ind_j':     b['ind_j'][p_lo:p_hi] - a_lo,
        'mu_i':      b['mu_i'][p_lo:p_hi],
        'mu_j':      b['mu_j'][p_lo:p_hi],
        'offsets':   b['offsets'][p_lo:p_hi],
        'pair_map':  b['pair_map'][p_lo:p_hi] - s_lo,
    }


NUMPY_OPTIMIZERS = ['BFGS', 'L-BFGS-B', 'CG', 'trf', 'dogbox', 'lm', 'dual_annealing']
TF_OPTIMIZERS = ['Adam', 'SGD']


class FitTensorPotential:
    def __init__(self, tensorpot, display_step=20, batch_size=10, eager=False):
        self.tensorpot = tensorpot
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
        self.optmzr = None
        self.eager = eager
        self.fit_metrics_data_dict = {}
        self.last_fit_metric_data = None
        self.last_test_metric_data = None
        self.fit_metric_callback = None
        self.test_metric_callback = None
        self.nfuncs = None

    def fit(self, df, test_df=None, batch_size=100, optimizer='BFGS', coefs=None, niter=100, jacobian_factor=None,
            callback=None, save_coefs=False, options=None, fit_metric_callback=None, test_metric_callback=None):
        if fit_metric_callback is not None:
            self.fit_metric_callback = fit_metric_callback
        if test_metric_callback is not None:
            self.test_metric_callback = test_metric_callback

        self.data_df = df
        self.batch_size = batch_size
        batches = batching_data(self.data_df, batch_size=batch_size)
        self.n_batches = len(batches)  # GLOBAL batch count (reg. scaling depends on this)

        # --- data-parallel work split. Two modes (mpi.shard_mode()):
        #  batchlist: each rank runs a contiguous slice of the batch LIST.
        #             Per-GPU memory unchanged (whole batches land on one GPU).
        #  inbatch:   every rank runs EVERY batch but only its contiguous slice of
        #             that batch's structures (see slice_batch_structs). Per-GPU
        #             memory ~1/W -> `batch_size` can go up ~Wx.
        #  inbatch makes W times more, smaller calls per rank, so it is only as fast
        #  as batchlist when each GPU still gets ~100 structures per call.
        # Metrics/weights below stay GLOBAL (full list) so per-iteration metrics
        # match the single-process result after the error arrays are allgathered.
        self._inbatch = False
        if mpi.is_distributed():
            if optimizer not in ('BFGS', 'L-BFGS-B', 'CG'):
                raise NotImplementedError(
                    "PACEMAKER_MPI data-parallel training supports optimizer in "
                    "{'BFGS','L-BFGS-B','CG'} only; got '%s'. Run single-GPU for "
                    "the least-squares / dual_annealing / Adam / SGD paths." % optimizer)
            if mpi.shard_mode() == 'inbatch':
                self._inbatch = True
                local_batches = batches  # every rank keeps every batch; sliced in fit_func
                # predict() runs root-only and un-sharded; the yaml batch_size is a
                # GLOBAL super-batch here (only batch_size/W ever lands on one GPU during
                # the fit), so cap predict's tiling to that per-GPU size or it OOMs.
                self._predict_batch_size = max(1, -(-self.batch_size // mpi.world_size()))
                _bs = -(-len(self.data_df) // self.n_batches)   # ceil structs per batch
                log.info("[rank %d/%d] in-batch sharding: ~%d of ~%d structs per batch",
                         mpi.rank(), mpi.world_size(),
                         -(-_bs // mpi.world_size()), _bs)
                # regularization is added once per external_fit call, scaled 1/n_batches.
                # In inbatch mode every batch is evaluated W times (once per rank), so
                # let ONLY rank 0 carry the reg term; Allreduce(sum) then counts it once.
                if not mpi.is_root():
                    for _k in (L1_REG, L2_REG, AUX_LOSS_FACTOR):
                        if _k in self.tensorpot.loss_specs:
                            self.tensorpot.loss_specs[_k] = tf.zeros_like(
                                self.tensorpot.loss_specs[_k])
            else:
                local_batches = _split_contiguous(batches)
                log.info("[rank %d/%d] %d of %d batches (contiguous shard)",
                         mpi.rank(), mpi.world_size(), len(local_batches), len(batches))
        else:
            local_batches = batches
        self._local_batches = local_batches

        w_e = np.vstack([b['eweights'] for b in batches])
        w_f = np.vstack([b['fweights'] for b in batches])
        force_scale = float(self.tensorpot.loss_specs[LOSS_FORCE_FACTOR])
        energy_scale = float(self.tensorpot.loss_specs[LOSS_ENERGY_FACTOR])
        self.metrics = FitMetrics(w_e, w_f, energy_scale, force_scale, self.ncoefs)
        self.metrics.s_scale = float(self.tensorpot.loss_specs.get(LOSS_STRESS_FACTOR, 0.))
        self.metrics.nfuncs = self.nfuncs

        if test_df is not None:
            self.test_df = test_df
            test_batches = batching_data(self.test_df, batch_size=batch_size)
            test_w_e = np.vstack([b['eweights'] for b in test_batches])
            test_w_f = np.vstack([b['fweights'] for b in test_batches])
            self.test_metrics = FitMetrics(test_w_e, test_w_f, energy_scale, force_scale, self.ncoefs)
            self.test_metrics.nfuncs = self.nfuncs

        self.global_callback = callback
        if jacobian_factor is not None:
            self.jacobian_factor = np.array(jacobian_factor)

        if coefs is None:
            coefs = self.tensorpot.potential.get_coefs().numpy()
        elif coefs is not None:
            try:
                assert len(coefs) == self.ncoefs
            except:
                raise ValueError('Number of provided coefficients {0} is different from the numbers used in model {1}' \
                                 .format(len(coefs), self.ncoefs))

        t0 = time.time()
        if optimizer in NUMPY_OPTIMIZERS:
            # TODO: Make args passable
            if optimizer in ['trf', 'dogbox', 'lm']:
                from scipy.optimize import least_squares

                # if self.tensorpot.potential.__class__.__name__ == 'ACE':
                #     assert self.tensorpot.potential.rankmax == 1, 'Only pair potential of Atomic Cluster Expansion is' \
                #                                                   ' supported for the least squares algorithms'

                ls_opt_func = LMfunc(self, w_e)

                res_opt = least_squares(ls_opt_func.func, coefs, jac=ls_opt_func.jac,
                                        method=optimizer, args=(batches,), ftol=1e-8, verbose=1, max_nfev=niter)
            elif optimizer == 'dual_annealing':

                da_opt_func = DAfunc(self)

                bounds = [(-10000, 10000) for _ in coefs]
                options = {'disp': True, 'maxiter': niter, 'jac': False}
                minimizer_kwargs = {'method': 'BFGS',
                                    # 'args':(batches,),
                                    'jac': da_opt_func.jac,
                                    'options': {'gtol': 1e-8, 'disp': True, 'maxiter': niter}}
                res_opt = dual_annealing(da_opt_func.func, bounds, args=(batches,), maxiter=niter,
                                         local_search_options=minimizer_kwargs, callback=self.callback, x0=coefs)
                #     minimizer_kwargs = minimizer_kwargs, options=options, )

            else:
                if options is None:
                    options = {'gtol': 1e-8, 'disp': True, 'maxiter': niter}
                options["maxiter"] = niter
                log.info("Minimizer options: {}".format(options))
                # reset fit_metrics_data_dict here and every time in self.callback. No need to store complete history
                self.fit_metrics_data_dict = {}
                res_opt = minimize(self.fit_func, coefs, method=optimizer, jac=True,
                                   args=(local_batches,), options=options,
                                   callback=self.callback)
                self.process_test_metric()

            self.res_opt = res_opt
            self.fit_coefs = res_opt.x
            self.tensorpot.potential.set_coefs(self.fit_coefs)

        elif optimizer in TF_OPTIMIZERS:
            self.opt = self.get_optimzer(optimizer)
            for epoch in range(niter):
                loss = self.tf_fit_func(batches)

                self.fit_coefs = self.tensorpot.potential.get_coefs().numpy()
                self.callback(self.fit_coefs)
        else:
            raise ValueError("Unknown optimizer `{}`. Should be one of {}".format(optimizer, NUMPY_OPTIMIZERS+TF_OPTIMIZERS))

        if save_coefs:
            self.save_coefs()
        # x = res_opt.x
        x = self.fit_coefs
        self.update_last_metrics(x)
        print('Fitting took {0:6.2f} seconds'.format(time.time() - t0))

    def restart_fit(self, df, batch_size=100, optimizer='BFGS', niter=100, jacobian_factor=None):
        if self.eval_count == 0:
            raise AttributeError('Cannot restart, no fit has been done so far')
        else:
            self.fit(df, batch_size=batch_size, optimizer=optimizer, coefs=self.fit_coefs, niter=niter,
                     jacobian_factor=jacobian_factor)

    def get_optimzer(self, opt_name):
        if opt_name == 'Adam':
            optmzr = tf.keras.optimizers.Adam(learning_rate=1e-3)
        elif opt_name == 'SGD':
            optmzr = tf.keras.optimizers.SGD(learning_rate=1e-3)

        return optmzr

    def tf_fit_func(self, batches):
        tot_loss = 0
        de, de_pa, df = ([] for _ in range(3))
        total_na = []
        t0 = time.time()
        idx = np.arange(len(batches))
        np.random.shuffle(idx)
        # for b in batches:
        for ind in idx:
            b = batches[ind]
            loss, grad_loss, e, f = self.tensorpot.native_fit(b, eager=self.eager)
            self.opt.apply_gradients(zip([grad_loss], [self.tensorpot.potential.fit_coefs]))
            tot_loss += loss.numpy()
            errors = self.compute_batch_errors(b, e.numpy(), f.numpy())
            de.append(errors[0])
            de_pa.append(errors[1])
            df.append(errors[2])
            total_na.append(errors[3])

        self.fit_coefs = self.tensorpot.potential.get_coefs().numpy()
        self.eval_count += 1
        self.eval_time = time.time() - t0
        # collect all relevant info into FitMetrics
        self.metrics.regs = self.get_reg_components()
        self.metrics.reg_weights = self.get_reg_weights()
        self.metrics.compute_metrics(np.vstack(de), np.vstack(de_pa), np.vstack(df),
                                     np.vstack(total_na), dataframe=self.data_df)
        self.metrics.loss = tot_loss
        self.metrics.eval_time = self.eval_time
        # do a snapshot of  FitMetrics into FitMetricsData: loss, e_loss, f_loss, reg_loss, RMSE, MAE, MAX_E (E,F), timing
        curr_fit_metrics_data = self.metrics.to_FitMetricsDict()
        curr_fit_metrics_data["eval_count"] = self.eval_count
        curr_fit_metrics_data["iter_num"] = self.iter_num
        # store metrics_data into dict (x-> curr_fit_metrics_data)
        self.fit_metrics_data_dict[hash(self.fit_coefs.tobytes())] = curr_fit_metrics_data
        self.last_fit_metric_data = curr_fit_metrics_data

        if self.iter_num == 0:
            if self.fit_metric_callback is not None:
                self.fit_metric_callback(curr_fit_metrics_data)
            else:
                print_detailed_metrics(curr_fit_metrics_data, title='Initial state:')
                print_extended_metrics(curr_fit_metrics_data, title='INIT STATS')
            self.iter_num += 1

        self.loss_history.append(tot_loss)
        self.last_loss = self.loss_history[-1]

        return tot_loss

    def fit_func(self, x, batches):
        tot_loss = 0
        tot_jac = 0
        de, de_pa, df = ([] for _ in range(3))
        total_na = []
        ds_l, sw_l = [], []
        fit_stress = self.tensorpot.fit_stress
        t0 = time.time()
        if not mpi.is_distributed():
            # single process: behave exactly like stock (let any error propagate
            # untouched so FitBackendAdapter's retry loop sees the real exception).
            for b in batches:
                loss, jac, e, f = self.tensorpot.external_fit(x, b, eager=self.eager)
                tot_jac += jac.numpy()
                tot_loss += loss.numpy()
                errors = self.compute_batch_errors(b, e.numpy(), f.numpy())
                if fit_stress:
                    _ds, _sw = stress_errors(b, self.tensorpot.last_stress.numpy())
                    ds_l.append(_ds)
                    sw_l.append(_sw)
                de.append(errors[0])
                de_pa.append(errors[1])
                df.append(errors[2])
                total_na.append(errors[3])
        else:
            # distributed: if ANY rank hits an OOM, every rank must raise together
            # so the batch-size-reduction retry in FitBackendAdapter.fit stays
            # collective (a lone rank raising would deadlock the others in Allreduce).
            inbatch = getattr(self, '_inbatch', False)
            oom = 0
            try:
                for b in batches:
                    if inbatch:
                        s_lo, s_hi = _my_struct_range(len(b['energy']))
                        sb = slice_batch_structs(b, s_lo, s_hi)
                        if sb is None:
                            continue          # no structures of this batch on this rank
                    else:
                        sb = b
                    loss, jac, e, f = self.tensorpot.external_fit(x, sb, eager=self.eager)
                    tot_jac = tot_jac + jac.numpy()
                    tot_loss += float(loss.numpy())
                    errors = self.compute_batch_errors(sb, e.numpy(), f.numpy())
                    if fit_stress:
                        _ds, _sw = stress_errors(sb, self.tensorpot.last_stress.numpy())
                        ds_l.append(_ds)
                        sw_l.append(_sw)
                    de.append(errors[0])
                    de_pa.append(errors[1])
                    df.append(errors[2])
                    total_na.append(errors[3])
            except tf.errors.ResourceExhaustedError:
                oom = 1
            oom = mpi.allreduce_max(oom)
            if oom:
                raise tf.errors.ResourceExhaustedError(
                    None, None, "OOM during batch evaluation (possibly on another rank)")
            if np.ndim(tot_jac) == 0:
                # this rank evaluated nothing (all its in-batch slices were empty)
                tot_jac = np.zeros(self.ncoefs, dtype=np.float64)

        # --- data-parallel reduction: global full-dataset loss + jacobian ---
        tot_loss = mpi.allreduce_scalar_sum(tot_loss)
        if mpi.is_distributed():
            tot_jac = mpi.allreduce_array_sum(tot_jac)

        self.fit_coefs = x
        self.eval_count += 1
        self.eval_time = time.time() - t0
        # gather the per-structure error arrays so every rank computes the SAME
        # dataset-global metrics (identical to the single-process result).
        g_de, g_de_pa, g_df, g_na = _combine_error_lists(de, de_pa, df, total_na)
        # collect all relevant info into FitMetrics
        self.metrics.regs = self.get_reg_components()
        self.metrics.reg_weights = self.get_reg_weights()
        # inbatch mode: allgather interleaves per-batch structure slices, so the
        # concatenation order != dataframe order -> the low-energy mask (indexed by
        # df position) would misalign. Global RMSE/MAE/MAX are order-independent and
        # stay exact; skip only the "low" sub-metrics by passing dataframe=None.
        _df_for_metrics = None if getattr(self, '_inbatch', False) else self.data_df
        self.metrics.compute_metrics(g_de, g_de_pa, g_df, g_na, dataframe=_df_for_metrics)
        if fit_stress:
            self.metrics.compute_stress_metrics(*_combine_stress_lists(ds_l, sw_l))
        self.metrics.loss = tot_loss
        self.metrics.eval_time = self.eval_time
        # do a snapshot of  FitMetrics into FitMetricsData: loss, e_loss, f_loss, reg_loss, RMSE, MAE, MAX_E (E,F), timing
        curr_fit_metrics_data = self.metrics.to_FitMetricsDict()
        curr_fit_metrics_data["eval_count"] = self.eval_count
        curr_fit_metrics_data["iter_num"] = self.iter_num
        # store metrics_data into dict (x-> curr_fit_metrics_data)
        self.fit_metrics_data_dict[hash(x.tobytes())] = curr_fit_metrics_data
        self.last_fit_metric_data = curr_fit_metrics_data

        if self.iter_num == 0:
            if self.fit_metric_callback is not None:
                self.fit_metric_callback(curr_fit_metrics_data)
            elif mpi.is_root():
                print_detailed_metrics(curr_fit_metrics_data, title='Initial state:')
                print_extended_metrics(curr_fit_metrics_data, title='INIT STATS')

            self.process_test_metric(title="TEST INIT STATS")

            self.iter_num += 1

        self.loss_history.append(tot_loss)
        self.last_loss = self.loss_history[-1]

        if self.jacobian_factor is not None:
            try:
                assert len(self.jacobian_factor) == tot_jac.shape[0]
                tot_jac *= self.jacobian_factor
            except:
                raise ValueError(
                    'Size of the provided Jacobian factor ({0}) is not compatible with Jacobian size ({1}).' \
                        .format(len(self.jacobian_factor), tot_jac.shape[0]))

        return tot_loss, tot_jac

    def compute_batch_errors(self, b, e, f):
        e_true = b['energy']
        f_true = b['forces']
        _, nat = np.unique(b['e_map'], return_counts=True)
        de = e_true - e
        de_pa = de / nat.reshape(-1, 1)
        df = f_true - f

        return de, de_pa, df, nat.reshape(-1, 1)

    def callback(self, x, *args, **kwargs):
        # call global callback
        self.metrics.record_time(self.eval_time)

        # --- data-parallel safety net: every rank runs its own (replicated) scipy
        # optimizer; they stay in lock-step only because fit_func returns identical
        # (loss, jac) to all ranks. Assert they have not drifted.
        if mpi.is_distributed():
            x_ref = mpi.bcast(np.ascontiguousarray(x) if mpi.is_root() else None)
            dev = mpi.allreduce_max(float(np.max(np.abs(x - x_ref))))
            tol = 1e-9 * (1.0 + float(np.max(np.abs(x_ref))))
            if dev > tol:
                raise RuntimeError(
                    "[rank %d] replicated optimizer diverged: max|x-x_root|=%.3e > %.3e"
                    % (mpi.rank(), dev, tol))

        # get true metrics data that correspond to actual successfull x
        true_fit_metric_data = self.update_last_metrics(x)

        if self.fit_metric_callback is not None:
            # example of fit_metric_callback is:  print_extended_metrics
            self.fit_metric_callback(true_fit_metric_data)
        elif mpi.is_root():
            if self.iter_num % self.display_step == 0:
                print_extended_metrics(true_fit_metric_data)
            else:
                print_detailed_metrics(true_fit_metric_data)
        # test metric: runs on every rank (each evaluates the full small test
        # set locally) so test-based early stopping is a collective decision; the
        # yaml snapshot inside is still root-only.
        if self.iter_num % self.display_step == 0:
            self.process_test_metric(updating_coefs=x)
        self.iter_num += 1

        if mpi.is_root() and self.iter_num % self.display_step == 0:
            self.tensorpot.potential.get_updated_config(prefix=self.iter_num)

        # clean fit_metrics_data_dict for next iteration
        # self.fit_metrics_data_dict = {}

        # global_callback chains into GeneralACEFit's interim-potential / plot
        # writers -> run on root only.
        if self.global_callback is not None and mpi.is_root():
            self.global_callback(x, *args, **kwargs)

    def process_test_metric(self, title="TEST_STATS", updating_coefs=None):
        if self.test_df is not None:
            curr_test_metrics_data = self.compute_test_metric(updating_coefs=updating_coefs)
            curr_test_metrics_data["eval_count"] = self.eval_count
            curr_test_metrics_data["iter_num"] = self.iter_num
            self.last_test_metric_data = curr_test_metrics_data
            if self.test_metric_callback is not None:
                # runs on every rank -> detect_early_stopping('test') decides identically
                self.test_metric_callback(curr_test_metrics_data)
            elif mpi.is_root():
                print_extended_metrics(curr_test_metrics_data, title=title)

    def update_last_metrics(self, x):
        x_hash = hash(x.tobytes())
        true_fit_metric_data = self.fit_metrics_data_dict[x_hash]
        true_fit_metric_data["iter_num"] = self.iter_num
        true_fit_metric_data["eval_count"] = len(self.loss_history)
        # update self.metrics with this true metrics data
        self.metrics.from_FitMetricsDict(true_fit_metric_data)
        self.last_fit_metric_data = true_fit_metric_data
        return true_fit_metric_data

    def predict(self, datadf=None):
        _pbs = getattr(self, '_predict_batch_size', self.batch_size)
        if datadf is None:
            batches = batching_data(self.data_df, batch_size=_pbs)
        else:
            batches = batching_data(datadf, batch_size=_pbs)

        pred_dict = {'energy_pred': [],
                     'forces_pred': [],
                     'stress_pred': [],
                     'de': [],
                     'de_pa': [],
                     'df': [],
                     'total_na': []}
        for b in batches:
            _, partition = np.unique(b['e_map'], return_counts=True)
            for i in range(1, len(partition)):
                partition[i] += partition[i - 1]
            e, f, s = self.tensorpot.evaluate(b)
            errors = self.compute_batch_errors(b, e.numpy(), f.numpy())
            e_pred = e.numpy().reshape(-1, )
            f_pred = np.split(f.numpy().reshape(-1, 3), partition)[:-1]
            s_pred = s.numpy().reshape(-1, 3, 3)
            pred_dict['energy_pred'] += list(e_pred)
            pred_dict['forces_pred'] += list(f_pred)
            pred_dict['stress_pred'] += list(s_pred)
            pred_dict['de'] += list(errors[0].reshape(-1, ))
            pred_dict['de_pa'] += list(errors[1].reshape(-1, ))
            pred_dict['df'] += list(np.split(errors[2].reshape(-1, 3), partition)[:-1])
            pred_dict['total_na'].append(errors[3])

        # return pd.DataFrame(pred_dict)
        return pred_dict

    def get_fitting_data(self):
        return self.data_df

    def get_reg_components(self):
        regs = self.tensorpot.reg_components.numpy().reshape(-1, 1).astype(np.float64)
        # return [float(regs[i]) for i in range(regs.shape[0])]
        return regs

    def get_reg_weights(self):
        loss_specs = self.tensorpot.loss_specs
        l1 = np.array(loss_specs[L1_REG]).reshape(-1, 1)
        l2 = np.array(loss_specs[L2_REG]).reshape(-1, 1)
        aux = np.array(loss_specs[AUX_LOSS_FACTOR]).reshape(-1, 1)

        return np.vstack([l1, l2, aux]) * self.n_batches

    def save_coefs(self):
        np.savetxt('Fit_coefficients_step_{0}'.format(self.eval_count), self.fit_coefs)

    def get_fitted_coefficients(self):
        if self.eval_count > 0:
            return self.fit_coefs
        else:
            raise AttributeError('Coefficients have not been fitted yet')

    def print_detailed_metrics(self, title='Iteration:'):
        print_detailed_metrics(self.last_fit_metric_data, title=title)

    def print_extended_metrics(self, title="FIT_STATS"):
        print_extended_metrics(self.last_fit_metric_data, title=title)

    def compute_test_metric(self, updating_coefs=None):
        if self.test_df is not None:
            # every rank must assign `updating_coefs` so its local predict() is correct;
            # only root also writes the interim_potential_<N>.yaml snapshot.
            if mpi.is_root():
                self.tensorpot.potential.get_updated_config(updating_coefs=updating_coefs, prefix=self.iter_num)
            else:
                self.tensorpot.potential.get_updated_config(updating_coefs=updating_coefs)
            # 'energy_pred', 'forces_pred', 'stress_pred',
            # 'de', 'de_pa', 'df', 'total_na'
            test_pred = self.predict(datadf=self.test_df)
            self.test_metrics.regs = self.get_reg_components()
            self.test_metrics.reg_weights = self.get_reg_weights()
            self.test_metrics.compute_metrics(np.vstack(test_pred['de']),
                                              np.vstack(test_pred['de_pa']),
                                              np.vstack(test_pred['df']),
                                              np.vstack(test_pred['total_na']),
                                              dataframe=self.test_df)
            # self.test_metrics.loss = tot_loss
            # self.test_metrics.eval_time = self.eval_time
            curr_test_metrics_data = self.test_metrics.to_FitMetricsDict()
            curr_test_metrics_data["loss"] = curr_test_metrics_data["e_loss_contrib"] + \
                                             curr_test_metrics_data["f_loss_contrib"] + \
                                             curr_test_metrics_data["l1_reg_contrib"] + \
                                             curr_test_metrics_data["l2_reg_contrib"] + \
                                             sum(curr_test_metrics_data["extra_regularization_contrib"])
            curr_test_metrics_data["iter_num"] = self.iter_num
            return curr_test_metrics_data


class LMfunc:
    def __init__(self, tensorpot_fit, w_e):
        self.tensorpot_fit = tensorpot_fit
        self.resid_value = None
        self.jacob_value = None
        self.x_current = None
        self.w_e = w_e

    def run_fit(self, x, batches):
        tot_res = []
        tot_jac = []
        de, de_pa, df = ([] for _ in range(3))
        total_na = []
        t0 = time.time()
        for index, b in enumerate(batches):
            loss, jac, e, f = self.tensorpot_fit.tensorpot.external_vector_fit(x, b)
            nregs = len(self.tensorpot_fit.get_reg_components())
            if (index + 1) < len(batches):
                tot_jac.append(jac.numpy().reshape(-1, len(x))[:-nregs])
                tot_res.append(loss.numpy().reshape(-1, 1)[:-nregs])
            else:
                tot_jac.append(jac.numpy().reshape(-1, len(x)))
                tot_res.append(loss.numpy().reshape(-1, 1))

            errors = self.tensorpot_fit.compute_batch_errors(b, e.numpy(), f.numpy())
            de.append(errors[0])
            de_pa.append(errors[1])
            df.append(errors[2])
            total_na.append(errors[3])
        tot_res = np.vstack(tot_res).reshape(-1, )
        tot_jac = np.vstack(tot_jac)
        tot_res[-nregs:] *= len(batches)

        scalar_loss = np.sum(self.w_e.reshape(-1, 1) * (tot_res[:-nregs].reshape(-1, 1) / self.w_e.reshape(-1, 1)) ** 2) \
                      + np.sum(tot_res[-nregs:])
        self.tensorpot_fit.fit_coefs = x
        self.tensorpot_fit.eval_count += 1
        self.tensorpot_fit.eval_time = time.time() - t0
        self.tensorpot_fit.metrics.regs = self.tensorpot_fit.get_reg_components()
        self.tensorpot_fit.metrics.compute_metrics(np.vstack(de), np.vstack(de_pa), np.vstack(df),
                                                   np.vstack(total_na), dataframe=self.tensorpot_fit.data_df)

        if self.tensorpot_fit.iter_num == 0:
            log.info('{:<32}'.format('Initial state:') + '{:>10}'.format('Loss: ') + "{loss: >3.6f}" \
                     .format(loss=scalar_loss) +
                     '{str:>21}{rmse_epa:>.2f} ({low_rmse_e:>.2f}) meV/at' \
                     .format(str=" | RMSE Energy(low): ",
                             rmse_epa=1e3 * float(self.tensorpot_fit.metrics.rmse_epa),
                             low_rmse_e=1e3 * self.tensorpot_fit.metrics.low_rmse_epa))
            self.tensorpot_fit.metrics.print_extended_metrics(self.tensorpot_fit.iter_num, scalar_loss,
                                                              self.tensorpot_fit.get_reg_components(),
                                                              self.tensorpot_fit.get_reg_weights())
            self.tensorpot_fit.iter_num += 1

        self.tensorpot_fit.loss_history.append(scalar_loss)

        return tot_res, tot_jac

    def update(self, x, batches):
        # self.resid_value, self.jacob_value = self.run_fit(x, batches)
        resid_value, jacob_value = self.run_fit(x, batches)
        self.resid_value = resid_value.copy()
        self.jacob_value = jacob_value.copy()
        self.x_current = x.copy()

    def func(self, x, batches):
        self.update(x, batches)
        # self.x_current = x
        self.tensorpot_fit.callback(x)
        return self.resid_value

    def jac(self, x, batches):
        if (x != self.x_current).any():
            # print(x != self.x_current, 'JAC CALLED FOR UPDATE')
            self.update(x, batches)
        return self.jacob_value


class DAfunc:
    def __init__(self, tensorpot_fit):
        self.tensorpot_fit = tensorpot_fit
        self.loss_value = None
        self.jacob_value = None
        self.x_current = None
        self.batches = None

    def run_fit(self, x, batches):
        tot_loss = 0
        tot_jac = 0
        de, de_pa, df = ([] for _ in range(3))
        total_na = []
        t0 = time.time()
        for b in batches:
            loss, jac, e, f = self.tensorpot_fit.tensorpot.external_fit(x, b, eager=self.tensorpot_fit.eager)
            tot_jac += jac.numpy()
            tot_loss += loss.numpy()
            errors = self.tensorpot_fit.compute_batch_errors(b, e.numpy(), f.numpy())
            de.append(errors[0])
            de_pa.append(errors[1])
            df.append(errors[2])
            total_na.append(errors[3])

        self.tensorpot_fit.fit_coefs = x
        self.tensorpot_fit.eval_count += 1
        self.tensorpot_fit.eval_time = time.time() - t0
        self.tensorpot_fit.metrics.regs = self.tensorpot_fit.get_reg_components()
        self.tensorpot_fit.metrics.compute_metrics(np.vstack(de), np.vstack(de_pa), np.vstack(df),
                                                   np.vstack(total_na), dataframe=self.tensorpot_fit.data_df)
        if self.tensorpot_fit.iter_num == 0:
            log.info('{:<32}'.format('Initial state:') + '{:>10}'.format('Loss: ') + "{loss: >3.6f}" \
                     .format(loss=tot_loss) +
                     '{str:>21}{rmse_epa:>.2f} ({low_rmse_e:>.2f}) meV/at' \
                     .format(str=" | RMSE Energy(low): ",
                             rmse_epa=1e3 * float(self.tensorpot_fit.metrics.rmse_epa),
                             low_rmse_e=1e3 * self.tensorpot_fit.metrics.low_rmse_epa))
            self.tensorpot_fit.metrics.print_extended_metrics(self.tensorpot_fit.iter_num, tot_loss,
                                                              self.tensorpot_fit.get_reg_components(),
                                                              self.tensorpot_fit.get_reg_weights())
            self.tensorpot_fit.iter_num += 1

        self.tensorpot_fit.loss_history.append(tot_loss)
        self.last_loss = self.tensorpot_fit.loss_history[-1]

        if self.tensorpot_fit.jacobian_factor is not None:
            try:
                assert len(self.tensorpot_fit.jacobian_factor) == tot_jac.shape[0]
                tot_jac *= self.tensorpot_fit.tensorpot_fit.jacobian_factor
            except:
                raise ValueError(
                    'Size of the provided Jacobian factor ({0}) is not compatible with Jacobian size ({1}).' \
                        .format(len(self.tensorpot_fit.jacobian_factor), tot_jac.shape[0]))

        return tot_loss, tot_jac

    def update(self, x, batches):
        # self.resid_value, self.jacob_value = self.run_fit(x, batches)
        loss_value, jacob_value = self.run_fit(x, batches)
        self.loss_value = loss_value.copy()
        self.jacob_value = jacob_value.copy()
        self.x_current = x.copy()

    def func(self, x, batches):
        self.batches = batches
        self.update(x, batches)
        # self.x_current = x
        self.tensorpot_fit.callback(x)
        return self.loss_value

    # def jac(self, x, batches):
    def jac(self, x):
        if (x != self.x_current).any():
            # print(x != self.x_current, 'JAC CALLED FOR UPDATE')
            self.update(x, self.batches)
        return self.jacob_value


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
             ' | Time/eval: {:>6.2f} mcs/at'.format(avg_t * 1e6))
