""" This file defines a GMM prior for policy linearization. """
import copy
import logging
import scipy as sp
import numpy as np
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps.algorithm.policy.config import CONTROLLER_PRIOR_GMM
from sklearn.mixture import GaussianMixture as GMM
from gps.algorithm.algorithm_utils import gauss_fit_joint_prior


LOGGER = logging.getLogger(__name__)


class ControllerPriorGMM(object):
    """
    A controller prior encoded as a GMM over [x_t, u_t] points, where u_t is
    the action for the given state x_t. This prior is used
    when computing the linearization of the controller from demonstrated trajectories
    """
    def __init__(self, hyperparams):
        """
        Hyperparameters:
            min_samples_per_cluster: Minimum number of samples.
            max_clusters: Maximum number of clusters to fit.
            max_samples: Maximum number of trajectories to use for
                fitting the GMM at any given time.
            strength: Adjusts the strength of the prior.
        """
        config = copy.deepcopy(CONTROLLER_PRIOR_GMM)
        config.update(hyperparams)
        self._hyperparams = config
        self.X = None
        self.obs = None
        self.gmm = None
        # TODO: handle these params better (e.g. should depend on N?)
        self._min_samp = self._hyperparams['min_samples_per_cluster']
        self._max_samples = self._hyperparams['max_samples']
        self._max_clusters = self._hyperparams['max_clusters']
        self._strength = self._hyperparams['strength']

    def update(self, X, U, mode='add'):
        """
        Update GMM using demo samples.
        By default does not replace old samples.
        """

        if self.X is None or mode == 'replace':
            self.X = X
            self.U = U
        elif mode == 'add' and X.size > 0:
            self.X = np.concatenate([self.X, X], axis=0)
            self.U = np.concatenate([self.U, U], axis=0)
            # Trim extra samples
            # TODO: how should this interact with replace_samples?
            N = self.X.shape[0]
            if N > self._max_samples:
                start = N - self._max_samples
                self.X = self.X[start:, :, :]
                self.U = self.U[start:, :, :]

        # Create the dataset
        N, T = self.X.shape[:2]
        dO = self.X.shape[2] + self.U.shape[2]
        XU = np.reshape(np.concatenate([self.X, self.U], axis=2), [T * N, dO])
        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samp))))

        LOGGER.debug('Generating %d clusters for policy prior GMM.', K)
        self.N = XU.shape[0]
        self.gmm = GMM(n_components=K,
                       covariance_type='full',
                       reg_covar=1e-6,
                       max_iter=100,
                       n_init=3,
                       random_state=0,
                       warm_start=True,
                       verbose=0)
        self.gmm.fit(XU)

    def eval(self, Ts, Ps):
        """ Evaluate prior. """
        # Construct query data point.
        pts = np.concatenate((Ts, Ps), axis=1)
        # Perform query.
        mu0, Phi, m, n0 = self.inference(pts)
        # Factor in multiplier.
        n0 *= self._strength
        m *= self._strength
        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0

    def fit(self, X, U):
        """
        Fit policy linearization.

        Args:
            X: Samples (N, T, dX)
            U: demo controller means (N, T, dU)
        """
        N, T, dX = X.shape
        dU = U.shape[2]
        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        # Allocate.
        pol_K = np.zeros([T, dU, dX])
        pol_k = np.zeros([T, dU])
        pol_S = np.zeros([T, dU, dU])
        cholPSig = np.zeros((T, dU, dU))  # Cholesky decomposition.
        invPSig = np.zeros((T, dU, dU))  # Inverse of covariance.

        # Fit policy linearization with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T):
            Ts = X[:, t, :]
            Ps = U[:, t, :]
            Ys = np.concatenate([Ts, Ps], axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.eval(Ts, Ps)
            sig_reg = np.zeros((dX+dU, dX+dU))
            # Slightly regularize on first timestep.
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8
            pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = \
                    gauss_fit_joint_prior(Ys,
                            mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
            cholPSig[t, :, :] = sp.linalg.cholesky(pol_S[t, :, :])
            U_chol = cholPSig[t, :, :]
            invPSig[t, :, :] = sp.linalg.solve_triangular(U_chol, sp.linalg.solve_triangular(U_chol.T, np.eye(dU), lower=True))

        return pol_K, pol_k, pol_S, cholPSig, invPSig


    def inference(self, pts):
        wts = self.gmm.predict_proba(pts)
        wts = wts.mean(axis=0).T
        wts = np.expand_dims(wts, axis=1)
        # Compute overall mean.
        mu = np.sum(self.gmm.means_ * wts, axis=0)

        # Compute overall covariance.
        # For some reason this version works way better than the "right"
        # one... could we be computing xxt wrong?#
        # Refer to https://github.com/cbfinn/gps/issues/72
        diff = self.gmm.means_ - np.expand_dims(mu, axis=0)
        diff_expand = np.expand_dims(self.gmm.means_, axis=1) * \
                      np.expand_dims(diff, axis=2)
        # diff_expand = np.expand_dims(diff, axis=1) * \
        #               np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(wts, axis=2)
        sigma = np.sum((self.gmm.covariances_ + diff_expand) * wts_expand, axis=0)

        # Set hyperparameters.
        m = self.N
        n0 = m - 2 - mu.shape[0]

        # Normalize.
        m = float(m) / self.N
        n0 = float(n0) / self.N
        return mu, sigma, m, n0
