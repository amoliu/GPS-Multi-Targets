""" This file defines the GMM prior for dynamics estimation. """
import copy
import logging

import numpy as np

from gps.algorithm.dynamics.config import DYN_PRIOR_GMM
from sklearn.mixture import GaussianMixture as GMM
LOGGER = logging.getLogger(__name__)


class DynamicsPriorGMM(object):
    """
    A dynamics prior encoded as a GMM over [x_t, u_t, x_t+1] points.
    See:
        S. Levine*, C. Finn*, T. Darrell, P. Abbeel, "End-to-end
        training of Deep Visuomotor Policies", arXiv:1504.00702,
        Appendix A.3.
    """
    def __init__(self, hyperparams):
        """
        Hyperparameters:
            min_samples_per_cluster: Minimum samples per cluster.
            max_clusters: Maximum number of clusters to fit.
            max_samples: Maximum number of trajectories to use for
                fitting the GMM at any given time.
            strength: Adjusts the strength of the prior.
        """
        config = copy.deepcopy(DYN_PRIOR_GMM)
        config.update(hyperparams)
        self._hyperparams = config
        self.X = None
        self.U = None
        self.N = None
        self.gmm = None
        self._min_samp = self._hyperparams['min_samples_per_cluster']
        self._max_samples = self._hyperparams['max_samples']
        self._max_clusters = self._hyperparams['max_clusters']
        self._strength = self._hyperparams['strength']

    def initial_state(self):
        """ Return dynamics prior for initial time step. """
        # Compute mean and covariance.
        mu0 = np.mean(self.X[:, 0, :], axis=0)
        Phi = np.diag(np.var(self.X[:, 0, :], axis=0))

        # Factor in multiplier.
        n0 = self.X.shape[2] * self._strength
        m = self.X.shape[2] * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi = Phi * m
        return mu0, Phi, m, n0

    def update(self, X, U):
        """
        Update prior with additional data.
        Args:
            X: A N x T x dX matrix of sequential state data.
            U: A N x T x dU matrix of sequential control data.
        """
        # Constants.
        T = X.shape[1] - 1

        # Append data to dataset.
        if self.X is None:
            self.X = X
        else:
            self.X = np.concatenate([self.X, X], axis=0)

        if self.U is None:
            self.U = U
        else:
            self.U = np.concatenate([self.U, U], axis=0)

        # Remove excess samples from dataset.
        start = max(0, self.X.shape[0] - self._max_samples + 1)
        self.X = self.X[start:, :]
        self.U = self.U[start:, :]

        # Compute cluster dimensionality.
        Do = X.shape[2] + U.shape[2] + X.shape[2]  #TODO: Use Xtgt.

        # Create dataset.
        N = self.X.shape[0]
        xux = np.reshape(
            np.c_[self.X[:, :T, :], self.U[:, :T, :], self.X[:, 1:(T+1), :]],
            [T * N, Do]
        )

        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samp))))
        LOGGER.debug('Generating %d clusters for dynamics GMM.', K)

        # Update GMM.
        self.N = xux.shape[0]
        self.gmm = GMM(n_components=K,
                       covariance_type='full',
                       reg_covar=1e-6,
                       max_iter=100,
                       n_init=3,
                       random_state=0,
                       warm_start=True,
                       verbose=2,
                       verbose_interval=1)
        self.gmm.fit(xux)


    def eval(self, Dx, Du, pts):
        """
        Evaluate prior.
        Args:
            pts: A N x Dx+Du+Dx matrix.
        """
        # Construct query data point by rearranging entries and adding
        # in reference.
        assert pts.shape[1] == Dx + Du + Dx

        # Perform query and fix mean.
        mu0, Phi, m, n0 = self.inference(pts)

        # Factor in multiplier.
        n0 = n0 * self._strength
        m = m * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0

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