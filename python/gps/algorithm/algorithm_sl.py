""" This file defines the MD-based GPS algorithm. """
import copy
import logging

import numpy as np

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.config import ALG_SL

LOGGER = logging.getLogger(__name__)


class AlgorithmSL(Algorithm):
    """
    Sample-based joint policy learning with supervised learning.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_SL)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        self.policy_opt = self._hyperparams['policy_opt']['type'](
            self._hyperparams['policy_opt'], self.dO, self.dU
        )

    def iteration(self, sample_lists):
        """
        Run iteration of supervised learning.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        self._update_policy(sample_lists)

        # # Prepare for next iteration
        # self._advance_iteration_variables()

    def _update_policy(self, sample_lists):
        """ Compute the new policy. """
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        for m in range(self.M):
            samples = sample_lists[m]
            U = samples.get_U()
            tgt_mu = np.concatenate((tgt_mu, U))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
        self.policy_opt.sl_update(obs_data, tgt_mu)


    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur'
        variables, and advance iteration counter.
        """
        Algorithm._advance_iteration_variables(self)
        for m in range(self.M):
            self.cur[m].traj_info.last_kl_step = \
                    self.prev[m].traj_info.last_kl_step
            self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)