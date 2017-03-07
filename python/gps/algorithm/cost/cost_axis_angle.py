""" This file defines the torque (action) cost. """
import copy

import numpy as np
from gps.algorithm.cost.config import COST_AXIS_ANGLE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier
from gps.proto.gps_pb2 import END_EFFECTOR_ROTATIONS

class CostAxisAngle(Cost):
    """ Computes torque penalties. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_AXIS_ANGLE)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX
        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        sample_quaternion = sample.get(END_EFFECTOR_ROTATIONS)
        tgt = self._hyperparams['target_quaternion']
        _, dim_sensor = sample_quaternion.shape

        wpm = get_ramp_multiplier(
            self._hyperparams['ramp_option'], T,
            wp_final_multiplier=self._hyperparams['wp_final_multiplier']
        )
        wp = self._hyperparams['wp'] * np.expand_dims(wpm, axis=-1)
        # Compute state penalty.
        dist = sample_quaternion - tgt
        dist[0] = (dist[0] + np.pi) % (2 * np.pi) - np.pi

        # Evaluate penalty term.
        l, ls, lss = evall1l2term(
            wp, dist, np.tile(np.eye(dim_sensor), [T, 1, 1]),
            np.zeros((T, dim_sensor, dim_sensor, dim_sensor)),
            self._hyperparams['l1'], self._hyperparams['l2'],
            self._hyperparams['alpha']
        )

        final_l += l

        sample.agent.pack_data_x(final_lx, ls, data_types=[END_EFFECTOR_ROTATIONS])
        sample.agent.pack_data_x(final_lxx, lss,
                                 data_types=[END_EFFECTOR_ROTATIONS, END_EFFECTOR_ROTATIONS])

        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux