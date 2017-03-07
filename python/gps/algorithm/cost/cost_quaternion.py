""" This file defines the torque (action) cost. """
import copy
import tensorflow as tf
import numpy as np
from gps.algorithm.cost.cost import Cost
from gps.proto.gps_pb2 import END_EFFECTOR_ROTATIONS

class CostQuaternion(Cost):
    """ Computes torque penalties. """
    def __init__(self, hyperparams):
        config = {}
        config.update(hyperparams)
        Cost.__init__(self, config)


    def cosine_distance(self, label, predictions):
        lp_dot_product = tf.reduce_sum(tf.multiply(predictions, label), axis=1)
        ll_norm = tf.sqrt(tf.reduce_sum(tf.multiply(label, label), axis=1))
        pp_norm = tf.sqrt(tf.reduce_sum(tf.multiply(predictions, predictions), axis=1))
        cos_dist = tf.div(tf.div(lp_dot_product, ll_norm), pp_norm)
        return cos_dist

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        graph = tf.Graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config, graph=graph)
        with sess.graph.as_default():
            sample_quaternion = tf.placeholder(tf.float32, shape=[None, 4])
            target_quaternion = tf.placeholder(tf.float32, shape=[1, 4])
            cos_dist = self.cosine_distance(target_quaternion, sample_quaternion)
            gradients = tf.gradients(cos_dist, sample_quaternion)[0]
            hessians = []
            for i in range(gradients.get_shape()[1].value):
                hessian_i = tf.gradients(gradients[:, i], sample_quaternion)
                hessians.append(hessian_i[0])
            hessians = tf.pack(hessians, axis=1)

        T = sample.T
        Du = sample.dU
        Dx = sample.dX
        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        sample_quaternion_tmp = sample.get(END_EFFECTOR_ROTATIONS)
        target_quaternion_tmp = self._hyperparams['target_quaternion'].reshape(1, -1)

        wp = self._hyperparams['wp']
        l = sess.run([cos_dist],
                     feed_dict={sample_quaternion: sample_quaternion_tmp,
                                target_quaternion: target_quaternion_tmp})
        ls = sess.run([gradients],
                      feed_dict={sample_quaternion: sample_quaternion_tmp,
                                 target_quaternion: target_quaternion_tmp})
        lss = sess.run([hessians],
                       feed_dict={sample_quaternion: sample_quaternion_tmp,
                                  target_quaternion: target_quaternion_tmp})
        l = np.asarray(l[0]) * wp
        ls = np.asarray(ls[0]) * wp
        lss = np.asarray(lss[0]) * wp
        final_l += l

        sample.agent.pack_data_x(final_lx, ls, data_types=[END_EFFECTOR_ROTATIONS])
        sample.agent.pack_data_x(final_lxx, lss,
                                 data_types=[END_EFFECTOR_ROTATIONS, END_EFFECTOR_ROTATIONS])

        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux