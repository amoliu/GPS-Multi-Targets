""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import numpy as np

# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.

from gps.algorithm.policy_opt.config import POLICY_OPT_TF
import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.tf_utils import TfSolver


LOGGER = logging.getLogger(__name__)


class PolicyOptTf(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        tf.set_random_seed(self._hyperparams['random_seed'])
        self.graph = tf.Graph()
        # tf_config = tf.ConfigProto(log_device_placement=True)
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        with self.sess.graph.as_default():
            self.tf_iter = 0
            self.batch_size = self._hyperparams['batch_size']
            self.device_string = "/cpu:0"
            if self._hyperparams['use_gpu'] == 1:
                self.gpu_device = self._hyperparams['gpu_id']
                self.device_string = "/gpu:" + str(self.gpu_device)
            self.act_op = None  # mu_hat
            self.feat_op = None # features
            self.loss_scalar = None
            self.obs_tensor = None
            self.precision_tensor = None
            self.action_tensor = None  # mu true
            self.solver = None
            self.feat_vals = None
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.init_network()
            self.init_solver()
            self.var = self._hyperparams['init_var'] * np.ones(dU)

            self.policy = TfPolicy(dU, self.obs_tensor, self.act_op, self.feat_op,
                                   np.zeros(dU), self.sess, self.device_string, self.is_training,
                                   copy_param_scope=self._hyperparams['copy_param_scope'])
            # List of indices for state (vector) data and image (tensor) data in observation.
            self.x_idx, self.img_idx, i = [], [], 0
            if 'obs_image_data' not in self._hyperparams['network_params']:
                self._hyperparams['network_params'].update({'obs_image_data': []})
            for sensor in self._hyperparams['network_params']['obs_include']:
                dim = self._hyperparams['network_params']['sensor_dims'][sensor]
                if sensor in self._hyperparams['network_params']['obs_image_data']:
                    self.img_idx = self.img_idx + list(range(i, i+dim))
                else:
                    self.x_idx = self.x_idx + list(range(i, i+dim))
                i += dim

            file_path = self._hyperparams['weights_file_prefix']
            current_directory = os.path.dirname(os.path.realpath(__file__))
            folder_suffix = file_path[file_path.index('/experiments/'):]
            folder_prefix = current_directory[:current_directory.index('/python/gps')]
            self._hyperparams['weights_file_prefix'] = folder_prefix + folder_suffix
            self.train_dir = os.path.join(os.path.dirname(self._hyperparams['weights_file_prefix']),
                                          'tf_train')
            self.checkpoint_name = self._hyperparams['weights_file_prefix'].split('/')[-1] + '_model.ckpt'
            if not os.path.exists(self.train_dir):
                os.makedirs(self.train_dir)
            self.train_writer = tf.summary.FileWriter(self.train_dir,
                                                      graph=self.sess.graph)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        tf_map_generator = self._hyperparams['network_model']
        tf_map, fc_vars, last_conv_vars = tf_map_generator(is_training=self.is_training,
                                                           dim_input=self._dO,
                                                           dim_output=self._dU,
                                                           batch_size=self.batch_size,
                                                           network_config=self._hyperparams['network_params'],
                                                           device_string=self.device_string,
                                                           weight_decay=self._hyperparams['weight_decay'])
        self.obs_tensor = tf_map.get_input_tensor()
        self.precision_tensor = tf_map.get_precision_tensor()
        self.action_tensor = tf_map.get_target_output_tensor()
        self.act_op = tf_map.get_output_op()
        self.feat_op = tf_map.get_feature_op()
        self.loss_scalar = tf_map.get_loss_op()
        self.fc_vars = fc_vars
        self.last_conv_vars = last_conv_vars

        # Setup the gradients
        # self.grads = [tf.gradients(self.act_op[:,u], self.obs_tensor)[0]
        #         for u in range(self._dU)]

    def init_solver(self):
        """ Helper method to initialize the solver. """
        if self._hyperparams['lr_policy'] == 'fixed':
            self.solver = TfSolver(loss_scalar=self.loss_scalar,
                                   solver_name=self._hyperparams['solver_type'],
                                   base_lr=self._hyperparams['lr'],
                                   lr_policy=self._hyperparams['lr_policy'],
                                   momentum=self._hyperparams['momentum'],
                                   fc_vars=self.fc_vars,
                                   last_conv_vars=self.last_conv_vars,
                                   device_string=self.device_string,)
        elif self._hyperparams['lr_policy'] == 'exp':
            self.solver = TfSolver(loss_scalar=self.loss_scalar,
                                   solver_name=self._hyperparams['solver_type'],
                                   base_lr=self._hyperparams['lr'],
                                   lr_policy=self._hyperparams['lr_policy'],
                                   momentum=self._hyperparams['momentum'],
                                   fc_vars=self.fc_vars,
                                   last_conv_vars=self.last_conv_vars,
                                   decay_rate=self._hyperparams['decay_rate'],
                                   decay_steps=self._hyperparams['decay_steps'],
                                   device_string=self.device_string,)
        else:
            raise NotImplementedError('learning rate policies other than fixed and exp are not implemented')
        self.saver = tf.train.Saver(max_to_keep=0)

    def update(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        obs = copy.deepcopy(obs)
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        if self.policy.scale is None or self.policy.bias is None:
            self.policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            self.policy.bias = - np.mean(
                obs[:, self.x_idx].dot(self.policy.scale), axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.policy.scale) + self.policy.bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N*T / self.batch_size)
        idx = range(N*T)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.solver.get_last_conv_values(self.sess, feed_dict, num_values, self.batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch * self.batch_size))
                idx_i = idx[start_idx:start_idx+self.batch_size]
                feed_dict = {self.is_training: True,
                             self.last_conv_vars: conv_values[idx_i],
                             self.action_tensor: tgt_mu[idx_i],
                             self.precision_tensor: tgt_prc[idx_i]}
                train_loss = self.solver(feed_dict,
                                         global_step=self.tf_iter,
                                         sess=self.sess,
                                         device_string=self.device_string,
                                         use_fc_solver=True)
                average_loss += train_loss

                if (i+1) % 500 == 0:
                    print('===========  tensorflow iteration %d, average loss %f  ===========' % (
                        i + 1, average_loss / 500))
                    # LOGGER.debug('tensorflow iteration %d, average loss %f',
                    #                 i+1, average_loss / 500)
                    average_loss = 0
            average_loss = 0

        # actual training.
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.is_training: True,
                         self.obs_tensor: obs[idx_i],
                         self.action_tensor: tgt_mu[idx_i],
                         self.precision_tensor: tgt_prc[idx_i]}
            train_loss = self.solver(feed_dict,
                                     global_step=self.tf_iter,
                                     sess=self.sess,
                                     device_string=self.device_string)

            average_loss += train_loss
            if (i+1) % 100 == 0:
                print('===========  tensorflow iteration %d, average loss %f  ===========' % (
                    i + 1, average_loss / 50))
                # LOGGER.debug('tensorflow iteration %d, average loss %f',
                #              i+1, average_loss / 50)
                average_loss = 0

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.feat_op is not None:
            self.feat_vals = self.solver.get_var_values(self.sess, self.feat_op, feed_dict, num_values, self.batch_size)
        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

        # if self.tf_iter % 50000 == 0:
        #     self.save_model()

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * T * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.var = 1 / np.diag(A)
        self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))

        return self.policy

    def sl_update(self, obs, tgt_mu):
        """
        Update policy with only supervised learning.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
        """
        obs = copy.deepcopy(obs)
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # Reshape inputs.
        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.eye(dU)
        tgt_prc = np.tile(tgt_prc, (N*T, 1, 1))

        # Normalize obs, but only compute normalzation at the beginning.
        if self.policy.scale is None or self.policy.bias is None:
            self.policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            self.policy.bias = - np.mean(
                obs[:, self.x_idx].dot(self.policy.scale), axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.policy.scale) + self.policy.bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N*T / self.batch_size)
        idx = range(N*T)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.solver.get_last_conv_values(self.sess, feed_dict, num_values, self.batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch * self.batch_size))
                idx_i = idx[start_idx:start_idx+self.batch_size]
                feed_dict = {self.is_training: True,
                             self.last_conv_vars: conv_values[idx_i],
                             self.action_tensor: tgt_mu[idx_i],
                             self.precision_tensor: tgt_prc[idx_i]}
                train_loss = self.solver(feed_dict,
                                         global_step=self.tf_iter,
                                         sess=self.sess,
                                         device_string=self.device_string,
                                         use_fc_solver=True)
                average_loss += train_loss

                if (i+1) % 500 == 0:
                    print('===========  tensorflow iteration %d, average loss %f  ===========' % (
                        i + 1, average_loss / 500))
                    # LOGGER.debug('tensorflow iteration %d, average loss %f',
                    #                 i+1, average_loss / 500)
                    average_loss = 0
            average_loss = 0

        # actual training.
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.is_training: True,
                         self.obs_tensor: obs[idx_i],
                         self.action_tensor: tgt_mu[idx_i],
                         self.precision_tensor: tgt_prc[idx_i]}
            train_loss = self.solver(feed_dict,
                                     global_step=self.tf_iter,
                                     sess=self.sess,
                                     device_string=self.device_string)

            average_loss += train_loss
            if (i+1) % 100 == 0:
                print('===========  tensorflow iteration %d, average loss %f  ===========' % (
                    i + 1, average_loss / 50))
                # LOGGER.debug('tensorflow iteration %d, average loss %f',
                #              i+1, average_loss / 50)
                average_loss = 0

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.feat_op is not None:
            self.feat_vals = self.solver.get_var_values(self.sess, self.feat_op, feed_dict, num_values, self.batch_size)
        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

    def prob(self, obs):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        obs = copy.deepcopy(obs)
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        if self.policy.scale is not None:
            # TODO: Should prob be called before update?
            for n in range(N):
                obs[n, :, self.x_idx] = (obs[n, :, self.x_idx].T.dot(self.policy.scale)
                                         + self.policy.bias).T

        output = np.zeros((N, T, dU))

        for i in range(N):
            for t in range(T):
                # Feed in data.
                feed_dict = {self.is_training: False,
                             self.obs_tensor: np.expand_dims(obs[i, t], axis=0)}
                with tf.device(self.device_string):
                    output[i, t, :] = self.sess.run(self.act_op, feed_dict=feed_dict)

        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams['ent_reg'] = ent_reg

    def save_model(self):
        fname = os.path.join(self.train_dir, self.checkpoint_name)
        LOGGER.debug('Saving model to: %s', fname)
        self.saver.save(self.sess, fname, global_step=self.tf_iter)

    def restore_model(self, model_checkpoint_path):
        # Restores from checkpoint
        self.saver.restore(self.sess, model_checkpoint_path)
        LOGGER.debug('Restoring model from: %s', model_checkpoint_path)
        print 'Restoring model from: ', model_checkpoint_path

    # For pickling.
    def __getstate__(self):
        self.save_model()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': self.policy.scale,
            'bias': self.policy.bias,
            'tf_iter': self.tf_iter,
            'x_idx': self.policy.x_idx,
            'chol_pol_covar': self.policy.chol_pol_covar
        }

    # For unpickling.
    def __setstate__(self, state):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        self.policy.scale = state['scale']
        self.policy.bias = state['bias']
        self.policy.x_idx = state['x_idx']
        self.policy.chol_pol_covar = state['chol_pol_covar']
        self.tf_iter = state['tf_iter']
        self.restore_model(os.path.join(self.train_dir, self.checkpoint_name+'-'+str(self.tf_iter)))

