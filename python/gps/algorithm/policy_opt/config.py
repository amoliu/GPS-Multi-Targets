""" Default configuration for policy optimization. """
try:
    from gps.algorithm.policy_opt.policy_opt_utils import construct_fc_network
except ImportError:
    construct_fc_network = None

import os

# config options shared by both caffe and tf.
GENERIC_CONFIG = {
    # Initialization.
    'init_var': 0.1,  # Initial policy variance.
    'ent_reg': 0.0,  # Entropy regularizer.
    # Solver hyperparameters.
    'iterations': 10000,  # Number of iterations per inner iteration.
    'batch_size': 128,
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.005, # Weight decay.
    'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', etc.).
    # set gpu usage.
    'use_gpu': 1,  # Whether or not to use the GPU for caffe training.
    'gpu_id': 0,
    'random_seed': 1,
}


POLICY_OPT_CAFFE = {
    # Other hyperparameters.
    'lr_policy': 'fixed',  #'fixed' for caffe, Learning rate policy.
    'network_model': construct_fc_network,  # Either a filename string
                                            # or a function to call to
                                            # create NetParameter.
    'network_arch_params': {},  # Arguments to pass to method above.
    'weights_file_prefix': '',
}

POLICY_OPT_CAFFE.update(GENERIC_CONFIG)


POLICY_OPT_TF = {
    # Other hyperparameters.
    'lr_policy': 'fixed',  #'fixed' or 'exp' for tensorflow, Learning rate policy.
    'copy_param_scope': 'conv_params',
    'fc_only_iterations': 0,
}
POLICY_OPT_TF.update(GENERIC_CONFIG)
if POLICY_OPT_TF['lr_policy'] == 'exp':
    POLICY_OPT_TF['decay_rate'] = 0.96
    POLICY_OPT_TF['decay_steps'] = 50000


