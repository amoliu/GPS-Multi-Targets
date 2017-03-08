""" Hyperparameters for Box2d Point Mass task with PIGPS."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from gps import __file__ as gps_filepath
from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.point_mass_world import PointMassWorld
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, END_EFFECTOR_POINT_TARGET_POSITION, ACTION
from gps.gui.config import generate_experiment_info
from gps.algorithm.policy_opt.tf_model_example import example_tf_network, example_tf_network_with_BN

SENSOR_DIMS = {
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    END_EFFECTOR_POINT_TARGET_POSITION: 3,
    ACTION: 2
}

random_seed = 12
np.random.seed(random_seed)

def generate_goals(bounds, start_num, target_num, random=False):
    if not random:
        target_state = np.mgrid[bounds[0][0]:bounds[0][1]:complex(target_num[0]),
                       bounds[1][0]:bounds[1][1]:complex(target_num[1])].reshape(2, -1).T
        target_state = np.concatenate((target_state, np.zeros((np.prod(target_num), 1))), axis=1)

        x0 = np.mgrid[bounds[0][0] + 5: bounds[0][1] - 5:complex(start_num[0]),
             bounds[1][0] + 5: bounds[1][1] - 5:complex(start_num[1])].reshape(2, -1).T

        x0 = np.concatenate((x0, np.zeros((np.prod(start_num), 4))), axis=1)
    else:
        target_state = [np.random.uniform(bounds[i][0], bounds[i][1], (np.prod(target_num), 1)) for i in range(2)]
        target_state = np.concatenate(target_state, axis=1)
        target_state = np.concatenate((target_state, np.zeros((np.prod(target_num), 1))), axis=1)
        x0 = [np.random.uniform(bounds[i][0], bounds[i][1], (np.prod(start_num), 1)) for i in range(2)]
        x0 = np.concatenate(x0, axis=1)
        x0 = np.concatenate((x0, np.zeros((np.prod(start_num), 4))), axis=1)

    target_state = np.tile(target_state, (np.prod(start_num), 1))
    x0 = np.reshape(np.tile(x0, (1, np.prod(target_num))), (-1, 6))
    return x0, target_state

bounds = np.array([[-8, 18],
                   [5, 35]])
start_num = [2, 2]
target_num = [2, 2]

RANDOM = False
if RANDOM:
    x0, target_state = generate_goals(bounds,
                                      start_num,
                                      target_num,
                                      random=True)
else:
    x0, target_state = generate_goals(bounds,
                                      start_num,
                                      target_num,
                                      random=False)
x0 = np.concatenate((x0, target_state), axis=1)
plt.plot(x0[:, 0], x0[:, 1], 'ro', label='start')
plt.plot(target_state[:, 0], target_state[:, 1], 'bs', label='target')
plt.legend(ncol=1, prop={'size': 12})
plt.axis('equal')
plt.xlim([bounds[0][0] - 5, bounds[0][1] + 15])
plt.ylim([bounds[1][0] - 5, bounds[1][1] + 5])
# plt.ion()
# plt.show()
plt.draw()
iteration = 20


EXP_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'

common = {
    'experiment_name': EXP_DIR.split('/')[-1] + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': x0.shape[0]
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentBox2D,
    'target_state': target_state,
    "world": PointMassWorld,
    'render': False,
    'x0': x0,
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, END_EFFECTOR_POINT_TARGET_POSITION],
    'obs_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, END_EFFECTOR_POINT_TARGET_POSITION],
    'smooth_noise_var': 3.0,
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    # 'policy_sample_mode': 'replace',
    # 'sample_on_policy': True,
    'iterations': iteration,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-1]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    # 'init_pol_wt': 0.01,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    # 'exp_step_increase': 2.0,
    # 'exp_step_decrease': 0.5,
    # 'exp_step_upper': 0.5,
    # 'exp_step_lower': 1.0,
    'max_policy_samples': 20,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_var': 1.0,
    'pos_gains': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.array([5e-5, 5e-5])
}

state_cost = {
    'type': CostState,
    'data_types' : {
        END_EFFECTOR_POINTS: {
            'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
            'target_state': agent["target_state"],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1.0, 1.0],
}


algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
    'kl_threshold': 2.0,
    'covariance_damping': 2.0,
    'min_temperature': 0.001,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'weights_file_prefix': EXP_DIR + 'policy',
    'iterations': 10000,
    'lr': 0.001,
    'network_params': {
        'dim_hidden': [128, 64],
        'obs_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'sensor_dims': SENSOR_DIMS,
        'dropout_keep_prob': 0.8,
    },
    'network_model': example_tf_network,
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': iteration,
    'num_samples': 5,
    'common': common,
    'verbose_trials': 1,
    'verbose_policy_trials': 0,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'random_seed': random_seed,
}

common['info'] = generate_experiment_info(config)
