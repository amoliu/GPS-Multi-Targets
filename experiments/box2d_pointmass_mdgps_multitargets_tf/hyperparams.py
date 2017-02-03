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
from gps.algorithm.policy_opt.tf_model_example import example_tf_network

SENSOR_DIMS = {
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    END_EFFECTOR_POINT_TARGET_POSITION: 3,
    ACTION: 2
}

random_seed = 12
np.random.seed(random_seed)
bound_x = [-8, 18]
bound_y = [5, 35]

def generate_goals(bound_x, bound_y, target_x_num, target_y_num, start_x_num, start_y_num, random=False):
    if not random:
        target_state = np.mgrid[bound_x[0]:bound_x[1]:complex(target_x_num),
                       bound_y[0]:bound_y[1]:complex(target_y_num)].reshape(2, -1).T
        target_state = np.concatenate((target_state, np.zeros((target_x_num * target_y_num, 1))), axis=1)

        x0 = np.mgrid[bound_x[0] + 5: bound_x[1] - 5:complex(start_x_num),
             bound_y[0] + 5: bound_y[1] - 5:complex(start_y_num)].reshape(2, -1).T

        x0 = np.concatenate((x0, np.zeros((start_x_num * start_y_num, 4))), axis=1)
    else:
        target_state = np.random.uniform(bound_x[0], bound_x[1], (target_x_num * target_y_num, 1))
        target_state = np.concatenate((target_state,
                                       np.random.uniform(bound_y[0], bound_y[1],
                                                         (target_x_num * target_y_num, 1))),
                                      axis=1)
        target_state = np.concatenate((target_state, np.zeros((target_x_num * target_y_num, 1))),
                                      axis=1)
        x0 = np.random.uniform(bound_x[0], bound_x[1], (start_x_num * start_y_num, 1))
        x0 = np.concatenate((x0, np.random.uniform(bound_y[0], bound_y[1], (start_x_num * start_y_num, 1))), axis=1)
        x0 = np.concatenate((x0, np.zeros((start_x_num * start_y_num, 4))), axis=1)

    target_state = np.tile(target_state, (start_x_num * start_y_num, 1))
    x0 = np.reshape(np.tile(x0, (1, target_x_num * target_y_num)), (-1, 6))
    return x0, target_state

start_x_num = 2
start_y_num = 2
start_conditions = start_x_num * start_y_num
target_x_num = 3
target_y_num = 3
target_state_conditions = target_x_num * target_y_num

x0, target_state = generate_goals(bound_x,
                                  bound_y,
                                  target_x_num,
                                  target_y_num,
                                  start_x_num,
                                  start_y_num)
x0 = np.concatenate((x0, target_state), axis=1)
train = True
if train:
    start_x_num_test = start_x_num
    start_y_num_test = start_y_num
    target_x_num_test = target_x_num
    target_y_num_test = target_y_num
    start_conditions_test = start_x_num_test * start_y_num_test
    target_state_conditions_test = target_x_num_test * target_y_num_test
    x0_test, target_state_test = x0, target_state
else:
    start_x_num_test = 2
    start_y_num_test = 2
    target_x_num_test = 4
    target_y_num_test = 4
    start_conditions_test = start_x_num_test * start_y_num_test
    target_state_conditions_test = target_x_num_test * target_y_num_test
    x0_test, target_state_test = generate_goals(bound_x,
                                                bound_y,
                                                target_x_num_test,
                                                target_y_num_test,
                                                start_x_num_test,
                                                start_y_num_test,
                                                random=True)
    x0_test = np.concatenate((x0_test, target_state_test), axis=1)

plt.plot(x0[:, 0], x0[:, 1], 'ro', label='start_train')
plt.plot(target_state[:, 0], target_state[:, 1], 'bs', label='target_train')
plt.plot(x0_test[:, 0], x0_test[:, 1], 'go', label='start_test')
plt.plot(target_state_test[:, 0], target_state_test[:, 1], 'ys', label='target_test')
plt.legend(ncol=1, prop={'size': 12})
plt.axis('equal')
plt.xlim([bound_x[0] - 5, bound_x[1] + 15])
plt.ylim([bound_y[0] - 5, bound_y[1] + 5])
# plt.ion()
# plt.show()
plt.draw()
iteration = 50


EXP_DIR = os.path.dirname(os.path.realpath(__file__))

common = {
    'experiment_name': EXP_DIR.split('/')[-1] + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + '/data_files/',
    'target_filename': EXP_DIR + '/target.npz',
    'log_filename': EXP_DIR + '/log.txt',
    'train_conditions': start_conditions * target_state_conditions,
    'test_conditions': start_conditions_test * target_state_conditions_test,
}
if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentBox2D,
    'target_state' : target_state,
    "world" : PointMassWorld,
    'render' : False,
    'x0': x0,
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['train_conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, END_EFFECTOR_POINT_TARGET_POSITION],
    'obs_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, END_EFFECTOR_POINT_TARGET_POSITION],
    'smooth_noise_var': 3.0,
    'x0_test': x0_test,
    'target_state_test': target_state_test
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['train_conditions'],
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
    'weights_file_prefix': EXP_DIR + '/policy',
    'iterations': 10000,
    'network_params': {
        'dim_hidden': [128, 64],
        'obs_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'sensor_dims': SENSOR_DIMS,
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