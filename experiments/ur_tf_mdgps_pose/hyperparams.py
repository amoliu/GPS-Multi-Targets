# To get started, copy over hyperparams from another experiment.
# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.
""" Hyperparameters for UR trajectory optimization experiment. """
from __future__ import division
import numpy as np
import os.path
import rospkg
from datetime import datetime
import copy
from gps import __file__ as gps_filepath
from gps.agent.ur_ros.agent_ur import AgentURROS
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, JOINT_SPACE
from gps.utility.general_utils import get_ee_points, get_position, get_rotation_matrix, quaternion_from_matrix
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.gui.config import generate_experiment_info
from gps.utility.general_utils import forward_kinematics
from gps.agent.ur_ros.tree_urdf import treeFromFile
# Topics for the robot publisher and subscriber.
JOINT_PUBLISHER = '/arm_controller/command'
JOINT_SUBSCRIBER = '/arm_controller/state'

# 'SLOWNESS' is how far in the future (in seconds) position control extrapolates
# when it publishs actions for robot movement.  1.0-10.0 is fine for simulation.
SLOWNESS = 1.0
# 'RESET_SLOWNESS' is how long (in seconds) we tell the robot to take when
# returning to its start configuration.
RESET_SLOWNESS = 1.0

# Set constants for joints
SHOULDER_PAN_JOINT = 'shoulder_pan_joint'
SHOULDER_LIFT_JOINT = 'shoulder_lift_joint'
ELBOW_JOINT = 'elbow_joint'
WRIST_1_JOINT = 'wrist_1_joint'
WRIST_2_JOINT = 'wrist_2_joint'
WRIST_3_JOINT = 'wrist_3_joint'

# Set constants for links
BASE = 'base'
BASE_LINK = 'base_link'
SHOULDER_LINK = 'shoulder_link'
UPPER_ARM_LINK = 'upper_arm_link'
FOREARM_LINK = 'forearm_link'
WRIST_1_LINK = 'wrist_1_link'
WRIST_2_LINK = 'wrist_2_link'
WRIST_3_LINK = 'wrist_3_link'
EE_LINK = 'ee_link'
# Set end effector constants
INITIAL_JOINTS = np.array([0, 0, 0, 0, 0, 0])

# # Set goal points w.r.t. ee_link coordinate system
EE_POINTS = np.asmatrix([[0.00, 0.00, 0.00],
                         [0.10, 0.10, 0.10],
                         [0.00, 0.10, 0.00],
                         [0.10, 0.00, 0.00]])
# EE_POINTS = np.asmatrix([[0.00, 0.00, 0.00]])
# Specify a goal state in cartesian coordinates.
EE_POS_TGT = np.asmatrix([-0.50, -0.60, .50])

# Set to identity unless you want the goal to have a certain orientation.
rotation_matrix = get_rotation_matrix(np.pi / 2, [1, 0, 0])
tgt_quaternion = quaternion_from_matrix(rotation_matrix)
EE_ROT_TGT = np.asmatrix(rotation_matrix[:3, :3])
# EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


# Only edit these when editing the robot joints and links.
# The lengths of these arrays define numerous parameters in GPS.
JOINT_ORDER = [SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT,
               WRIST_1_JOINT, WRIST_2_JOINT, WRIST_3_JOINT]
LINK_NAMES = [BASE, BASE_LINK, SHOULDER_LINK, UPPER_ARM_LINK, FOREARM_LINK,
              WRIST_1_LINK, WRIST_2_LINK, WRIST_3_LINK, EE_LINK]


UR_PREFIXES = ['u1']
m_joint_order = [copy.deepcopy(JOINT_ORDER) for prefix in UR_PREFIXES]
m_link_names = [copy.deepcopy(LINK_NAMES) for prefix in UR_PREFIXES]
m_joint_publishers = [copy.deepcopy(JOINT_PUBLISHER) for prefix in UR_PREFIXES]
m_joint_subscribers = [copy.deepcopy(JOINT_SUBSCRIBER) for prefix in UR_PREFIXES]
for prefix_idx in range(len(UR_PREFIXES)):
    UR_PREFIX = UR_PREFIXES[prefix_idx]
    if UR_PREFIX:
        for i, item in enumerate(m_joint_order[prefix_idx]):
            m_joint_order[prefix_idx][i] = UR_PREFIX +'_' + item

        for i, item in enumerate(m_link_names[prefix_idx]):
            m_link_names[prefix_idx][i] = UR_PREFIX +'_' + item

        insert_idx = m_joint_publishers[prefix_idx].index('arm_controller')+len('arm_controller')
        m_joint_publishers[prefix_idx] = m_joint_publishers[prefix_idx][:insert_idx] + '_' + UR_PREFIX + m_joint_publishers[prefix_idx][insert_idx:]
        insert_idx = m_joint_subscribers[prefix_idx].index('arm_controller')+len('arm_controller')
        m_joint_subscribers[prefix_idx] = m_joint_subscribers[prefix_idx][:insert_idx] + '_' + UR_PREFIX + m_joint_subscribers[prefix_idx][insert_idx:]
ROS_NODE_SUFFIX = UR_PREFIXES[0]

# Hyperparamters to be tuned for optimizing policy learning on the specific robot.
# UR_GAINS = np.array([2.195, 0.925, 0.600, 0.393, 0.211, 0.192])
# UR_GAINS = np.array([2.195, 0.922, 0.582, 0.393, 0.211, 0.192])
# UR_GAINS = np.array([2.195, 0.925, 0.585, 0.393, 0.211, 0.192])
# UR_GAINS = np.array([2.195, 0.922, 0.582, 0.393, 0.151, 0.152]) #good for one ee point
UR_GAINS = np.array([2.195, 1.922, 1.582, 1.393, 1.151, 1.152])
# Packaging sensor dimensional data for reference.
SENSOR_DIMS = {
    JOINT_ANGLES: len(JOINT_ORDER),
    JOINT_VELOCITIES: len(JOINT_ORDER),
    END_EFFECTOR_POINTS: EE_POINTS.size,
    END_EFFECTOR_POINT_VELOCITIES: EE_POINTS.size,
    ACTION: len(UR_GAINS),
}

# States to check in agent._process_observations.
STATE_TYPES = {'positions': JOINT_ANGLES,
               'velocities': JOINT_VELOCITIES}


# Path to urdf of robot.
rospack = rospkg.RosPack()
TREE_PATH = rospack.get_path('ur_description') + '/urdf/ur10_robot.urdf'
_, ur_tree = treeFromFile(TREE_PATH)
# Retrieve a chain structure between the base and the start of the end effector.
ur_chain = ur_tree.getChain(LINK_NAMES[0], LINK_NAMES[-1])
# Be sure to specify the correct experiment directory to save policy data at.
EXP_DIR = os.path.dirname(os.path.realpath(__file__)) + '/test/'

# Set the number of seconds per step of a sample.
TIMESTEP = 0.05  # Typically 0.01.
# Set the number of timesteps per sample.
STEP_COUNT = 100  # Typically 100.
# Set the number of samples per condition.
SAMPLE_COUNT = 5  # Typically 5.
# set the number of conditions per iteration.
CONDITIONS = 1  # Typically 2 for Caffe and 1 for LQR.
# Set the number of trajectory iterations to collect.
ITERATIONS = 20  # Typically 10.

x0s = []
ee_tgts = []
reset_conditions = []
tgt_quaternions = []

common = {
    'experiment_name': 'my_experiment' + '_' + \
                       datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
}

# Set up each condition.
for i in xrange(common['conditions']):
    # Use hardcoded default vals init and target locations
    ja_x0 = np.zeros(SENSOR_DIMS[JOINT_ANGLES])
    ee_pos_x0 = np.zeros((1, 3))
    ee_rot_x0 = np.zeros((3, 3))

    ee_pos_tgt = EE_POS_TGT
    ee_rot_tgt = EE_ROT_TGT

    state_space = sum(SENSOR_DIMS.values()) - SENSOR_DIMS[ACTION]

    joint_dim = SENSOR_DIMS[JOINT_ANGLES] + SENSOR_DIMS[JOINT_VELOCITIES]

    # Initialized to start position and inital velocities are 0
    x0 = np.zeros(state_space)
    x0[:SENSOR_DIMS[JOINT_ANGLES]] = ja_x0

    trans, rot = forward_kinematics(ur_chain,
                                    m_link_names[0],
                                    INITIAL_JOINTS,
                                    base_link=m_link_names[0][0],
                                    end_link=m_link_names[0][-1])
    x0[joint_dim:(joint_dim + EE_POINTS.size)] = np.ndarray.flatten(get_ee_points(EE_POINTS, trans, rot).T)


    # Initialize target end effector position
    ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)

    reset_condition = {
        JOINT_ANGLES: INITIAL_JOINTS,
        JOINT_VELOCITIES: []
    }

    x0s.append(x0)
    ee_tgts.append(ee_tgt)
    reset_conditions.append(reset_condition)
    tgt_quaternions.append(tgt_quaternion)
x0s = np.array(x0s)
ee_tgts = np.array(ee_tgts)
if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])


agent = {
    'type': AgentURROS,
    'dt': TIMESTEP,
    'dU': SENSOR_DIMS[ACTION],
    'conditions': common['conditions'],
    'T': STEP_COUNT,
    'x0': x0s,
    'ee_points_tgt': ee_tgts,
    'reset_conditions': reset_conditions,
    'sensor_dims': SENSOR_DIMS,
    'joint_order': m_joint_order,
    'link_names': m_link_names,
    'state_types': STATE_TYPES,
    'tree_path': TREE_PATH,
    'joint_publisher': m_joint_publishers,
    'joint_subscriber': m_joint_subscribers,
    'slowness': SLOWNESS,
    'reset_slowness': RESET_SLOWNESS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES,
                      END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'end_effector_points': EE_POINTS,
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'node_suffix': ROS_NODE_SUFFIX,
    'control_plot_dir': EXP_DIR + 'test_control_plots/',
    'ee_quaternion_tgt': tgt_quaternions,
    'demo': True,
    'demo_trials': SAMPLE_COUNT,
    'demo_pos_var': 0.06,
    'demo_quat_var': 0.05,
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': ITERATIONS,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-1]),
    'policy_dual_rate': 0.1,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'init_pol_wt': 0.01,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'exp_step_increase': 2.0,
    'exp_step_decrease': 0.5,
    'exp_step_upper': 0.5,
    'exp_step_lower': 1.0,
    'max_policy_samples': 6,
    'policy_sample_mode': 'add',
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': 1.0 / UR_GAINS,#np.ones(SENSOR_DIMS[ACTION])
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 0.5,
    'stiffness_vel': .25, 
    'final_weight': 50, 
    'dt': agent['dt'],
    'T': agent['T'],
}

# This cost function takes into account the distance between the end effector's
# current and target positions, weighted in a linearly increasing fassion
# as the number of trials grows from 0 to T-1. 
fk_cost_ramp = {
    'type': CostFK,
    # Target end effector is subtracted out of EE_POINTS in pr2 c++ plugin so goal
    # is 0. The UR agent also subtracts this out for consistency.
    'target_end_effector': [np.zeros(EE_POINTS.size)],
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 0.0001,
    'ramp_option': RAMP_LINEAR,
}

# This cost function takes into account the distance between the end effector's
# current and target positions at time T-1 only.
fk_cost_final = {
    'type': CostFK,
    'target_end_effector': np.zeros(EE_POINTS.size),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 1.0,
    'l2': 0.0,
    'wp_final_multiplier': 10.0,  # Weight multiplier on final timestep.
    'ramp_option': RAMP_FINAL_ONLY,
}

action_cost = {
    'type': CostAction,
    'wu': 0.0155 / UR_GAINS,
}
# Combines the cost functions in 'costs' to produce a single cost function
# algorithm['cost'] = {
#     'type': CostSum,
#     'costs': [action_cost, fk_cost_ramp, fk_cost_final],
#     'weights': [1.0, 1.0, 1.0],
# }
algorithm['cost'] = {
    'type': CostSum,
    'costs': [fk_cost_ramp, fk_cost_final],
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
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'weights_file_prefix': EXP_DIR + 'policy',
    'iterations': 10000,
    # 'network_arch_params': {
    #     'n_layers': 3,
    #     'dim_hidden': [40],
    # },
    'network_params': {
        'obs_include': agent['obs_include'],
        'sensor_dims': SENSOR_DIMS,
        'dim_hidden': [128, 64],
    },
    'network_model': example_tf_network,
}


algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}

config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': 0,
    'verbose_policy_trials': 1,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': SAMPLE_COUNT,
}

common['info'] = generate_experiment_info(config)
