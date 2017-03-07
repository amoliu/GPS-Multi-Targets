# To get started, copy over hyperparams from another experiment.
# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.
""" Hyperparameters for UR trajectory optimization experiment. """
from __future__ import division
import numpy as np
import os.path
import rospkg
import copy
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gps import __file__ as gps_filepath
from gps.agent.ur_ros.agent_ur import AgentURROS
from gps.algorithm.algorithm_pigps import AlgorithmPIGPS
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_quaternion import CostQuaternion
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPi2
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, END_EFFECTOR_ROTATIONS, ACTION, \
        TRIAL_ARM, JOINT_SPACE
from gps.utility.general_utils import get_ee_points, get_position, get_rotation_matrix, quaternion_from_matrix
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.gui.config import generate_experiment_info
from gps.utility.general_utils import forward_kinematics, inverse_kinematics
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

random_seed = 12
np.random.seed(random_seed)

def generate_goals(robot_chain, initial_bounds, target_bounds, initial_num, target_num, random=False):
    """
    :param robot_chain: robot chain
    :param initial_bounds: 3 x 2 numpy array, min and max values for x, y, z for starting point
    :param target_bounds: 3 x 2 numpy array, min and max values for x, y, z for target point
    :param initial_num: number of points on x, y, z axis direction for starting point
    :param target_num: number of points on x, y, z axis direction for target point
    :param random: whether to generate points randomly within the given bounds
    :return: starting end-effector position, starting robot joint angles, target points
    """
    min_joints = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -2 * np.pi])
    max_joints = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, 2 * np.pi])
    rot = np.eye(3)
    initial_joint_state = []
    initial_state_cart = []
    if not random:
        target_state = np.mgrid[target_bounds[0][0]:target_bounds[0][1]:complex(target_num[0]),
                       target_bounds[1][0]:target_bounds[1][1]:complex(target_num[1]),
                       target_bounds[2][0]:target_bounds[2][1]:complex(target_num[2])].reshape(3, -1).T
        initial_state = np.mgrid[initial_bounds[0][0]: initial_bounds[0][1]: complex(initial_num[0]),
                        initial_bounds[1][0]: initial_bounds[1][1]: complex(initial_num[1]),
                        initial_bounds[2][0]: initial_bounds[2][1]: complex(initial_num[2])].reshape(3, -1).T

        for i in range(target_state.shape[1]):
            target_state[:, i] = target_state[:, i] if target_num[i] > 1 else target_bounds.mean(axis=1)[i]

        for i in range(initial_state.shape[1]):
            initial_state[:, i] = initial_state[:, i] if initial_num[i] > 1 else initial_bounds.mean(axis=1)[i]

        for i in range(initial_state.shape[0]):
            pos = initial_state[i, :]
            joint_state = inverse_kinematics(robot_chain, pos, rot, min_joints=min_joints, max_joints=max_joints)
            if joint_state is not None:
                initial_joint_state.append(joint_state)
                initial_state_cart.append(pos)
            else:
                print 'The following training initial state position cannot be solved by KDL IK solver: '
                print '     Position:', pos
                print '     Orientation:'
                for idx in range(rot.shape[0]):
                    print '                 ', rot[idx]

    else:
        target_state = [np.random.uniform(target_bounds[i][0], target_bounds[i][1], (np.prod(target_num), 1))
                        for i in range(3)]
        target_state = np.concatenate(target_state, axis=1)
        try_times = 0
        while True:
            initial_state = [np.random.uniform(initial_bounds[i][0], initial_bounds[i][1], (1, 1))
                            for i in range(3)]
            initial_state = np.concatenate(initial_state, axis=1)
            pos = initial_state[0]
            joint_state = inverse_kinematics(robot_chain, pos, rot, min_joints=min_joints, max_joints=max_joints)
            try_times += 1
            if joint_state is not None:
                initial_joint_state.append(joint_state)
                initial_state_cart.append(pos)
                if len(initial_state_cart) >= np.prod(initial_num):
                    break
            elif try_times > 50:
                raise Exception('Cannot find a suitable starting position within 50 trials')
            else:
                continue
    # initial_state_cart = np.array(initial_state_cart)
    # initial_joint_state = np.array(initial_joint_state)
    initial_state_cart = np.array([[-0.57, -0.26, 0.62]])
    initial_joint_state = np.array([[0, -np.pi / 2, np.pi / 2, 0, np.pi / 2, 0]])
    # initial_joint_state = np.array([[np.pi / 2, -np.pi / 2, np.pi / 2, 0, 0, 0]])
    num_target_state = target_state.shape[0]
    num_initial_state = initial_joint_state.shape[0]

    print '================================'
    print '================================'
    print '          Robot States          '
    print '================================'
    print '================================'
    print('Starting states (total number {:d})'.format(initial_state_cart.shape[0]))
    for i in range(initial_state_cart.shape[0]):
        print '  ', np.around(initial_state_cart[i], 2)
    print('Target states (total number {:d})'.format(target_state.shape[0]))
    for i in range(target_state.shape[0]):
        print '  ', np.around(target_state[i], 2)

    target_state = np.tile(target_state, (num_initial_state, 1))
    initial_joint_state = np.reshape(np.tile(initial_joint_state, (1, num_target_state)), (-1, 6))
    return initial_state_cart, initial_joint_state, target_state

# Only edit these when editing the robot joints and links.
# The lengths of these arrays define numerous parameters in GPS.
JOINT_ORDER = [SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT,
               WRIST_1_JOINT, WRIST_2_JOINT, WRIST_3_JOINT]
LINK_NAMES = [BASE, BASE_LINK, SHOULDER_LINK, UPPER_ARM_LINK, FOREARM_LINK,
              WRIST_1_LINK, WRIST_2_LINK, WRIST_3_LINK, EE_LINK]

UR_PREFIXES = ['u5', 'u6', 'u7', 'u8', 'u9']
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

UR_GAINS = np.array([2.195, 1.922, 1.582, 1.393, 1.151, 1.152])

# Path to urdf of robot.
rospack = rospkg.RosPack()
TREE_PATH = rospack.get_path('ur_description') + '/urdf/ur10_robot.urdf'
_, ur_tree = treeFromFile(TREE_PATH)
# Retrieve a chain structure between the base and the start of the end effector.
ur_chain = ur_tree.getChain(m_link_names[0][0], m_link_names[0][-1])

initial_state_num = [1, 1, 1] #Training conditions
target_state_num = [3, 6, 5] #Training conditions
# initial_state_num = [1, 1, 1]
# target_state_num = [3, 5, 4]
# boundaries for x, y, z coordinates
goal_position_bounds = np.array([[-0.9, -0.50],
                                 [-0.5, 0.5],
                                 [-0.4, 0.4]])
start_position_bounds = goal_position_bounds
# start_position_bounds = np.array([[-0.50, -0.90],
#                                  [-0.6, 0.6],
#                                  [0.00, 0.80]])


RANDOM = False
if RANDOM:
    initial_state_cart, initial_joint_state, target_state = generate_goals(ur_chain,
                                                                           start_position_bounds,
                                                                           goal_position_bounds,
                                                                           initial_state_num,
                                                                           target_state_num,
                                                                           random=True)
else:
    initial_state_cart, initial_joint_state, target_state = generate_goals(ur_chain,
                                                                           start_position_bounds,
                                                                           goal_position_bounds,
                                                                           initial_state_num,
                                                                           target_state_num,
                                                                           random=False)
# # Set goal points w.r.t. ee_link coordinate system
EE_POINTS = np.asmatrix([[0.00, 0.00, 0.00],
                         [0.00, 0.00, 0.10],
                         [0.00, 0.10, 0.00]])
INITIAL_JOINTS = initial_joint_state
# Specify a goal state in cartesian coordinates.
EE_POS_TGT = np.asmatrix(target_state)

# Set to identity unless you want the goal to have a certain orientation.
rotation_matrix = get_rotation_matrix(-np.pi / 2, [0, 1, 0])
tgt_quaternion = quaternion_from_matrix(rotation_matrix)
EE_ROT_TGT = np.asmatrix(rotation_matrix[:3, :3])

def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(initial_state_cart[:, 0], initial_state_cart[:, 1], initial_state_cart[:, 2],
           c='r', marker='o', label='train_start')
ax.scatter(target_state[:, 0], target_state[:, 1], target_state[:, 2],
           c='b', marker='^', label='train_target')
ax.legend(ncol=1, prop={'size': 12})
# ax.set_aspect('equal')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
set_aspect_equal_3d(ax)
# ax.auto_scale_xyz([train_goal_position_bounds[0][0], train_goal_position_bounds[0][1]],
#                   [train_goal_position_bounds[1][0], train_goal_position_bounds[1][1]],
#                   [train_goal_position_bounds[2][0], train_goal_position_bounds[2][1]])
# plt.ion()
# plt.show()
plt.draw()

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


# Be sure to specify the correct experiment directory to save policy data at.
EXP_DIR = os.path.dirname(os.path.realpath(__file__)) + '/test/'

# Set the number of seconds per step of a sample.
TIMESTEP = 0.06  # Typically 0.01.
# Set the number of timesteps per sample.
STEP_COUNT = 100  # Typically 100.
# Set the number of samples per condition.
SAMPLE_COUNT = 30  # Typically 5.
# set the number of conditions per iteration.
CONDITIONS = initial_joint_state.shape[0]
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

    ee_pos_tgt = EE_POS_TGT[i]
    ee_rot_tgt = EE_ROT_TGT

    state_space = sum(SENSOR_DIMS.values()) - SENSOR_DIMS[ACTION]

    joint_dim = SENSOR_DIMS[JOINT_ANGLES] + SENSOR_DIMS[JOINT_VELOCITIES]

    # Initialized to start position and inital velocities are 0
    x0 = np.zeros(state_space)
    x0[:SENSOR_DIMS[JOINT_ANGLES]] = ja_x0

    trans, rot = forward_kinematics(ur_chain,
                                    m_link_names[0],
                                    INITIAL_JOINTS[i],
                                    base_link=m_link_names[0][0],
                                    end_link=m_link_names[0][-1])
    x0[joint_dim:(joint_dim + EE_POINTS.size)] = np.ndarray.flatten(get_ee_points(EE_POINTS, trans, rot).T)


    # Initialize target end effector position
    ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)

    reset_condition = {
        JOINT_ANGLES: INITIAL_JOINTS[i],
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
    'offline_demo': True,
    'parallel_num': len(UR_PREFIXES),
    'parallel_on_conditions': True,
    'num_samples': SAMPLE_COUNT,
    'demo_pos_sigma': 0.01,
    'demo_quat_sigma': 0.001,
    'noise_on': 'noise_on_actions',#'noise_on_target'# 'noise_on_actions'
    'action_noise_sigma': 0.2,
    'action_noise_mu': 0.0,
    'action_smooth_noise_var': 2.0,
}

algorithm = {
    'type': AlgorithmPIGPS,
    'iterations': ITERATIONS,
    'conditions': common['conditions'],
    'policy_sample_mode': 'replace',
    'sample_on_policy': True,
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
quaternion_cost = {
    'type': CostQuaternion,
    'wp': 0.03 * np.array([2/np.pi, 1.0, 1.0, 1.0]),#np.ones(SENSOR_DIMS[END_EFFECTOR_ROTATIONS]),
    'target_quaternion': tgt_quaternion
}
# Combines the cost functions in 'costs' to produce a single cost function
algorithm['cost'] = {
    'type': CostSum,
    'costs': [fk_cost_ramp, fk_cost_final],
    'weights': [1.0, 1.0, 1.0],
}
# algorithm['cost'] = {
#     'type': CostSum,
#     'costs': [fk_cost_ramp, fk_cost_final],
#     'weights': [1.0, 1.0],
# }

algorithm['traj_opt'] = {
    'type': TrajOptPi2,
    'kl_threshold': 2.0,
    'covariance_damping': 2.0,
    'min_temperature': 0.001,
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
        'dim_hidden': [68, 128, 64],
        'activation_fn': 'elu',
    },
    'network_model': example_tf_network,
}


algorithm['policy_prior'] = {
    'type': PolicyPrior,
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
