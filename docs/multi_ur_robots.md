# Run Multiple gps_main.py with UR Robot Agent in Gazebo

## Table of Contents
<!-- MarkdownTOC -->

- Possible Errors
- Create Multiple UR Robots in Gazebo
    - ur10_robot.urdf.xacro
    - arm_controller_ur10.yaml
    - ur10.launch
- Modify GPS Agent for Running Multiple UR Agent Nodes
    - Example of `hyperparams.py`
    - Example of `agent_ur.py`

<!-- /MarkdownTOC -->

## Possible Errors

`Error 1`: cannot run gps with tensorflow, but can debug it.
`Solution`: add `export LD_PRELOAD="/usr/lib/libtcmalloc.so.4"` to your ~/.bashrc

## Create Multiple UR Robots in Gazebo

Take UR10 as an example, and we are gonna add two UR10 robots in Gazebo:

Modify three files: `ur10_robot.urdf.xacro`, `arm_controller_ur10.yaml`, `ur10.launch`

### ur10_robot.urdf.xacro
```bash
roscd ur_description
cd urdf
```

Edit `ur10_robot.urdf.xacro` to be like these:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro"
       name="ur10" >

  <!-- common stuff -->
  <xacro:include filename="$(find ur_description)/urdf/common.gazebo.xacro" />

  <!-- ur10 -->
  <xacro:include filename="$(find ur_description)/urdf/ur10.urdf.xacro" />

  <!-- arm -->
  <!-- <xacro:ur10_robot prefix="" joint_limited="false"/> -->
  <xacro:ur10_robot prefix="ur0_" joint_limited="false"/>
  <xacro:ur10_robot prefix="ur1_" joint_limited="false"/>
  
  <link name="world" />
<!--   <joint name="world_joint" type="fixed">
    <parent link="world" />
    <child link = "base_link" />
    <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0" />
  </joint> -->
  <joint name="world_joint_0" type="fixed">
    <parent link="world" />
    <child link = "ur0_base_link" />
    <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0" />
  </joint>

  <joint name="world_joint_1" type="fixed">
    <parent link="world" />
    <child link = "ur1_base_link" />
    <origin xyz="3.0 3.0 0.0" rpy="0.0 0.0 0.0" />
  </joint>
</robot>
```

Then run:
```bash
rosrun xacro xacro.py -o ur10_robot.urdf ur10_robot.urdf.xacro 
```

### arm_controller_ur10.yaml

```bash
roscd ur_gazebo
cd controller
```

Edit `arm_controller_ur10.yaml` to be like these:

```yaml
# arm_controller:
#   type: position_controllers/JointTrajectoryController
#   joints:
#      - shoulder_pan_joint
#      - shoulder_lift_joint
#      - elbow_joint
#      - wrist_1_joint
#      - wrist_2_joint
#      - wrist_3_joint
#   constraints:
#       goal_time: 0.6
#       stopped_velocity_tolerance: 0.05
#       shoulder_pan_joint: {trajectory: 0.1, goal: 0.1}
#       shoulder_lift_joint: {trajectory: 0.1, goal: 0.1}
#       elbow_joint: {trajectory: 0.1, goal: 0.1}
#       wrist_1_joint: {trajectory: 0.1, goal: 0.1}
#       wrist_2_joint: {trajectory: 0.1, goal: 0.1}
#       wrist_3_joint: {trajectory: 0.1, goal: 0.1}
#   stop_trajectory_duration: 0.5
#   state_publish_rate:  25
#   action_monitor_rate: 10

arm_controller_ur0:
  type: position_controllers/JointTrajectoryController
  joints:
     - ur0_shoulder_pan_joint
     - ur0_shoulder_lift_joint
     - ur0_elbow_joint
     - ur0_wrist_1_joint
     - ur0_wrist_2_joint
     - ur0_wrist_3_joint
  constraints:
      goal_time: 0.6
      stopped_velocity_tolerance: 0.05
      ur0_shoulder_pan_joint: {trajectory: 0.1, goal: 0.1}
      ur0_shoulder_lift_joint: {trajectory: 0.1, goal: 0.1}
      ur0_elbow_joint: {trajectory: 0.1, goal: 0.1}
      ur0_wrist_1_joint: {trajectory: 0.1, goal: 0.1}
      ur0_wrist_2_joint: {trajectory: 0.1, goal: 0.1}
      ur0_wrist_3_joint: {trajectory: 0.1, goal: 0.1}
  stop_trajectory_duration: 0.5
  state_publish_rate:  100
  action_monitor_rate: 50

arm_controller_ur1:
  type: position_controllers/JointTrajectoryController
  joints:
     - ur1_shoulder_pan_joint
     - ur1_shoulder_lift_joint
     - ur1_elbow_joint
     - ur1_wrist_1_joint
     - ur1_wrist_2_joint
     - ur1_wrist_3_joint
  constraints:
      goal_time: 0.6
      stopped_velocity_tolerance: 0.05
      ur1_shoulder_pan_joint: {trajectory: 0.1, goal: 0.1}
      ur1_shoulder_lift_joint: {trajectory: 0.1, goal: 0.1}
      ur1_elbow_joint: {trajectory: 0.1, goal: 0.1}
      ur1_wrist_1_joint: {trajectory: 0.1, goal: 0.1}
      ur1_wrist_2_joint: {trajectory: 0.1, goal: 0.1}
      ur1_wrist_3_joint: {trajectory: 0.1, goal: 0.1}
  stop_trajectory_duration: 0.5
  state_publish_rate:  100
  action_monitor_rate: 50
```

### ur10.launch

```bash
roscd ur_gazebo
cd launch
```

Edit `ur10.launch` to be like these:

```xml
<?xml version="1.0"?>
<launch>
  <arg name="limited" default="false"/>
  <arg name="paused" default="false"/>
  <arg name="gui" default="true"/>
  
  <!-- startup simulated world -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" default="worlds/empty.world"/>
    <arg name="paused" value="$(arg paused)"/>
    <arg name="gui" value="$(arg gui)"/>
  </include>

  <!-- send robot urdf to param server -->
  <include file="$(find ur_description)/launch/ur10_upload.launch">
    <arg name="limited" value="$(arg limited)"/>
  </include>

  <!-- push robot_description to factory and spawn robot in gazebo -->
  <node name="spawn_gazebo_model" pkg="gazebo_ros" type="spawn_model" args="-urdf -param robot_description -model robot -z 3.0" respawn="false" output="screen" />

  <include file="$(find ur_gazebo)/launch/controller_utils.launch"/>

  <rosparam file="$(find ur_gazebo)/controller/arm_controller_ur10.yaml" command="load"/>
  <node name="arm_controller_spawner" pkg="controller_manager" type="controller_manager" args="spawn arm_controller_ur0 arm_controller_ur1" respawn="false" output="screen"/>
  <!-- <node name="arm_controller_spawner" pkg="controller_manager" type="controller_manager" args="spawn arm_controller" respawn="false" output="screen"/> -->

</launch>
```

You are almost done here, just run:
```bash
roslaunch ur_gazebo  ur10.launch 
```

## Modify GPS Agent for Running Multiple UR Agent Nodes

You need to modify the `JOINT_PUBLISHER`, `JOINT_SUBSCRIBER`, `JOINT_ORDER`, `LINK_NAMES`, and ros node name.

The following sample `hyperparams.py` and `agent_ur.py` are just for your reference.

Make sure that **`ROS_NODE_SUFFIX` are different** for each experiment if you are running serveral `gps_main.py` with UR agents

### Example of `hyperparams.py`
```python
# To get started, copy over hyperparams from another experiment.
# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.
""" Hyperparameters for UR trajectory optimization experiment. """
from __future__ import division
import numpy as np
import os.path
import rospy
import rospkg
from datetime import datetime
from tf import TransformListener 

from gps import __file__ as gps_filepath
from gps.agent.ur_ros.agent_ur import AgentURROS
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, JOINT_SPACE
from gps.utility.general_utils import get_ee_points, get_position
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.gui.config import generate_experiment_info

# Topics for the robot publisher and subscriber.
JOINT_PUBLISHER = '/arm_controller/command'
JOINT_SUBSCRIBER = '/arm_controller/state'

# 'SLOWNESS' is how far in the future (in seconds) position control extrapolates
# when it publishs actions for robot movement.  1.0-10.0 is fine for simulation.
SLOWNESS = 1.0
# 'RESET_SLOWNESS' is how long (in seconds) we tell the robot to take when
# returning to its start configuration.
RESET_SLOWNESS = 2.0

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
INITIAL_JOINTS = [0, -np.pi / 2, np.pi / 2, 0, 0, 0]

# Set the number of goal points. 1 by default for a single end effector tip.
NUM_EE_POINTS = 1
EE_POINTS = np.array([[0, 0, 0]])

# Specify a goal state in cartesian coordinates.
EE_POS_TGT = np.asmatrix([.70, .70, .50])
"""UR 10 Examples:
EE_POS_TGT = np.asmatrix([.29, .52, .62]) # Target where all joints are 0.
EE_POS_TGT = np.asmatrix([.65, .80, .30]) # Target in positive octant near ground.
EE_POS_TGT = np.asmatrix([.70, .70, .50]) # Target in positive octant used for debugging convergence.
The Gazebo sim converges to the above point with non-action costs:
(-589.75, -594.71, -599.54, -601.54, -602.75, -603.28, -604.28, -604.79, -605.55, -606.29)
Distance from Goal: (0.014, 0.005, -0.017)
"""

# Set to identity unless you want the goal to have a certain orientation.
EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Only edit these when editing the robot joints and links.
# The lengths of these arrays define numerous parameters in GPS.
JOINT_ORDER = [SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT,
               WRIST_1_JOINT, WRIST_2_JOINT, WRIST_3_JOINT]
LINK_NAMES = [BASE, BASE_LINK, SHOULDER_LINK, UPPER_ARM_LINK, FOREARM_LINK,
              WRIST_1_LINK, WRIST_2_LINK, WRIST_3_LINK, EE_LINK]

UR_PREFIX = 'ur0'
if UR_PREFIX:
    for i, item in enumerate(JOINT_ORDER):
        JOINT_ORDER[i] = UR_PREFIX +'_' + item

    for i, item in enumerate(LINK_NAMES):
        LINK_NAMES[i] = UR_PREFIX +'_' + item

    insert_idx = JOINT_PUBLISHER.index('arm_controller')+len('arm_controller')
    JOINT_PUBLISHER = JOINT_PUBLISHER[:insert_idx] + '_' + UR_PREFIX + JOINT_PUBLISHER[insert_idx:]
    insert_idx = JOINT_SUBSCRIBER.index('arm_controller')+len('arm_controller')
    JOINT_SUBSCRIBER = JOINT_SUBSCRIBER[:insert_idx] + '_' + UR_PREFIX + JOINT_SUBSCRIBER[insert_idx:]

# Hyperparamters to be tuned for optimizing policy learning on the specific robot.
UR_GAINS = np.array([1, 1, 1, 1, 1, 1])

# Packaging sensor dimensional data for reference.
SENSOR_DIMS = {
    JOINT_ANGLES: len(JOINT_ORDER),
    JOINT_VELOCITIES: len(JOINT_ORDER),
    END_EFFECTOR_POINTS: NUM_EE_POINTS * EE_POINTS.shape[1],
    END_EFFECTOR_POINT_VELOCITIES: NUM_EE_POINTS * EE_POINTS.shape[1],
    ACTION: len(UR_GAINS),
}

# States to check in agent._process_observations.
STATE_TYPES = {'positions': JOINT_ANGLES,
               'velocities': JOINT_VELOCITIES}


# Path to urdf of robot.
rospack = rospkg.RosPack()
TREE_PATH = rospack.get_path('ur_description') + '/urdf/ur10_robot.urdf'
ROS_NODE_SUFFIX = UR_PREFIX
# Be sure to specify the correct experiment directory to save policy data at.
EXP_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'

# Set the number of seconds per step of a sample.
TIMESTEP = 0.01  # Typically 0.01.
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

    # Need for this node will go away upon migration to KDL
    rospy.init_node('gps_agent_ur_ros_node' + '_' + ROS_NODE_SUFFIX)

    # Set starting end effector position using TF
    tf = TransformListener()

    # Sleep for .1 secs to give the node a chance to kick off
    rospy.sleep(1)
    time = tf.getLatestCommonTime(LINK_NAMES[-1], LINK_NAMES[0])

    x0[joint_dim:(joint_dim + NUM_EE_POINTS * EE_POINTS.shape[1])] = get_position(tf, LINK_NAMES[-1], LINK_NAMES[0], time)

    # Initialize target end effector position
    ee_tgt = np.ndarray.flatten(
        get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T
    )

    reset_condition = {
        JOINT_ANGLES: INITIAL_JOINTS,
        JOINT_VELOCITIES: []
    }

    x0s.append(x0)
    ee_tgts.append(ee_tgt)
    reset_conditions.append(reset_condition)
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
    'joint_order': JOINT_ORDER,
    'link_names': LINK_NAMES,
    'state_types': STATE_TYPES,
    'tree_path': TREE_PATH,
    'joint_publisher': JOINT_PUBLISHER,
    'joint_subscriber': JOINT_SUBSCRIBER,
    'slowness': SLOWNESS,
    'reset_slowness': RESET_SLOWNESS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES,
                      END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'end_effector_points': [EE_POINTS],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'node_suffix': ROS_NODE_SUFFIX
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
    'init_gains': 1.0 / UR_GAINS,
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
    'target_end_effector': [np.zeros(NUM_EE_POINTS * EE_POINTS.shape[1])],
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 0.0001,
    'ramp_option': RAMP_LINEAR,
}

# This cost function takes into account the distance between the end effector's
# current and target positions at time T-1 only.
fk_cost_final = {
    'type': CostFK,
    'target_end_effector': np.zeros(NUM_EE_POINTS * EE_POINTS.shape[1]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 1.0,
    'l2': 0.0,
    'wp_final_multiplier': 100.0,  # Weight multiplier on final timestep.
    'ramp_option': RAMP_FINAL_ONLY,
}

# Combines the cost functions in 'costs' to produce a single cost function
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
    'iterations': 6000,
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
```

### Example of `agent_ur.py`
```python
import os
import copy # Only used to copy the agent config data.
import numpy as np # Used pretty much everywhere.
import threading # Used for time locks to synchronize position data.
import rospy # Needed for nodes, rate, sleep, publish, and subscribe.

from gps.agent.agent import Agent # GPS class needed to inherit from.
from gps.agent.agent_utils import setup, generate_noise # setup used to get hyperparams in init and generate_noise to get noise in sample.
from gps.agent.config import AGENT_UR_ROS # Parameters needed for config in __init__.
from gps.sample.sample import Sample # Used to build a Sample object for each sample taken.
from gps.utility.general_utils import get_position # For getting points and velocities.

from tf import TransformListener # Needed for listening to and transforming robot state information.
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing UR joint angles.
from control_msgs.msg import JointTrajectoryControllerState # Used for subscribing to the UR.
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, ACTION, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_JACOBIANS, END_EFFECTOR_POINT_VELOCITIES
from tree_urdf import treeFromFile # For KDL Jacobians
from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians

class MSG_INVALID_JOINT_NAMES_DIFFER(Exception):
    """Error object exclusively raised by _process_observations."""
    pass

class ROBOT_MADE_CONTACT_WITH_GAZEBO_GROUND_SO_RESTART_ROSLAUNCH(Exception):
    """Error object exclusively raised by reset."""
    pass

class AgentURROS(Agent):
    """Connects the UR actions and GPS algorithms."""

    def __init__(self, hyperparams, init_node=True):
        """Initialized Agent.
        hyperparams: Dictionary of hyperparameters.
        init_node:   Whether or not to initialize a new ROS node."""


        # Pull parameters from hyperparams file.
        config = copy.deepcopy(AGENT_UR_ROS)
        config.update(hyperparams)
        Agent.__init__(self, config)
        conditions = self._hyperparams['conditions']

        # Setup the main node.
        if init_node:
            rospy.init_node('gps_agent_ur_ros_node' + '_' + self._hyperparams['node_suffix'])

        # for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
        #     self._hyperparams[field] = setup(self._hyperparams[field],
        #                                      conditions)
        self._hyperparams['reset_conditions'] = setup(self._hyperparams['reset_conditions'], conditions)
        # x0 is not used by the agent, but it needs to be set on the agent
        # object because the GPS algorithm implementation gets the value it
        # uses for x0 from the agent object.  (This should probably be changed.)
        self.x0 = self._hyperparams['x0']

        # Used by subscriber to get the present time and by main thread to
        # get buffered positions.
        self.tf = TransformListener()

        # Times when EE positions were most recently recorded.  Set by the
        # subscriber thread when it collects the EE positions from ROS and
        # stores them for later use on the main thread.
        self._ee_time = None
        self._previous_ee_time = None
        self._ref_time = None

        # Lock used to make sure the subscriber does not update the EE times
        # while they are being used by the main thread.
        # The subscriber thread acquires this lock with blocking=False to
        # skip an update of the protected data if the lock is held elsewhere
        # (ie, the main thread).  The main thread acquires this lock with
        # blocking=True to block until the lock has been acquired.
        self._time_lock = threading.Lock()

        # Flag denoting if observations read in from ROS messages are fresh
        # or have already been used in processing.  Used to make sure that the
        # observations have been updated by the subscriber before the main
        # thread will try to use them.
        self._observations_stale = True

        # Message sent via the joint_subscriber topic indicating the
        # newly observed state of the UR.
        self._observation_msg = None

        # set dimensions of action
        self.dU = self._hyperparams['dU']

        self._valid_joint_set = set(hyperparams['joint_order'])
        self._valid_joint_index = {joint:index for joint, index in
                                   enumerate(hyperparams['joint_order'])}

        # Initialize the publisher and subscriber.  The subscriber begins
        # checking the time every 40 msec by default.
        self._pub = rospy.Publisher(self._hyperparams['joint_publisher'], 
                                     JointTrajectory, queue_size=5)
        self._sub = rospy.Subscriber(self._hyperparams['joint_subscriber'],
                                     JointTrajectoryControllerState,
                                     self._observation_callback)

        # Used for enforcing the period specified in hyperparameters.
        self.period = self._hyperparams['dt']
        self.r = rospy.Rate(1. / self.period)
        self.r.sleep()

        self._currently_resetting = False
        self._reset_cv = threading.Condition(self._time_lock)

    def _observation_callback(self, message):
        """This callback is set on the subscriber node in self.__init__().
        It's called by ROS every 40 ms while the subscriber is listening.
        Primarily updates the present and latest times.
        This callback is invoked asynchronously, so is effectively a
        "subscriber thread", separate from the control flow of the rest of
        GPS, which runs in the "main thread".
        message: observation from the robot to store each listen."""

        with self._time_lock:
            # Gets the most recent time according to the TF node when a TF message
            # transforming between base and end effector frames is published.
            new_time = self.tf.getLatestCommonTime(self._hyperparams['link_names'][-1],
                self._hyperparams['link_names'][0])

            # Skip updating when _observation_callback() is called multiple times.
            # between TF updates.
            if new_time == self._ee_time:
                return

            # Set the reference time if it has not yet been set.
            if self._ref_time is None:
                self._ref_time = new_time
                return

            # Set previous_ee_time if not set or same as present time.
            if self._previous_ee_time is None or self._previous_ee_time == self._ee_time:
                self._previous_ee_time = self._ee_time
                self._ee_time = new_time
                return

            # Compute the amount of time that has passed between the last
            # "present" time and now.
            delta_t = new_time - self._ref_time
            delta_ref_t = float(delta_t.secs) + float(delta_t.nsecs) * 1e-9
            delta_t = new_time - self._ee_time

            # Fields such as self._observation_msg keep track of all
            # valid messages received over the wire regardless of whether
            # "enough time has passed".  This is useful in cases such as
            # when you would like to examine the immediately prior value
            # to approximate velocity.  If these values change faster than
            # those such as self._ref_time, there will be no ill effect
            # as self._observations_stale will indicate when the latter
            # are ready to be used.
            self.delta_t = float(delta_t.secs) + float(delta_t.nsecs) * 10.**-9
            self._observation_msg = message
            self._previous_ee_time = self._ee_time
            self._ee_time = new_time

            if self._currently_resetting:
                epsilon = 1e-3
                reset_action = np.asarray(
                    self._hyperparams['reset_conditions'][0][JOINT_ANGLES])
                now_action = np.asarray(
                    self._observation_msg.actual.positions[:len(reset_action)])
                du = np.linalg.norm(reset_action-now_action, float('inf'))
                if du < epsilon:
                    self._currently_resetting = False
                    self._reset_cv.notify_all()

            # Check if not enough time has passed.
            if delta_ref_t < self.period:
                return

            # Set observations stale and release lock.
            self._ref_time = new_time
            self._observations_stale = False

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """This is the main method run when the Agent object is called by GPS.
        Draws a sample from the environment, using the specified policy and
        under the specified condition.
        If "save" is True, then append the sample object of type Sample to
        self._samples[condition].
        TensorFlow is not yet implemented (FIXME)."""

        # Reset the arm to initial configuration at start of each new trial.
        self.reset(condition)

        # Generate noise to be used in the policy object to compute next state.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Execute the trial.
        sample_data = self._run_trial(policy, noise,
                          time_to_run=self._hyperparams['trial_timeout'])

        # Write trial data into sample object.
        sample = Sample(self)
        for sensor_id, data in sample_data.iteritems():
            sample.set(sensor_id, np.asarray(data))

        # Save the sample to the data structure. This is controlled by gps_main.py.
        if save:
            self._samples[condition].append(sample)

        return sample

    def reset(self, condition):
        """Not necessarily a helper function as it is inherited.
        Reset the agent for a particular experiment condition.
        condition: An index into hyperparams['reset_conditions']."""

        # Set the reset position as the initial position from agent hyperparams.
        action = self._hyperparams['reset_conditions'][condition][JOINT_ANGLES]

        # Prepare the present positions to see how far off we are.
        now_position = np.asarray(self._observation_msg.actual.positions[:len(action)])

        # Raise error if robot has made contact with the ground in simulation.
        # This occurs because Gazebo sets joint angles beyond what they can possibly
        # be when the robot makes contact with the ground and "breaks." 
        if max(abs(now_position)) >= 2*np.pi:
            raise ROBOT_MADE_CONTACT_WITH_GAZEBO_GROUND_SO_RESTART_ROSLAUNCH

        # Wait until the arm is within epsilon of reset configuration.
        with self._time_lock:
            self._currently_resetting = True
            self._pub.publish(self._get_ur_trajectory_message(action,
                              self._hyperparams['reset_slowness']))
            self._reset_cv.wait()

    def _run_trial(self, policy, noise, time_to_run=5):
        """'Private' method only called by sample() to collect sample data.
        Runs an async controller with a policy.
        The async controller receives observations from ROS subscribers and
        then uses them to publish actions.
        policy:      policy object used to get next state
        noise:       noise necessary in order to carry out the policy
        time_to_run: is not used in this agent

        Returns:
            result: a dictionary keyed with each of the constants that
            appear in the state_include, obs_include, and meta_include
            sections of the hyperparameters file.  Each of these should
            be associated with an array indexed by the timestep at which
            a certain state/observation/meta param occurred and the
            value representing a particular state/observation/meta
            param.  Through this indexing scheme the value of each
            state/observation/meta param at each timestep is stored."""

        # Initialize the data structure to be passed to GPS.
        result = {param: [] for param in self.x_data_types +
                         self.obs_data_types + self.meta_data_types +
                         [END_EFFECTOR_POINT_JACOBIANS, ACTION]}

        # Carry out the number of trials specified in the hyperparams.  The
        # index is only called by the policy.act method.  We use a while
        # instead of for because we do not want to iterate if we do not publish.
        time_step = 0
        while time_step < self._hyperparams['T']:

            # Skip the action if the subscriber thread has not finished
            # initializing the times.
            if self._previous_ee_time is None:
                continue

            # Only read and process ROS messages if they are fresh.
            if self._observations_stale is False:

                # Acquire the lock to prevent the subscriber thread from
                # updating times or observation messages.
                self._time_lock.acquire(True)

                prev_ros_time = self._previous_ee_time
                ros_time = self._ee_time
                self._previous_ee_time = ros_time
                obs_message = self._observation_msg

                # Make it so that subscriber's thread observation callback
                # must be called before publishing again.  FIXME: This field
                # should be protected by the time lock.
                self._observations_stale = True

                # Release the lock after all dynamic variables have been updated.
                self._time_lock.release()

                # Collect the end effector points and velocities in
                # cartesian coordinates for the state.
                ee_points, ee_velocities = \
                    self._get_points_and_vels(ros_time, prev_ros_time, 
                        self._hyperparams['link_names'][-1],
                        self._hyperparams['link_names'][0], debug=False)

                # Collect the present joint angles and velocities from ROS for the state.
                last_observations = self._process_observations(obs_message, result)

                # Get Jacobians from present joint angles and KDL trees
                # The Jacobians consist of a 6x6 matrix getting its from from
                # (# joint angles) x (len[x, y, z] + len[roll, pitch, yaw])
                ee_jacobians = self._get_jacobians(last_observations[:6])

                # Concatenate the information that defines the robot state
                # vector, typically denoted as 'x' in GPS articles.
                state = np.r_[np.reshape(last_observations, -1),
                              np.reshape(ee_points, -1),
                              np.reshape(ee_velocities, -1)]

                # Pass the state vector through the policy to get a vector of
                # joint angles to go to next.
                action = policy.act(state, state, time_step, noise[time_step])

                # Primary print statements of the action development.
                print '\nTimestep', time_step
                print 'Joint States ', np.around(state[:6], 2)
                print 'Policy Action', np.around(action, 2)

                # Display meters off from goal.
                print 'Distance from Goal', np.around(ee_points, 3)

                # Stop the robot from moving past last position when sample is done.
                # If this is not called, the robot will complete its last action even
                # though it is no longer supposed to be exploring.
                if time_step == self._hyperparams['T']-1:
                    action[:6] = last_observations[:6]

                # Publish the action to the robot.
                self._pub.publish(self._get_ur_trajectory_message(action, 
                    self._hyperparams['slowness']))

                # Only update the time_step after publishing.
                time_step += 1

                # Build up the result data structure to return to GPS.
                result[ACTION].append(action)
                result[END_EFFECTOR_POINTS].append(ee_points)
                result[END_EFFECTOR_POINT_JACOBIANS].append(ee_jacobians)
                result[END_EFFECTOR_POINT_VELOCITIES].append(ee_velocities)

            # The subscriber is listening during this sleep() call, and
            # updating the time "continuously" (each hyperparams['period'].
            self.r.sleep()

        # Sanity check the results to make sure nothing is infinite.
        for value in result.values():
            if not np.isfinite(value).all():
                print 'There is an infinite value in the results.'
            assert np.isfinite(value).all()
        return result

    def _get_jacobians(self,state):
        """Produce a Jacobian from the urdf that maps from joint angles to x, y, z.
        This makes a 6x6 matrix from 6 joint angles to x, y, z and 3 angles.
        The angles are roll, pitch, and yaw (not Euler angles) and are not needed.
        Returns a repackaged Jacobian that is 3x6.
        """

        # Initialize a Jacobian for 6 joint angles by 3 cartesian coords and 3 orientation angles
        jacobian = Jacobian(6)

        # Initialize a joint array for the present 6 joint angles.
        angles = JntArray(6)

        # Construct the joint array from the most recent joint angles.
        for i in range(6):
            angles[i] = state[i]

        # Initialize a tree structure from the robot urdf. 
        # Note that the xacro of the urdf is updated by hand.
        # Then the urdf must be compiled.
        _, ur_tree = treeFromFile(self._hyperparams['tree_path'])

        # Retrieve a chain structure between the base and the start of the end effector.
        ur_chain = ur_tree.getChain(self._hyperparams['link_names'][0],
            self._hyperparams['link_names'][-1])

        # Initialize a KDL Jacobian solver from the chain.
        jac_solver = ChainJntToJacSolver(ur_chain)

        # Update the jacobian by solving for the given angles.
        jac_solver.JntToJac(angles, jacobian)

        # Initialize a numpy array to store the Jacobian.
        J = np.zeros((6,6))
        for i in range(jacobian.rows()):
            for j in range(jacobian.columns()):
                J[i,j] = jacobian[i,j]

        # Only want the cartesian position, not Roll, Pitch, Yaw (RPY) Angles
        ee_jacobians = J[0:3,:] 

        return ee_jacobians

    def _get_points_and_vels(self, t, t_last, target, source, debug=False):
        """
        Helper function to _run_trial that gets the cartesian positions
        and velocities from ROS.  Uses tf to convert from the Base to
        End Effector and and get cartesian coordinates.  Draws positions
        from a 'buffer' of past positions by checking recent times.
        t: the 'present' time.  This is the time of the most recent observation,
            so slightly earlier than the actual present.
        t_last:          the 'present' time from last time this function was called
        target:          the End Effector link
        source:          the Base Link
        debug:           activate extra print statements for debugging"""

        # Assert that time has passed since the last step.
        assert self.delta_t > 0.

        # Listen to the cartesian coordinate of the target link.
        pos_last = get_position(self.tf, target, source, t_last)
        pos_now = get_position(self.tf, target, source, t)

        # Use the past position to get the present velocity.
        velocity = (pos_now - pos_last)/self.delta_t

        # Shift the present position by the End Effector target.
        # Since we subtract the target point from the current position, the optimal
        # value for this will be 0.
        position = np.asarray(pos_now)-self._hyperparams['ee_points_tgt'][0, :]

        if debug:
            print 'VELOCITY:', velocity
            print 'POSITION:', position
            print 'BEFORE  :', t_last.to_sec(), pos_last
            print 'PRESENT :', t.to_sec(), pos_now

        return position, velocity

    def _process_observations(self, message, result):
        """Helper fuinction only called by _run_trial to convert a ROS message
        to joint angles and velocities.
        Check for and handle the case where a message is either malformed
        or contains joint values in an order different from that expected
        in hyperparams['joint_order']"""


        # Check if joint values are in the expected order and size.
        if message.joint_names != self._hyperparams['joint_order']:

            # Check that the message is of same size as the expected message.
            if len(message.joint_names) != len(self._hyperparams['joint_order']):
                raise MSG_INVALID_JOINT_NAMES_DIFFER

            # Check that all the expected joint values are present in a message.
            if not all(map(lambda x,y: x in y, message.joint_names, 
                [self._valid_joint_set for _ in range(len(message.joint_names))])):

                raise MSG_INVALID_JOINT_NAMES_DIFFER

            # If necessary, reorder the joint values to conform to the order
            # expected in hyperparams['joint_order'].
            new_message = [None for _ in range(len(message))]
            for joint, index in message.joint_names.enumerate():
                for state_type in self._hyperparams['state_types']:
                    new_message[self._valid_joint_index[joint]] = message[state_type][index]

            message = new_message

        # Package the positions, velocities, amd accellerations of the joint angles.
        for (state_type, state_category), state_value_vector in zip(
            self._hyperparams['state_types'].iteritems(),
            [message.actual.positions, message.actual.velocities,
            message.actual.accelerations]):

            # Assert that the length of the value vector matches the corresponding
            # number of dimensions from the hyperparameters file
            assert len(state_value_vector) == self._hyperparams['sensor_dims'][state_category]

            # Write the state value vector into the results dictionary keyed by its
            # state category
            result[state_category].append(state_value_vector)

        return np.array(result[JOINT_ANGLES][-1] + result[JOINT_VELOCITIES][-1])

    def _get_ur_trajectory_message(self, action, slowness):
        """Helper function only called by reset() and run_trial().
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion"""

        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names = self._hyperparams['joint_order']

        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        target.positions = action

        # These times determine the speed at which the robot moves:
        # it tries to reach the specified target position in 'slowness' time.
        target.time_from_start = rospy.Duration(slowness)

        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]

        return action_msg
```
