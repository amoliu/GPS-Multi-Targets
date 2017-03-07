import os
import time
import copy # Only used to copy the agent config data.
import numpy as np # Used pretty much everywhere.
import threading # Used for time locks to synchronize position data.
import rospy # Needed for nodes, rate, sleep, publish, and subscribe.
from timeit import default_timer as timer
from scipy import stats
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


        # Initialize a tree structure from the robot urdf.
        # Note that the xacro of the urdf is updated by hand.
        # Then the urdf must be compiled.

        _, self.ur_tree = treeFromFile(self._hyperparams['tree_path'])
        # Retrieve a chain structure between the base and the start of the end effector.
        self.ur_chain = self.ur_tree.getChain(self._hyperparams['link_names'][0],
            self._hyperparams['link_names'][-1])
        # Initialize a KDL Jacobian solver from the chain.
        self.jac_solver = ChainJntToJacSolver(self.ur_chain)

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
            try:
                new_time = self.tf.getLatestCommonTime(self._hyperparams['link_names'][-1],
                                                       self._hyperparams['link_names'][0])
            except:
                return

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
        sample_data = self._run_trial(policy, noise, condition=condition,
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
        # if max(abs(now_position)) >= 2*np.pi:
        #     raise ROBOT_MADE_CONTACT_WITH_GAZEBO_GROUND_SO_RESTART_ROSLAUNCH

        # Wait until the arm is within epsilon of reset configuration.
        with self._time_lock:
            self._currently_resetting = True
            self._pub.publish(self._get_ur_trajectory_message(action,
                              self._hyperparams['reset_slowness']))
            self._reset_cv.wait()

    def _run_trial(self, policy, noise, condition, time_to_run=5):
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
        publish_frequencies = []
        start = timer()
        while time_step < self._hyperparams['T']:

            # # Skip the action if the subscriber thread has not finished
            # # initializing the times.
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
                # ee_points = self._get_points(ros_time,
                #                              self._hyperparams['link_names'][-1],
                #                              self._hyperparams['link_names'][0],
                #                              condition=condition)
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
                # ee_velocities = np.dot(ee_jacobians, np.array(result[JOINT_VELOCITIES][-1]).T)
                # print '\nee_points:', ee_points
                #
                # print 'ee_jacobians:', ee_jacobians
                # print 'velocity:',np.array(result[JOINT_VELOCITIES][-1]).T
                # print 'ee_velocities:', ee_velocities
                # print 'cal:',np.dot(ee_jacobians, np.array(result[JOINT_VELOCITIES][-1]).T)
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
                end = timer()
                elapsed_time = end-start
                frequency = 1 / float(elapsed_time)
                print('Time interval(s): {0:8.4f},  Hz: {1:8.4f}'.format(elapsed_time, frequency))
                publish_frequencies.append(frequency)
                start = timer()
            # The subscriber is listening during this sleep() call, and
            # updating the time "continuously" (each hyperparams['period'].
            # self.r.sleep()

        n, min_max, mean, var, skew, kurt = stats.describe(publish_frequencies)
        median = np.median(publish_frequencies)
        first_quantile = np.percentile(publish_frequencies, 25)
        third_quantile = np.percentile(publish_frequencies, 75)
        print('\nPublisher frequencies statistics:')
        print("Minimum: {0:9.4f} Maximum: {1:9.4f}".format(min_max[0], min_max[1]))
        print("Mean: {0:9.4f}".format(mean))
        print("Variance: {0:9.4f}".format(var))
        print("Median: {0:9.4f}".format(median))
        print("First quantile: {0:9.4f}".format(first_quantile))
        print("Third quantile: {0:9.4f}".format(third_quantile))
        # Sanity check the results to make sure nothing is infinite.
        for value in result.values():
            if not np.isfinite(value).all():
                print 'There is an infinite value in the results.'
            assert np.isfinite(value).all()
        return result

    def _get_jacobians(self, state):
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

        # Update the jacobian by solving for the given angles.
        self.jac_solver.JntToJac(angles, jacobian)

        # Initialize a numpy array to store the Jacobian.
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(jacobian.rows())])

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
        pos_now = get_position(self.tf, target, source, t)
        pos_last = get_position(self.tf, target, source, t_last)

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

    def _get_points(self, t, target, source, condition):
        """
        Helper function to _run_trial that gets the cartesian positions
        from ROS.  Uses tf to convert from the Base to
        End Effector and and get cartesian coordinates.  Draws positions
        from a 'buffer' of past positions by checking recent times.
        t: the 'present' time.  This is the time of the most recent observation,
            so slightly earlier than the actual present.
        target:          the End Effector link
        source:          the Base Link"""

        # Listen to the cartesian coordinate of the target link.
        pos_now = get_position(self.tf, target, source, t)

        # Shift the present position by the End Effector target.
        # Since we subtract the target point from the current position, the optimal
        # value for this will be 0.
        position = np.asarray(pos_now)-self._hyperparams['ee_points_tgt'][condition, :]
        return position

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

