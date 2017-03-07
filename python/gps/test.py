# !/usr/bin/env python
import rospy
import rospkg
import numpy as np
import scipy.interpolate as itp
from timeit import default_timer as timer
import time
import moveit_commander
import PyKDL as kdl
from tf import TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from agent.ur_ros.tree_urdf import treeFromFile
from utility.general_utils import get_position
from gazebo_msgs.srv import GetLinkState
SHOULDER_PAN_JOINT = 'shoulder_pan_joint'
SHOULDER_LIFT_JOINT = 'shoulder_lift_joint'
ELBOW_JOINT = 'elbow_joint'
WRIST_1_JOINT = 'wrist_1_joint'
WRIST_2_JOINT = 'wrist_2_joint'
WRIST_3_JOINT = 'wrist_3_joint'
JOINT_ORDER = [SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT,
               WRIST_1_JOINT, WRIST_2_JOINT, WRIST_3_JOINT]

def get_ur_trajectory_message(action, slowness):
    """Helper function only called by reset() and run_trial().
    Wraps an action vector of joint angles into a JointTrajectory message.
    The velocities, accelerations, and effort do not control the arm motion"""

    # Set up a trajectory message to publish.
    action_msg = JointTrajectory()
    action_msg.joint_names = JOINT_ORDER

    # Create a point to tell the robot to move to.
    target = JointTrajectoryPoint()
    target.positions = action

    # These times determine the speed at which the robot moves:
    # it tries to reach the specified target position in 'slowness' time.
    target.time_from_start = rospy.Duration(slowness)

    # Package the single point into a trajectory of points with length 1.
    action_msg.points = [target]

    return action_msg

## Use PyKDL to perform inverse kinematics
## Not good, often cannot get solution
def joint_list_to_kdl(q):
    if q is None:
        return None
    if type(q) == np.matrix and q.shape[1] == 0:
        q = q.T.tolist()[0]
    q_kdl = kdl.JntArray(len(q))
    for i, q_i in enumerate(q):
        q_kdl[i] = q_i
    return q_kdl

def joint_kdl_to_list(q):
    if q == None:
        return None
    return [q[i] for i in range(q.rows())]

def inverse_kinematics(chain, pos, rot, q_guess=None, min_joints=None, max_joints=None):
    pos_kdl = kdl.Vector(pos[0], pos[1], pos[2])
    rot_kdl = kdl.Rotation(rot[0, 0], rot[0, 1], rot[0, 2],
                           rot[1, 0], rot[1, 1], rot[1, 2],
                           rot[2, 0], rot[2, 1], rot[2, 2])
    frame_kdl = kdl.Frame(rot_kdl, pos_kdl)
    if min_joints is None:
        min_joints = -np.pi * np.ones(6)
    if max_joints is None:
        max_joints = np.pi * np.ones(6)
    min_joints[-1] = -2 * np.pi
    max_joints[-1] = 2 * np.pi
    mins_kdl = joint_list_to_kdl(min_joints)
    maxs_kdl = joint_list_to_kdl(max_joints)
    fk_kdl = kdl.ChainFkSolverPos_recursive(chain)
    ik_v_kdl = kdl.ChainIkSolverVel_pinv(chain)
    ik_p_kdl = kdl.ChainIkSolverPos_NR_JL(chain, mins_kdl, maxs_kdl,
                                          fk_kdl, ik_v_kdl)

    if q_guess == None:
        # use the midpoint of the joint limits as the guess
        lower_lim = np.where(np.isfinite(min_joints), min_joints, 0.)
        upper_lim = np.where(np.isfinite(max_joints), max_joints, 0.)
        q_guess = (lower_lim + upper_lim) / 2.0
        q_guess = np.where(np.isnan(q_guess), [0.]*len(q_guess), q_guess)

    q_kdl = kdl.JntArray(6)
    q_guess_kdl = joint_list_to_kdl(q_guess)
    if ik_p_kdl.CartToJnt(q_guess_kdl, frame_kdl, q_kdl) >= 0:
        return joint_kdl_to_list(q_kdl)
    else:
        return None


def get_ee_link_pos(tf, target, source):
    target = 'ee_link'
    source = 'base'
    new_time = tf.getLatestCommonTime(target, source)
    pos_now = get_position(tf, target, source, new_time)
    return pos_now
    # ee_link_pos = np.zeros(3)
    # ee_link_ori = np.zeros(4)
    # rospy.wait_for_message('/gazebo/get_link_state')
    # try:
    #     link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
    #     resp = link_state('ackermann_vehicle::base_link', '')
    #     ee_link_pos[0] = resp.link_state.pose.position.x
    #     ee_link_pos[1] = resp.link_state.pose.position.y
    #     ee_link_pos[2] = resp.link_state.pose.position.z
    #     ee_link_ori[0] = resp.link_state.pose.orientation.x
    #     ee_link_ori[1] = resp.link_state.pose.orientation.y
    #     ee_link_ori[2] = resp.link_state.pose.orientation.z
    #     ee_link_ori[3] = resp.link_state.pose.orientation.w
    # except rospy.ServiceException, e:
    #     print "Service call failed: %s" % e

class Sample_from_moveit:

    def __init__(self):
        self._robot = moveit_commander.RobotCommander()
        self._scene = moveit_commander.PlanningSceneInterface()
        self._group = moveit_commander.MoveGroupCommander("manipulator")
        self._group.set_planner_id("RRTkConfigDefault")
        self._group.set_num_planning_attempts(3)
        self._group.set_goal_position_tolerance(0.005)
        self._group.set_max_velocity_scaling_factor(1)
        self.rate = 20
        self.speed = 0.05
        rospy.Rate(self.rate)
        self.n_a_n = [0, 0, 0, 0, 0, 0]
        self._pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
        rospy.sleep(2)
        self._action_msg = JointTrajectory()
        self._action_msg.joint_names = JOINT_ORDER

    def get_samples(self, target):
        n_t = []
        n_a = []
        self._group.set_named_target(target)
        plan = self._group.plan()
        count = len(plan.joint_trajectory.points)
        print 'Points on the planned trajectory: ', count

        time_at_points = [float(point_i.time_from_start.secs)
                          + float(point_i.time_from_start.nsecs) / 1.0e9
                          for point_i in plan.joint_trajectory.points]
        action_at_points = [point_i.positions for point_i in plan.joint_trajectory.points]

        if time_at_points[-1] > 100 / float(self.rate):
            print 'The trajectory planned by MoveIt may not be reached within the GPS\'s running time'
        #
        # if n_step_count > 100:
        #     print ("time error")
        # n_t_n = np.linspace(0, n_t[-1], n_step_count)
        # self.n_a_n = list(itp.spline(n_t, n_a, n_t_n))
        #
        # e_n_a_n = []
        # for i in range(0, 100-n_step_count):
        #     e_n_a_n.append(self.n_a_n[-1])
        # self.n_a_n.extend(e_n_a_n)
        # # self._group = []
        # return self.n_a_n

    def test_samples(self):
        for i in range(0, 100):
            print "step:", i
            target = JointTrajectoryPoint()
            target.positions = self.n_a_n[i]
            target.time_from_start = rospy.Duration.from_sec(self.speed)
            self._action_msg.points = [target]
            self._pub.publish(self._action_msg)
            self._rate.sleep()

    def move_xyz(self, xyz=[-0.70, -0.70, 0.50]):
        print self._group.get_end_effector_link()
        self._group.set_position_target(xyz)
        plan3 = self._group.plan()
        self._group.execute(plan3)

    def reset(self):
        target = JointTrajectoryPoint()
        target.positions = [0,   0,  0,  0,  0, 0]
        target.time_from_start = rospy.Duration.from_sec(self.speed)
        self.__action_msg.points = [target]
        self._pub.publish(self.__action_msg)

    def reset_moviet(self):
        self._group.set_named_target('home')
        plan3 = self._group.plan()
        self._group.execute(plan3)

def forward_kinematics(q, end_link=None, base_link=None):
    base_trans = do_kdl_fk(q, 0)
    if base_trans is None:
        print "FK KDL failure on base transformation."
    end_trans = do_kdl_fk(q, -1)
    if end_trans is None:
        print "FK KDL failure on end transformation."
    pos = np.dot(np.linalg.inv(base_trans), end_trans)
    return pos

def do_kdl_fk(q, link_number):
    endeffec_frame = kdl.Frame()
    fk_kdl = kdl.ChainFkSolverPos_recursive(ur_chain)
    kinematics_status = fk_kdl.JntToCart(joint_list_to_kdl(q),
                                               endeffec_frame,
                                               link_number)
    if kinematics_status >= 0:
        p = endeffec_frame.p
        M = endeffec_frame.M
        return np.array([[M[0, 0], M[0, 1], M[0, 2], p.x()],
                         [M[1, 0], M[1, 1], M[1, 2], p.y()],
                         [M[2, 0], M[2, 1], M[2, 2], p.z()],
                         [     0,      0,      0,     1]])
    else:
        return None

if __name__ == "__main__":
    rospy.init_node('ur_ros_node')
    rospack = rospkg.RosPack()
    TREE_PATH = rospack.get_path('ur_description') + '/urdf/ur10_robot.urdf'
    _, ur_tree = treeFromFile(TREE_PATH)
    # Retrieve a chain structure between the base and the start of the end effector.
    base_link = 'base'
    end_link = 'ee_link'
    ur_chain = ur_tree.getChain(base_link, end_link)

    # Use PyKDL to perform inverse kinematics
    pos = np.array([-0.30, 0.0, 0.00])
    # pos = forward_kinematics([0, -np.pi / 2, np.pi / 2, 0, 0, 0])[:3, 3].reshape(3)
    rot = np.eye(3)#forward_kinematics([0, -np.pi / 2, np.pi / 2, 0, 0, 0])[:3, :3]
    print 'pos:', pos
    print 'rot:', rot
    action = [0, -np.pi / 2, np.pi / 2, 0, np.pi / 2, 0]#inverse_kinematics(ur_chain, pos=pos, rot=rot)
    print 'action:', action
    if action is not None:
        pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=5)
        time.sleep(0.5)
        tf = TransformListener()
        pub.publish(get_ur_trajectory_message(action, 2.0))
        time.sleep(3)
        ee_link_pos = get_ee_link_pos(tf, end_link, base_link)
        print 'ee_link_pos:', ee_link_pos
