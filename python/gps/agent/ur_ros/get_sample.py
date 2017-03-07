import rospy
import moveit_commander
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from sensor_msgs.msg import JointState
import scipy.interpolate as itp
import numpy as np
import geometry_msgs.msg
import math
def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
def get_rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> np.allclose(np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2, np.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                   [ direction[2], 0.0,          -direction[0]],
                   [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)

    # exchange (w, x, y, z) to (x, y, z, w)
    q_new = np.empty(4)
    q_new[:3] = q[1:]
    q_new[3] = q[0]
    return q_new

class Sample_from_moveit:

    def __init__(self):
        print "============ Starting tutorial setup"
        # self._robot = moveit_commander.RobotCommander()
        # self._scene = moveit_commander.PlanningSceneInterface()
        self._group = moveit_commander.MoveGroupCommander("manipulator")
        self._group.set_planner_id("RRTkConfigDefault")
        self._group.set_num_planning_attempts(3)
        self._group.set_goal_position_tolerance(0.001)
        self._group.set_goal_orientation_tolerance(0.001)
        self._group.set_max_velocity_scaling_factor(1)
        self.rate = 20
        self.speed = 0.05
        self._rate = rospy.Rate(self.rate)
        self.n_a_n = []
        self._pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
        rospy.sleep(2)
        self.__action_msg = JointTrajectory()
        self.__action_msg.joint_names = ('shoulder_pan_joint',
                                         'shoulder_lift_joint',
                                         'elbow_joint',
                                         'wrist_1_joint',
                                         'wrist_2_joint',
                                         'wrist_3_joint')
    def move_joints(self, joints=None):
        joints = JointState()
        group_variable_values = self._group.get_current_joint_values()
        print "============ Joint values: ", group_variable_values
        # self._group.clear_pose_targets()

        joints.position = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])

        self._group.set_joint_value_target([0, -np.pi / 2, np.pi / 2, 0, np.pi, 0])

        plan3 = self._group.plan()
        self._group.execute(plan3)
        group_variable_values = self._group.get_current_joint_values()
        print "============ Joint values: ", group_variable_values
        return plan3

    def get_samples(self, plan3):
        """
        get gps samples from a plan obtained from moveit
        """
        n_t = []
        n_a = []
        count = len(plan3.joint_trajectory.points)
        print count

        for i in plan3.joint_trajectory.points:
            n_t.append(float(i.time_from_start.secs) + float(i.time_from_start.nsecs) / 1.0e9)
            n_a.append(i.positions)
            print n_t[-1], "|", n_a[-1]

        n_step_count = int(n_t[-1]*self.rate)

        if n_step_count > 100:
            print ("time error")
        n_t_n = np.linspace(0, n_t[-1], n_step_count)
        self.n_a_n = list(itp.spline(n_t, n_a, n_t_n))

        e_n_a_n = []
        for i in range(0, 100-n_step_count):
            e_n_a_n.append(self.n_a_n[-1])
        self.n_a_n.extend(e_n_a_n)
        # self._group = []
        return self.n_a_n

    def test_samples(self):
        for i in range(0, 100):
            print "step:", i
            target = JointTrajectoryPoint()
            target.positions = self.n_a_n[i]
            target.time_from_start = rospy.Duration.from_sec(self.speed)
            self.__action_msg.points = [target]
            self._pub.publish(self.__action_msg)
            self._rate.sleep()

    def move_pose(self):
        """
        pose : position and orientation which can be obtained from tf shown in rviz
        orientation.w^2 + pose_target.orientation.x^2 + pose_target.orientation.y^2 + pose_target.orientation.z^2 = 1
        """
        pose_target = geometry_msgs.msg.Pose()
        trans = np.array([-0.593025, -0.57496, 0.44333])
        rotation_matrix = get_rotation_matrix(-np.pi / 1.5, [0, 1, 0.])
        # rotation_matrix = np.empty((4, 4))
        # rotation_matrix[:3, :3] = np.array([[1, 0, 0],
        #                                     [0, 1, 0],
        #                                     [0, 0, 1]])
        rotation_matrix[:3, 3] = trans
        # q = [0.7136, -0.06977, -0.02925, 0.69637]
        q = quaternion_from_matrix(rotation_matrix)
        pose_target.orientation.x = q[0]
        pose_target.orientation.y = q[1]
        pose_target.orientation.z = q[2]
        pose_target.orientation.w = q[3]

        pose_target.position.x = trans[0]
        pose_target.position.y = trans[1]
        pose_target.position.z = trans[2]
        self._group.set_pose_target(pose_target)
        plan3 = self._group.plan()
        self._group.execute(plan3)
        return plan3

    def reset(self):
        """
        reset to  joint position [0,0,0,0,0,0]
        """
        self._group.set_named_target('home')
        plan3 = self._group.plan()
        self._group.execute(plan3)
        return plan3


if __name__ == "__main__":
    rospy.init_node("zz")
    rospy.sleep(1)
    sample_from_moveit = Sample_from_moveit()
    # moveit_target = raw_input("moveit_target")
    # print moveit_target
    # print sample_from_moveit.get_samples(moveit_target)
    # t = raw_input("test sample:")
    # sample_from_moveit.test_samples()
    sample_from_moveit.reset()
    # sample_from_moveit.move_joints()
    plan = sample_from_moveit.move_pose()
    # sample_from_moveit.move_joints()
    # sample_from_moveit.get_samples(plan3=plan)
    rospy.spin()

