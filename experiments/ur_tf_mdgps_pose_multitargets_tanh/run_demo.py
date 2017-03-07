#!/usr/bin/env
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import imp
import scipy.ndimage as sp_ndimage
import threading
from geometry_msgs.msg import Pose
import moveit_commander
from scipy.interpolate import spline
# Add gps/python to path so that imports work.
sys.path.append(os.path.join('/'.join(str.split(os.path.realpath(__file__), '/')[:-3]), 'python'))
from gps.proto.gps_pb2 import JOINT_ANGLES

class Demo(object):
    def __init__(self, config):
        self._hyperparams = config
        self.agent = self._hyperparams['agent']
        self._group = moveit_commander.MoveGroupCommander("manipulator")
        self.moveit_velocity_scale = 0.5
        # self._group.set_planner_id("RRTkConfigDefault")
        self._group.set_planner_id("RRTConnectkConfigDefault")
        self._group.set_num_planning_attempts(3)
        self._group.set_goal_position_tolerance(0.005)
        self._group.set_goal_orientation_tolerance(0.005)
        self._group.set_max_velocity_scaling_factor(self.moveit_velocity_scale)
        self.period = self.agent['dt']
        self.num_samples = self.agent['num_samples']
        self.step_count = self.agent['T']

        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.demo_folder = os.path.join(self.current_directory, 'demo')

        if not os.path.exists(self.demo_folder):
            os.makedirs(self.demo_folder)

        self.noise_on = self.agent.get('noise_on', 'noise_on_target')  # 'noise_on_actions'
        if self.noise_on == 'noise_on_target':
            self.pos_sigma = self.agent.get('demo_pos_sigma', 0.001)
            self.quat_sigma = self.agent.get('demo_quat_sigma', 0.001)
        elif self.noise_on == 'noise_on_actions':
            self.action_noise_sigma = self.agent.get('action_noise_sigma', 0.2)
            self.action_noise_mu = self.agent.get('action_noise_mu', 0)
            self.action_smooth_noise_var = self.agent.get('action_smooth_noise_var', 2.0)
            self.demo_no_noise_folder = os.path.join(self.current_directory, 'demo_no_noise')
            if not os.path.exists(self.demo_no_noise_folder):
                os.makedirs(self.demo_no_noise_folder)
        else:
            raise Exception('Unknown noise implementing method :{0:s}, '
                            'which should have been \'noise_on_target\' or \'noise_on_actions\''.format(
                self.noise_method))

    def samples(self):
        for condition in xrange(self.agent['conditions']):
            self.generate_moveit_sample(condition)
            print 'Condition ' + str(condition) + ' Sampled'
        print 'All conditions sampled!!!'

    def generate_moveit_sample(self, condition):
        reset_joints = self.agent['reset_conditions'][condition][JOINT_ANGLES].tolist()
        tgt_pos = self.agent['ee_points_tgt'][condition, :3]
        tgt_quat = self.agent['ee_quaternion_tgt'][condition]
        tmp_tgt_pos = {}
        tmp_tgt_quaternion = {}
        tmp_moveit_action_dict = {}
        if self.noise_on == 'noise_on_target':
            for idx in xrange(self.num_samples):
                while True:
                    self._group.set_joint_value_target(reset_joints)
                    reset_plan = self._group.plan()
                    self._group.execute(reset_plan)
                    tgt_pos = np.random.normal(tgt_pos, self.pos_sigma)
                    tgt_quaternion = np.random.normal(tgt_quat, self.quat_sigma)
                    tgt_quaternion = tgt_quaternion / np.linalg.norm(tgt_quaternion)
                    actions = self.moveit_samples(tgt_pos, tgt_quaternion)
                    if actions is not None:
                        tmp_tgt_pos[idx] = tgt_pos.tolist()
                        tmp_tgt_quaternion[idx] = tgt_quaternion.tolist()
                        tmp_moveit_action_dict[idx] = actions.tolist()
                        break
        elif self.noise_on == 'noise_on_actions':
            while True:
                self._group.set_joint_value_target(reset_joints)
                reset_plan = self._group.plan()
                self._group.execute(reset_plan)
                actions = self.moveit_samples(tgt_pos, tgt_quat)
                if actions is not None:
                    actions_no_noise_filename = os.path.join(self.demo_no_noise_folder,
                                                    'actions' + str('{:04d}'.format(condition) + '.json'))
                    tmp_moveit_action_no_noise_list = actions.tolist()
                    with open(actions_no_noise_filename, 'w') as act_file:
                        json.dump(tmp_moveit_action_no_noise_list, act_file, indent=4)
                    for idx in xrange(self.num_samples):
                        tmp_tgt_pos[idx] = tgt_pos.tolist()
                        tmp_tgt_quaternion[idx] = tgt_quat.tolist()
                        noise = self.action_noise_sigma * np.random.randn(self.agent['T'], self.agent['dU']) + self.action_noise_mu
                        for i in range(self.agent['dU']):
                            noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], self.action_smooth_noise_var)
                        tmp_moveit_action_dict[idx] = (actions + noise).tolist()
                    break
        else:
            raise Exception('Unknown noise implementing method :{0:s}, '
                            'which should have been \'noise_on_target\' or \'noise_on_actions\''.format(self.noise_method))



        actions_filename = os.path.join(self.demo_folder, 'actions' + str('{:04d}'.format(condition) + '.json'))
        pos_filename = os.path.join(self.demo_folder, 'pos' + str('{:04d}'.format(condition) + '.json'))
        quat_filename = os.path.join(self.demo_folder, 'quat' + str('{:04d}'.format(condition) + '.json'))
        with open(actions_filename, 'w') as action_file:
            json.dump(tmp_moveit_action_dict, action_file, indent=4)
        with open(pos_filename, 'w') as pos_file:
            json.dump(tmp_tgt_pos, pos_file, indent=4)
        with open(quat_filename, 'w') as quat_file:
            json.dump(tmp_tgt_quaternion, quat_file, indent=4)



    def moveit_samples(self, tgt_pos, tgt_quaternion):
        print 'Moveit tgt pos:', tgt_pos
        print 'Moveit tgt quaternion', tgt_quaternion
        pose_target = Pose()
        pose_target.orientation.x = tgt_quaternion[0]
        pose_target.orientation.y = tgt_quaternion[1]
        pose_target.orientation.z = tgt_quaternion[2]
        pose_target.orientation.w = tgt_quaternion[3]

        pose_target.position.x = tgt_pos[0]
        pose_target.position.y = tgt_pos[1]
        pose_target.position.z = tgt_pos[2]
        self._group.set_pose_target(pose_target)
        moveit_plan_time_step = self.period
        maximum_try_times = 15
        try_times = 0
        while True:
            moveit_plan = self._group.plan()
            last_point = moveit_plan.joint_trajectory.points[-1]
            last_action_time = float(last_point.time_from_start.secs) +\
                               float(last_point.time_from_start.nsecs) / 1.0e9
            if last_action_time / float(moveit_plan_time_step) < self.step_count:
                break
            else:
                try_times += 1
                if try_times > maximum_try_times:
                    raise Exception('Cannot find a trajectory after ' + str(maximum_try_times) +' trials!!!')
                elif self.moveit_velocity_scale == 1:
                    print 'Cannot find a trajectory within the given time steps, retrying...'
                else:
                    print 'Cannot find a trajectory within the given time steps under current max speed, increasing velocity scale...'
                    self.moveit_velocity_scale *= 1.5
                    if self.moveit_velocity_scale > 1:
                        self.moveit_velocity_scale = 1
                    print 'MoveIt max velocity scaling factor: ', self.moveit_velocity_scale
                    self._group.set_max_velocity_scaling_factor(self.moveit_velocity_scale)
        self._group.execute(moveit_plan)
        while True:
            save = raw_input('Do you want to save this trajectory (y / n), \'q\' to exit ?\n')
            if save == 'y':
                time_from_start = []
                actions = []
                for point in moveit_plan.joint_trajectory.points:
                    time_tmp = float(point.time_from_start.secs) + float(point.time_from_start.nsecs) / 1.0e9
                    time_from_start.append(time_tmp)
                    actions.append(point.positions)
                ipl_time_from_start = np.arange(0, time_from_start[-1], step=moveit_plan_time_step)
                ipl_actions = spline(time_from_start, actions, ipl_time_from_start)
                last_action_repeat = np.tile(ipl_actions[-1, :], (self.step_count - ipl_time_from_start.shape[0], 1))
                ipl_actions = np.concatenate((ipl_actions, last_action_repeat), axis=0)
                break
            elif save == 'n':
                ipl_actions = None
                break
            elif save == 'q':
                sys.exit()
            else:
                continue
        return ipl_actions

    def add_noise_to_actions(self):
        for condition in xrange(self.agent['conditions']):
            act_no_noise_filename = os.path.join(self.demo_no_noise_folder,
                                            'actions' + str('{:04d}'.format(condition) + '.json'))
            with open(act_no_noise_filename, 'r') as act_file:
                actions_no_noise = np.array(json.load(act_file))

            tmp_moveit_action_dict = {}
            for idx in xrange(self.num_samples):
                noise = self.action_noise_sigma * np.random.randn(self.agent['T'],
                                                                  self.agent['dU']) + self.action_noise_mu
                for i in range(self.agent['dU']):
                    noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], self.action_smooth_noise_var)
                tmp_moveit_action_dict[idx] = (actions_no_noise + noise).tolist()

            actions_filename = os.path.join(self.demo_folder, 'actions' + str('{:04d}'.format(condition) + '.json'))
            with open(actions_filename, 'w') as action_file:
                json.dump(tmp_moveit_action_dict, action_file, indent=4)
        print 'Adding Noise Done!!!'

def main():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    hyperparams_file = current_directory + '/hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    demo = Demo(hyperparams.config)
    # demo.samples()
    demo.add_noise_to_actions()
    # run_demo = threading.Thread(target=demo.samples)
    # run_demo.daemon = True
    # run_demo.start()
    # plt.ioff()
    # plt.show()



if __name__ == "__main__":
    main()
