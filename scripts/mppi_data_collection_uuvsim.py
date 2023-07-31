#!/usr/bin/env python3
import rospy

from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

import rosbag
import os

from scipy.spatial.transform import Rotation as R


def rotBtoI_np(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    return np.array([
                        [1 - 2 * (y**2 + z**2),
                        2 * (x * y - z * w),
                        2 * (x * z + y * w)],
                        [2 * (x * y + z * w),
                        1 - 2 * (x**2 + z**2),
                        2 * (y * z - x * w)],
                        [2 * (x * z - y * w),
                        2 * (y * z + x * w),
                        1 - 2 * (x**2 + y**2)]
                    ])


class MPPIDataCollection(object):
    def __init__(self, sDim, aDim):
        self._sDim, self._aDim = sDim, aDim
        self._forces = np.zeros(self._sDim)
        # publisher to thrusters.
        self._thrustPub = rospy.Publisher(
            'thruster_input', WrenchStamped, queue_size=1)

        # subscriber to pose_gt
        self._odomSub = rospy.Subscriber("{}/pose_gt".
                                            format(self._uuvName),
                                         numpy_msg(Odometry),
                                         self.odom_callback)
        self._run = False
        self.collect_data(self._n, self._logDir)

    def load_ros_params(self):
        if rospy.has_param("~rollouts"):
            self._n = rospy.get_param("~rollouts")
        else:
            self._n = 100

        if rospy.has_param("~max_steps"):
            self._maxSteps = rospy.get_param("~max_steps")
        else:
            self._maxSteps = 20

        if rospy.has_param("~log_dir"):
            self._logDir = rospy.get_param("~log_dir")
            if os.path.exists(self._logDir):
                rospy.loginfo("Saving directory already exists.")
            else:
                os.mkdir(self._logDir)
        else:
            rospy.logerr("Need to give a saveing directory.")

        if rospy.has_param("~buffer_size"):
            self._bufferSize = rospy.get_param("~buffer_size")
        else:
            self._bufferSize = self._n*self._maxSteps

        if rospy.has_param("~uuv_name"):
            self._uuvName = rospy.get_param("~uuv_name")
        else:
            rospy.logerr("Need to specify the vehicule's name")


        if rospy.has_param("~max_thrust"):
            self._maxThrust = rospy.get_param("~max_thrust")
            self._std = 0.1*self._maxThrust
        else:
            rospy.logerr("Did not specify the max thrust of the vehicle")

    def collect_data(self, n, dir):
        # launch self.run n times
        # Save observer transitions to file.
        stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        dir_stamp = os.path.join(dir, stamp)
        if not os.path.exists(dir_stamp):
            os.makedirs(dir_stamp)

        rospy.loginfo("Start recording")
        for i in tqdm(range(n)):
            filename = "run{}.bag".format(i)
            file = os.path.join(dir_stamp, filename)
            self.bag = rosbag.Bag(file, 'w')

            if self._run == False:
                self.run()
            while self._run:
                time.sleep(1)

            self.bag.close()
        rospy.loginfo("Stop recording")

    def run(self):
        # delete robot instance.
        # reset simulation and pause it
        # spawn new robot with random speed and postion.
        # rollout the robot with random actions.
        # log transtions in the observer.
        self.stop()
        self.reset()
        self.spawn()
        self.rollout()

    def stop(self):
        # Resets the simulation and pauses it.
        try:
            rospy.wait_for_service('/gazebo/pause_physics', 1)
            pauseSim = rospy.ServiceProxy('/gazebo/pause_physics',
                                          Empty)
            resp = pauseSim()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(2)

    def reset(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            resetSim = rospy.ServiceProxy('/gazebo/reset_world',
                                          Empty)
            resp = resetSim()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(2)

    def init_state(self):
        p = np.random.rand(3)
        q = np.array([0., 0., 0., 1.])
        pDot = np.random.rand(3)*5
        rDot = np.zeros(3)
        return p, q, pDot, rDot

    def spawn(self):
        # generate random state dict.
        state = ModelState()
        p, q, pDot, rDot = self.init_state()
        state.model_name = self._uuvName
        state.pose.position.x = p[0]
        state.pose.position.y = p[1]
        state.pose.position.z = p[2]

        state.pose.orientation.x = q[0]
        state.pose.orientation.y = q[1]
        state.pose.orientation.z = q[2]
        state.pose.orientation.w = q[3]

        state.twist.linear.x = pDot[0]
        state.twist.linear.y = pDot[1]
        state.twist.linear.z = pDot[2]

        state.twist.angular.x = rDot[0]
        state.twist.angular.y = rDot[1]
        state.twist.angular.z = rDot[2]

        state.reference_frame = 'world'

        # spawn the robot using the gazebo service.
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            setModel = rospy.ServiceProxy('/gazebo/set_model_state',
                                          SetModelState)
            resp = setModel(state)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def rollout(self):
        # performs a rollout for `steps` steps
        self._step = 0
        self._first = True
        self.start_sim()

    def start_sim(self):
        # Starts the simulation when everything is ready
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            pauseSim = rospy.ServiceProxy('/gazebo/unpause_physics',
                                          Empty)
            resp = pauseSim()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(0.25)
        self._run = True

    def odom_callback(self, msg):
        if not self._run or self._step < 10:
            self.update_odometry(msg)
            if self._run and self._step < 10:
                self._step += 1
            return
        
        self.update_odometry(msg)
        self._forces = self.gen_action(rospy.Time.now().to_sec())
        self.publish_control_wrench(self._forces.copy())

        self._step += 1
        if self._step >= self._maxSteps:
            self._run = False
            self.stop()

    def gen_action(self, t):
        return np.random.rand(6)*100

    def update_odometry(self, msg):
        """Odometry topic subscriber callback function."""
        if self._run:
            self.bag.write("/{}/pose_gt".format(self._uuvName), msg)

    def publish_control_wrench(self, forces):
        forceMsg = WrenchStamped()
        forceMsg.header.stamp = rospy.Time.now()
        forceMsg.header.frame_id = '{}/{}'.format(self._uuvName,
                                                  'base_link')

        # Force
        forceMsg.wrench.force.x = forces[0]
        forceMsg.wrench.force.y = forces[1]
        forceMsg.wrench.force.z = forces[2]
        # Torque
        forceMsg.wrench.torque.x = forces[3]
        forceMsg.wrench.torque.y = forces[4]
        forceMsg.wrench.torque.z = forces[5]

        self._thrustPub.publish(forceMsg)
        self.bag.write("/thruster_input", forceMsg)


class MPPIDataCollectionLin(MPPIDataCollection):
    def __init__(self, sDim, aDim):
        self.load_ros_params()
        super(MPPIDataCollectionLin, self).__init__(sDim, aDim)

    def init_state(self):
        p = np.array([0., 0., -20.])
        euler = np.zeros(3)
        euler[2] = np.random.uniform(0, 2*np.pi)
        q = R.from_euler('xyz', euler).as_quat()
        pDot = np.random.rand(3)*5
        rDot = np.zeros(3)

        activated = np.random.choice([0, 1], size=(self._aDim))
        values = np.random.uniform(low=-self._maxThrust, high=self._maxThrust, size=(self._aDim))
        self._const_force = activated*values
        return p, q, pDot, rDot
    
    def gen_action(self, t):
        return self._const_force


class MPPIDataCollection2D(MPPIDataCollection):
    def __init__(self, sDim, aDim):
        self.tau = 1000
        self.load_ros_params()
        super(MPPIDataCollection2D, self).__init__(sDim, aDim)

    def init_state(self):
        '''
            This generates config in 2D XY plane.
        '''
        p = np.random.rand(3)
        p[2] = -50.
        yaw = np.random.uniform(0., 2*np.pi)
        r = R.from_euler('XYZ', [0., 0., yaw])
        q = r.as_quat()
        pDot = np.zeros(3)
        qDot = np.zeros(3)

        self.force_fct = gen_actions_fct(self.tau, [100, 100, 80])

        return p, q, pDot, qDot

    def gen_action(self, t):
        '''
            Nullify Fz, Tx, Ty in (Fx:0, Fy:1, Fz:2, Tx:3, Ty:4, Tz:5)
        '''
        return self.force_fct(t)


class MPPIModelVerif(MPPIDataCollection):
    def __init__(self, sDim, aDim, debug=False):
        self.load_ros_params()
        self.call = 0
        if debug:
            # subscriber to pose_gt
            self._debugAddedMassCoriSub = rospy.Subscriber("/debug/forces/{}/base_link/added_coriolis".
                                            format(self._uuvName),
                                         numpy_msg(WrenchStamped),
                                         self.coriolis_callback)

            self._debugAddedMassSub = rospy.Subscriber("/debug/forces/{}/base_link/added_mass".
                                            format(self._uuvName),
                                         numpy_msg(WrenchStamped),
                                         self.added_mass_callback)

            self._debugDampingSub = rospy.Subscriber("/debug/forces/{}/base_link/damping".
                                            format(self._uuvName),
                                         numpy_msg(WrenchStamped),
                                         self.damping_callback)

            self._debugRestoringSub = rospy.Subscriber("/debug/forces/{}/base_link/restoring".
                                            format(self._uuvName),
                                         numpy_msg(WrenchStamped),
                                         self.restoring_callback)
        super(MPPIModelVerif, self).__init__(sDim, aDim)

    def init_state(self):
        p = np.zeros(3)
        p[2] = -50.
        r = R.from_euler('XYZ', [0., 0., 0.])
        q = r.as_quat()
        pDot = np.zeros(3)
        qDot = np.zeros(3)

        f = np.zeros(self._aDim)
        f[self.call] = 200.

        self.force_fct = f

        self.call +=1

        return p, q, pDot, qDot

    def gen_action(self, t):
        return self.force_fct

    def coriolis_callback(self, msg):
        """Coriolis topic subscriber callback function."""
        if self._run:
            print(msg)
            self.bag.write("/{}/coriolis_wrench".format(self._uuvName), msg)

    def added_mass_callback(self, msg):
        """Coriolis topic subscriber callback function."""
        if self._run:
            self.bag.write("/{}/added_mass_wrench".format(self._uuvName), msg)

    def damping_callback(self, msg):
        """Coriolis topic subscriber callback function."""
        if self._run:
            self.bag.write("/{}/damping_wrench".format(self._uuvName), msg)

    def restoring_callback(self, msg):
        """Coriolis topic subscriber callback function."""
        if self._run:
            self.bag.write("/{}/restoring_wrench".format(self._uuvName), msg)


def gen_action_wave_fct(tau, max_thrust, max_waves):
    # Sample the number of sin waves.
    # Based on the number sample a number of 
    # scaling and width for every waves.
    nb_waves = np.random.randint(max_waves)
    s = np.random.uniform(-max_thrust, max_thrust, nb_waves)
    w = np.random.uniform(0, tau, nb_waves)
    return lambda t: np.clip((s * np.sin(np.pi/w*t)).sum(), -max_thrust, max_thrust)


def gen_action_cst_fct(max_thrust):
    force = np.random.uniform(-max_thrust, max_thrust)
    return lambda t: force


def gen_action_fct(tau, max_thrust):
    nu = np.random.uniform()
    if nu < 0.2:
        return gen_action_cst_fct(max_thrust)
    return gen_action_wave_fct(tau, max_thrust, 10)


def gen_actions_fct(tau, max_thrusts):
    Fx = gen_action_fct(tau, max_thrusts[0])
    Fy = gen_action_fct(tau, max_thrusts[1])
    Tz = gen_action_fct(tau, max_thrusts[2])
    return lambda t: np.array([Fx(t), Fy(t), 0., 0., 0., Tz(t)])


if __name__ == "__main__":
    print("Mppi - Data Collection")
    rospy.init_node("MPPI_DP_CONTROLLER")

    try:
        node = MPPIModelVerif(13, 6)
        # rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")
