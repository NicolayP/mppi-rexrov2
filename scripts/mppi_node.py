#!/usr/bin/env python3
import rospy
import tensorflow as tf

rospy.init_node("MPPI_DP_CONTROLLER")

rospy.loginfo("Set GPU")
if rospy.has_param("~gpu_idx"):
    gpu_idx = rospy.get_param("~gpu_idx")
    gpus = tf.config.list_physical_devices('GPU')
    rospy.loginfo(f"gpus: {gpus}")
    if len(gpus) > gpu_idx:
        rospy.loginfo(f"selected: {gpus[gpu_idx]}")
        tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
        #tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
    else:
        rospy.logerr("GPU index out of range")
rospy.loginfo("Done")

# Import after setting the GPU otherwise tensorflow complains.
from mppi_tf.scripts.src.controller import get_controller
from mppi_tf.scripts.src.cost import get_cost
from mppi_tf.scripts.src.model import get_model

from geometry_msgs.msg import Wrench, WrenchStamped, Twist, PoseStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from mppi_ros.msg import Transition
from mppi_ros.srv import UpdateModelParam, SaveRb, WriteWeights, WriteWeightsResponse, SetLogPath

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from mppi_tf.scripts.src.misc.utile import dtype, npdtype

import numpy as np
import time as t
from datetime import datetime
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

def tItoB_np(euler):
        r = euler[0]
        p = euler[1]
        T = np.array([[1., 0., -np.sin(p)],
                      [0., np.cos(r), np.cos(p) * np.sin(r)],
                      [0., -np.sin(r), np.cos(p) * np.cos(r)]])
        return T

def quat_to_rot_np(quat_state):
    pos = quat_state[:3]
    quat = quat_state[3:7]
    vel = quat_state[-6:]
    r = R.from_quat(quat[:, 0])
    rot = r.as_matrix().reshape((9, 1))

    return np.concatenate([pos, rot, vel], axis=0)


class ContInput(object):
    def __init__(self, history, rot="quat"):
        self._states = np.zeros((history, 13, 1), dtype=npdtype)
        self._actions = np.zeros((history-1, 6, 1), dtype=npdtype)

        self._h_state = history
        self._h_action = history - 1
        self._rot = rot
        self._current_action = 0
        self._current_state = 0
        self._filled_state = False
        self._filled_action = False
        self._filled = False

    def add_state(self, state):
        if not self._filled_state:
            self._states[self._current_state] = state
            self._current_state += 1
            if self._current_state >= self._h_state:
                self._filled_state = True
        else:
            tmpState = self._states[1:]
            self._states = np.concatenate([tmpState, state[None, ...]], axis=0)

    def add_action(self, action):
        if self._h_action < 1:
            self._filled_action = True
            return

        if not self._filled_action:
            self._actions[self._current_action] = action
            self._current_action += 1
            if self._current_action >= self._h_action:
                self._filled_action = True
        else:
            tmpAct = self._actions[:-1]
            self._actions = np.concatenate([tmpAct, action[None, ...]], axis=0)

    def is_filled(self):
        return self._filled_action and self._filled_state

    def get_input(self):
        return self.pred_controller_input(self._states, self._actions)

    def pred_controller_input(self, lagged_state, lagged_action):
        if self._rot == "rot":
            lagged_state_rot = self.quat_to_rot(lagged_state)
        else:
            lagged_state_rot = lagged_state
        if self._h_action < 1:
            return (lagged_state_rot.astype(npdtype), None)
        return (lagged_state_rot.astype(npdtype), lagged_action.astype(npdtype))

    def get_last_state(self):
        if self._rot == "rot":
            lagged_state_rot = self.quat_to_rot(self._states)
        else:
            lagged_state_rot = self._states
        return lagged_state_rot[-1].astype(npdtype)

    def quat_to_rot(self, lagged_state):
        pos = lagged_state[:, :3]
        quat = lagged_state[:, 3:7]
        vel = lagged_state[:, -6:]
        r = R.from_quat(quat[:, :, 0])
        rot = r.as_matrix().reshape((-1, 9, 1))

        return np.concatenate([pos, rot, vel], axis=1)


class MPPINode(object):
    def __init__(self):
        # The action tensor that will be applied to the system.
        # currently we assume a 6DOF force tensor for AUVs.
        self._forces = np.zeros(6, dtype=npdtype)
        # Namespace for identify the vehicle.
        self._namespace = "rexrov2"
        # Flag indicating when the controller is connected to
        # ros and recieves odometry updates.
        self._initOdom = False
        # Variables to profile the controller during
        # its execution.
        self._elapsed = 0.
        self._steps = 0
        self._timeSteps = 0.
        # load the ros parameters and instanciate the
        # desired objects.
        self.load_ros_params()

        # indicates how many previous states
        # needs to be stored and be fed to the
        # model.
        self.lagged = 1
        if "history" in self._config:
            self.lagged = self._config['history']
        # Creates and object that will contain
        # the input state for the controller.
        # it alows to define the rotation representation
        # as well as the number of previous states to
        # maintain.
        self._cont_input = ContInput(self.lagged, self._modelConf['rot'])

        # If the internal model is a ML model
        # It will be trained every x steps.
        if "learnable" in self._modelConf:
            self._learnable = self._modelConf["learnable"]
        else:
            self._learnable = False

        # Logging of the controller's execution.
        if self._log:
            stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
            path = 'graphs/python/'
            if self._dev:
                path = os.path.join(path, 'debug')
            self._logPath = os.path.join(self._logPath,
                                         path,
                                         self._modelConf['type'],
                                         "k" + str(self._samples.numpy()),
                                         "T" + str(self._horizon),
                                         "L" + str(self._lambda),
                                         stamp)
            if self._learnable:
                self.set_learner_path()

        rospy.loginfo("Setup Controller...")

        rospy.loginfo("Get cost")
        # Create the cost function object.
        self._cost = get_cost(
            self._task,
            self._lambda,
            self._gamma,
            self._upsilon,
            self._noise
        )

        rospy.loginfo("Get Model")
        # Create the model object.
        self._model = get_model(
            model_dict=self._modelConf,
            samples=self._samples,
            dt=self._dt,
            state_dim=self._stateDim,
            action_dim=self._actionDim,
            limMax=self._modelConf['limMax'],
            limMin=self._modelConf['limMin'],
            name=self._modelConf['type']
        )

        rospy.loginfo("Get controller")
        init_seq = np.zeros((self._horizon, self._actionDim, 1), dtype=npdtype)
        #init_seq[:, 2] = -263.
        # Create the MPPI controller object.
        self._controller = get_controller(
            model=self._model,
            cost=self._cost,
            k=self._samples,
            tau=self._horizon,
            sDim=self._stateDim,
            aDim=self._actionDim,
            lam=self._lambda,
            upsilon=self._upsilon,
            sigma=self._noise,
            initSeq=init_seq,
            normalizeCost=self._normCost,
            filterSeq=self._filterSeq,
            log=self._log,
            logPath=self._logPath,
            graphMode=self._graphMode,
            debug=self._dev,
            configDict=self._config,
            taskDict=self._task,
            modelDict=self._modelConf
        )

        # If graphMode is activate, the control
        # loop is traced and stored on the gpu
        # for faster inference.
        # if self._graphMode:
            # TODO: figure out why the tracing doesn't maintain the
            # computaitonal graph in memory before the first run of
            # the controller.
            ## USELESS FOR NOW FOR SOME REASON
            # start = t.perf_counter()
            # rospy.loginfo("Trace the tensorflow computational graph...")
            # self._controller.trace()
            # end = t.perf_counter()
            # rospy.loginfo("Tracing done in {:.4f} s".format(end-start))

        rospy.loginfo("Subscrive to odometrie topics...")
        self.init_subscribers()
        rospy.loginfo("Done")

        rospy.loginfo("Setup publisher to thruster topics...")
        self.init_publishers()
        rospy.loginfo("Done")
        
        # Setup a service that updates the model when new
        # parameters are recieved.
        if self._learnable:
            rospy.loginfo("Creating service for parameter update...")
            self._writeWeightsSrv = rospy.Service(
                "mppi/controller/write_weights",
                WriteWeights,
                self.write_weights
            )
        rospy.loginfo("Controller loaded.")

    def init_subscribers(self):
        # Subscribe to odometry topic
        self._odomTopicSub = rospy.Subscriber(
            "odom".
            format(self._uuvName),
            numpy_msg(Odometry),
            self.odometry_callback,
            queue_size=1)
        # Alows to dynamically set the goal in the
        # StaticCost object.
        # TODO: Get the right Goal for the right Cost
        self._setGoalSub = rospy.Subscriber(
            "mppi/setGoal",
            numpy_msg(PoseStamped),
            self.set_goal,
            queue_size=1
        )

    def init_publishers(self):
        # Create the action publisher object.
        # Iterates throught the message type
        # to know which publisher to instanciate.
        if self._thrusterMessage == "WrenchStamped":
            self._thrusterPublisher = rospy.Publisher(
                'thruster_input', WrenchStamped, queue_size=1
            )
            self._publisher_fct = self.publish_control_wrench_stamped

        elif self._thrusterMessage == "Wrench":
            self._thrusterPublisher = rospy.Publisher(
                'thruster_input', Wrench, queue_size=1
            )
            self._publisher_fct = self.publish_control_wrench

        elif self._thrusterMessage == "Twist":
            self._thrusterPublisher = rospy.Publisher(
                'thruster_input', Twist, queue_size=1
            )
            self._publisher_fct = self.publish_control_twist

        else:
            rospy.logerr(
                f"Failed to understand thruster message type\
                 expected |WrenchStamped|Wrench|Twist but\
                 got {self._thrusterMessage}"
            )
        # Publish transition (x_{t}, u_{t}, x_{t+1}) as observed
        # by the controller.
        self._transPub = rospy.Publisher(
            '/mppi/controller/transition',
            Transition,
            queue_size=1
        )

        # Publish the model generated trajectories so that they
        # can be rendered in rviz.
        self._trajsPub = rospy.Publisher(
            "/mppi/samples_traj/",
            MarkerArray,
            queue_size=10
        )

        # Publish obstacle in rviz.
        self._obsPub = rospy.Publisher(
            "mppi/obstacles/",
            MarkerArray,
            queue_size=10
        )

        # Publish cost representation in rviz.
        self._costPub = rospy.Publisher(
            "mppi/cost/",
            Marker,
            queue_size=10
        )

        # Publish the current action sequence
        self._seqPub = rospy.Publisher(
            "mppi/seq",
            Float32MultiArray,
            queue_size=10,
        )

    def load_ros_params(self):
        rospy.loginfo("loading param")
        if rospy.has_param("~model_name"):
            self._uuvName = rospy.get_param("~model_name")
            self._namespace = rospy.get_param("~model_name")
        else:
            rospy.logerr("Need to specify the model name to publish on")
            return

        if rospy.has_param("~samples"):
            self._samples = tf.Variable(rospy.get_param("~samples"))
        else:
            rospy.logerr("Need to set the number of samples to use")
            return

        if rospy.has_param("~horizon"):
            self._horizon = rospy.get_param("~horizon")
        else:
            rospy.logerr("Need to set the number of samples to use")
            return

        if rospy.has_param("~lambda"):
            self._lambda = rospy.get_param("~lambda")
        else:
            rospy.logerr("Need to set the number of samples to use")
            return

        if rospy.has_param("~alpha"):
            alpha = rospy.get_param("~alpha")
            if alpha > 1. or alpha < 0.:
                rospy.logerr("aplha needs to be between [0., lambda]")
                return
            self._gamma = self._lambda * ( 1 - alpha)
        else:
            rospy.logerr("Need to set the number of samples to use")
            return

        if rospy.has_param("~upsilon"):
            self._upsilon = rospy.get_param("~upsilon")
            if self._upsilon < 1.:
                rospy.logerr("Upsilon needs to be higher than 1.")
        else:
            rospy.logerr("Need to set the number of samples to use")
            return

        if rospy.has_param("~dt"):
            self._dt = rospy.get_param("~dt")
        else:
            rospy.logerr("Don't know the timestep.")
            return

        if rospy.has_param("~state_dim"):
            self._stateDim = rospy.get_param("~state_dim")
        else:
            rospy.logerr("Don't know the state dimensionality.")
            return

        if rospy.has_param("~action_dim"):
            self._actionDim = rospy.get_param("~action_dim")
        else:
            rospy.logerr("Don't know the actuator dimensionality.")
            return

        if rospy.has_param("~cost"):
            self._task = rospy.get_param("~cost")
        else:
            rospy.logerr("No cost function given.")
            return

        if rospy.has_param("~cost_norm"):
            self._normCost = rospy.get_param("~cost_norm")
        else:
            rospy.logwarn("Cost norm not given assuemd False")
            self._normCost = False

        if rospy.has_param("~model"):
            self._modelConf = rospy.get_param("~model")
        else:
            rospy.logerr("No internal model given.")
            return

        if rospy.has_param("~filter_seq"):
            self._filterSeq = rospy.get_param("~filter_seq")
        else:
            self._filterSeq = False

        if rospy.has_param("~config"):
            self._config = rospy.get_param("~config")
            if "noise" in self._config:
                self._noise = self._config['noise']
            else:
                rospy.logerr("No noise given in controller Config file")
                return
        else:
            rospy.logerr("No internal config given.")
            return

        if rospy.has_param("~log"):
            self._log = rospy.get_param("~log")
            if rospy.has_param("~log_path"):
                self._logPath = rospy.get_param("~log_path")
        else:
            rospy.logerr("No log flag given.")

        if rospy.has_param("~dev"):
            self._dev = rospy.get_param("~dev")
        else:
            rospy.logerr("No flag for dev mode given.")

        if rospy.has_param("~visu_samples"):
            self._visu_samples = rospy.get_param("~visu_samples")
        else:
            rospy.logwarn("No flag for visualisation mode given. Set to False")
            self._visu_samples = False
        
        if rospy.has_param("~odom_frame"):
            self._odom_frame = rospy.get_param("~odom_frame")
        else:
            rospy.logwarn("No odometry frame given, publishing of markers is not guarenteed.")
            self._odom_frame = "odom"

        if rospy.has_param("~graph_mode"):
            self._graphMode = rospy.get_param("~graph_mode")
        else:
            rospy.logerr("No flag for graph mode given.")

        if rospy.has_param("~thruster_message"):
            self._thrusterMessage = rospy.get_param("~thruster_message")
        else:
            rospy.logwarn("No thruster message type given, assumes WrenchStamped")
            self._thrusterMessage = "WrenchStamped"

    def publish_control_wrench(self, forces):
        if not self._initOdom:
            return
        
        forceMsg = Wrench()
        # Force
        forceMsg.force.x = forces[0]
        forceMsg.force.y = forces[1]
        forceMsg.force.z = forces[2]
        # Torque
        forceMsg.torque.x = forces[3]
        forceMsg.torque.y = forces[4]
        forceMsg.torque.z = forces[5]

        self._thrusterPublisher.publish(forceMsg)

    def publish_control_wrench_stamped(self, forces):
        if not self._initOdom:
            return

        forceMsg = WrenchStamped()
        forceMsg.header.stamp = rospy.Time.now()
        forceMsg.header.frame_id = f"{self._namespace}/base_link"
        # Force
        forceMsg.wrench.force.x = forces[0]
        forceMsg.wrench.force.y = forces[1]
        forceMsg.wrench.force.z = forces[2]
        # Torque
        forceMsg.wrench.torque.x = forces[3]
        forceMsg.wrench.torque.y = forces[4]
        forceMsg.wrench.torque.z = forces[5]

        self._thrusterPublisher.publish(forceMsg)

    def publish_control_twist(self, forces):
        if not self._initOdom:
            return

        #sign = np.sign(forces)
        #sign = sign - np.sign(self._prevForce)
        # Normalize forces vector.
        forces = forces / np.linalg.norm(forces)
        forceMsg = Twist()
        # Force
        forceMsg.linear.x = forces[0]
        forceMsg.linear.y = forces[1]
        forceMsg.linear.z = forces[2]
        # Torque
        forceMsg.angular.x = forces[3]
        forceMsg.angular.y = forces[4]
        forceMsg.angular.z = forces[5]

        self._prevForce = forces

        self._thrusterPublisher.publish(forceMsg)
        rospy.loginfo(f"Setting goal to \n {np.squeeze(self._controller._cost.get_goal())} \n Succesfull.")


        # Header
        transMsg = Transition()
        transMsg.header.stamp = rospy.Time.now()
        transMsg.header.frame_id = f"{self._namespace}/base_link"

        # Transition
        transMsg.x = x
        transMsg.u = u
        transMsg.xNext = xNext

        self._transPub.publish(transMsg)

    def call_controller(self, controller_input):
        start = t.perf_counter()
        self._forces, trajs, weights, seq = self._controller.next(controller_input)
        end = t.perf_counter()
        # disable controller for roll and pitch as there is non for falcon.
        #self._forces[3] = 0.
        #self._forces[4] = 0.
        # Normalize forces vector.

        self._elapsed += (end-start)
        self._timeSteps += 1
        self._steps += 1

        # If dev mode, don't publish the forces.
        if not self._dev:
            self._publisher_fct(self._forces)

        # Publish the generated trajectories
        # to rviz together with the weights.
        if self._visu_samples:
            #self.publish_expected(expected)
            self.publish_samples_traj(trajs, weights)
            self.publish_obstacles()
            self.publish_cost(state=controller_input[0][-1])
            self.publish_seq(seq)

        # Print the performances of the controller
        # every 10 steps. ~1/sec
        if self._steps % 10 == 0:
            rospy.loginfo("*"*5 + " MPPI Time stats " + "*"*5)
            rospy.loginfo("* Next step : {:.4f} (sec)".format(self._elapsed/self._timeSteps))
            self._elapsed = 0.
            self._timeSteps = 0

    def odometry_callback(self, msg):
        # If first call, we need to boot the controller
        # by initializing the controller input.
        if not self._initOdom:
            # First call
            self._prevTime = rospy.get_rostime()
            self._prevState  = self.update_odometry(msg)
            self._state = self._prevState.copy()
            self._cont_input.add_state(self._state.copy())
            self._cont_input.add_action(self._forces[..., None].copy())
            # When the controller input is filled we can start the controller.
            self._initOdom = self._cont_input.is_filled()
            self._first = True
            return

        else:
            # check if the elapsed time is
            # enough, to call the controller
            time = rospy.get_rostime()
            dt = time - self._prevTime
            if dt.to_sec() < self._dt:
                return

        # Get the current state.
        self._prevTime = time
        self._state = self.update_odometry(msg)

        # TODO: save the transition.
        self._controller.save(
            self._cont_input.get_input(),
            self._forces[..., None],
            self._state
        )

        # Send the data for the learner.
        # self.publish_transition(
        #     self._prevState,
        #     np.expand_dims(self._forces, -1),
        #     self._state
        # )

        # Add the current state to the controller input.
        self._cont_input.add_state(self._state.copy())
        # update previous state.
        self._prevState = self._state

        if self._first:
            rospy.loginfo("First call to the controller (suspicious tracing)!")
            if self._graphMode:
                self._controller.tracing = True

        self.call_controller(self._cont_input.get_input())
        if self._first:
            rospy.loginfo("First call to the controller finished!")
            self._first = False
            if self._graphMode:
                self._controller.tracing = False

        self._cont_input.add_action(self._forces[..., None].copy())

        if self._learnable:
            if self._steps % 50 == 0:
                self.update_model()

    def update_odometry(self, msg):

        """Odometry topic subscriber callback function."""
        # The frames of reference delivered by the odometry seems to be as
        # follows
        # position -> world frame
        # orientation -> world frame
        # linear velocity -> world frame
        # angular velocity -> world frame

        #if self._model._inertialFrameId != msg.header.frame_id:
        #    raise rospy.ROSException('The inertial frame ID used by the '
        #                             'vehicle model does not match the '
        #                             'odometry frame ID, vehicle=%s, odom=%s' %
        #                             (self._model._inertialFrameId,
        #                              msg.header.frame_id))

        # Update the velocity vector
        # Update the pose in the inertial frame
        state = np.zeros((13, 1), dtype=npdtype)
        state[0:3, :] = np.array([[msg.pose.pose.position.x],
                                  [msg.pose.pose.position.y],
                                  [msg.pose.pose.position.z]])

        # Using the (w, x, y, z) format for quaternions
        state[3:7, :] = np.array([[msg.pose.pose.orientation.x],
                                  [msg.pose.pose.orientation.y],
                                  [msg.pose.pose.orientation.z],
                                  [msg.pose.pose.orientation.w]])

        # Linear velocity on the INERTIAL frame
        linVel = np.array([msg.twist.twist.linear.x,
                           msg.twist.twist.linear.y,
                           msg.twist.twist.linear.z])
        # Transform linear velocity to the BODY frame
        rotItoB = rotBtoI_np(state[3:7, 0]).T

        linVel = np.expand_dims(np.dot(rotItoB, linVel), axis=-1)
        # Angular velocity in the INERTIAL frame
        angVel = np.array([msg.twist.twist.angular.x,
                           msg.twist.twist.angular.y,
                           msg.twist.twist.angular.z])
        # Transform angular velocity to BODY frame
        angVel = np.expand_dims(np.dot(rotItoB, angVel), axis=-1)
        # Store velocity vector
        state[7:13, :] = np.concatenate([linVel, angVel], axis=0)
        return state

    def write_weights(self, req):
        self._model.update_weights(req.weights)
        return WriteWeightsResponse(True)

    def update_model(self):
        rospy.loginfo("Updating parameter model")
        rospy.wait_for_service('/mppi/learner/update_model_params')
        try:
            start = t.perf_counter()
            updateModelSrv = rospy.ServiceProxy('/mppi/learner/update_model_params', UpdateModelParam)
            resp = updateModelSrv(train=True, save=self._log, step=self._steps)
            end = t.perf_counter()
            rospy.loginfo("Service replied in {:.4f} s".format(end-start))
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def save_rb(self, file):
        rospy.loginfo("Creating client for save service..")
        rospy.wait_for_service('/mppi/learner/save_rb')
        try:
            saveRbSrv = rospy.ServiceProxy('/mppi/learner/save_rb', SaveRb)
            resp = saveRbSrv(file)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        rospy.loginfo("Done")

    def set_learner_path(self):
        rospy.loginfo("Creating client for update service..")
        rospy.wait_for_service('/mppi/learner/set_log_path')
        try:
            setLogPath = rospy.ServiceProxy('/mppi/learner/set_log_path', SetLogPath)
            resp = setLogPath(self._logPath)
            print(resp)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        rospy.loginfo("Done")

    def set_goal(self, msg):
        goal = np.zeros((13,1))
        goal[:3] = np.array(
            [[msg.pose.position.x],
             [msg.pose.position.y],
             [msg.pose.position.z]]
        )
        goal[3:7] = np.array(
            [[msg.pose.orientation.x],
             [msg.pose.orientation.y],
             [msg.pose.orientation.z],
             [msg.pose.orientation.w]]
        )
        self._controller.set_goal(goal)
        rospy.loginfo(f"Setting goal to \n {np.squeeze(self._controller._cost.get_goal())} \n Succesfull.")

    def populate_marker(self, traj, w, id):
        m = Marker()
        m.header.frame_id = self._odom_frame
        m.type = m.LINE_STRIP
        m.action = m.ADD
        m.id = id
        m.scale.x = 0.05
        m.scale.y = 0.05
        #m.color.a = w
        m.color.a = w
        m.color.g = 1.

        m.pose.orientation.w = 1.0
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        for t in traj:
            p = Point()
            p.x = t[0]
            p.y = t[1]
            p.z = t[2]
            m.points.append(p)
        return m

    def populate_expected_marker(self, traj):
        m = Marker()
        m.header.frame_id = self._odom_frame
        m.type = m.LINE_STRIP
        m.action = m.ADD
        m.id = 0
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.color.a = 1.
        m.color.g = 0.
        m.color.r = 1.

        m.pose.orientation.w = 1.0
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        for t in traj:
            p = Point()
            p.x = t[0]
            p.y = t[1]
            p.z = t[2]
            m.points.append(p)
        return m

    def populate_marker_array(self, trajs, weights, nb_plots=100, expected=None):
        weights = (weights - np.min(weights))/np.max(weights)
        ind = np.argsort(-weights) # negate to get highest weight on first elements
        w = weights[ind]
        sort_trajs = trajs[ind]
        a = MarkerArray()
        for i in range(nb_plots):
            m = self.populate_marker(sort_trajs[i], w[i], i)
            a.markers.append(m)
        if expected is not None:
            m = self.populate_expected_marker(expected)
            a.markers.append(m)
        return a

    def publish_samples_traj(self, trajs, weights):
        nb_plots = 1000
        if trajs.shape[0] < nb_plots:
            nb_plots = trajs.shape[0]
        a = self.populate_marker_array(trajs, weights, nb_plots)
        self._trajsPub.publish(a)

    def publish_obstacles(self):
        obs = self._cost.get_obstacles()
        a = MarkerArray()
        for i, o in enumerate(obs):
            pc, q = o.get_pose()
            h, r = o.get_param()
            m = Marker()
            m.id = i
            m.header.frame_id = self._odom_frame
            m.type = m.CYLINDER
            m.action = m.ADD
            m.color.a = 1.
            m.color.r = 1.
            m.scale.x = 2.*r
            m.scale.y = 2.*r
            m.scale.z = h
            m.pose.position.x = pc[0]
            m.pose.position.y = pc[1]
            m.pose.position.z = pc[2]

            m.pose.orientation.x = q[0]
            m.pose.orientation.y = q[1]
            m.pose.orientation.z = q[2]
            m.pose.orientation.w = q[3]

            a.markers.append(m)

        self._obsPub.publish(a)

    def publish_cost(self, state):
        segments = self._cost.get_3D(state)
        m = Marker()
        m.id = 0
        m.header.frame_id = self._odom_frame
        m.type = m.LINE_STRIP
        m.action = m.ADD
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.color.a = 1.
        m.color.r = 1.
        m.color.g = 0.5
        m.color.b = 0.0

        m.pose.orientation.w = 1.0
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        for t in segments:
            p = Point()
            p.x = t[0]
            p.y = t[1]
            p.z = t[2]
            m.points.append(p)
        self._costPub.publish(m)

    def publish_expected(self, traj):
        m = Marker()
        m.header.frame_id = self._odom_frame
        m.type = m.LINE_STRIP
        m.action = m.ADD
        m.id = 0
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.color.a = 1.
        m.color.g = 0.
        m.color.r = 1.

        m.pose.orientation.w = 1.0
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        for t in traj:
            p = Point()
            p.x = t[0]
            p.y = t[1]
            p.z = t[2]
            m.points.append(p)
        self._expectPub.publish(m)

    def publish_seq(self, seq):
        msg = Float32MultiArray()
        msg.data = seq.reshape((-1))
        msg.layout.data_offset = 0
        msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]

        # dim[0] is the vertical dimension of your matrix
        msg.layout.dim[0].label = "Tau"
        msg.layout.dim[0].size = self._horizon
        msg.layout.dim[0].stride = self._actionDim*self._horizon
        # dim[1] is the horizontal dimension of your matrix
        msg.layout.dim[1].label = "aDim"
        msg.layout.dim[1].size = self._actionDim
        msg.layout.dim[1].stride = self._actionDim

        self._seqPub.publish(msg)

if __name__ == "__main__":
    try:
        node = MPPINode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")
