import argparse
import numpy as np
import threading
import time

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import Base_pb2, ControlConfig_pb2
from kortex_api.Exceptions.KServerException import KServerException

def euler2rotm(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def ee_pose_dist(ee_pose1, ee_pose2):
    xyz1, rpy1 = np.split(np.array(ee_pose1), 2)
    xyz2, rpy2 = np.split(np.array(ee_pose2), 2)
    xyz_d = np.linalg.norm(xyz1-xyz2)
    rpy1, rpy2 = np.deg2rad(rpy1), np.deg2rad(rpy2)
    rpy_d = np.arccos((np.trace(euler2rotm(rpy1).T @ euler2rotm(rpy2))-1)/2)
    return xyz_d + rpy_d

def angles_dist(angles1, angles2):
    angle_diff = np.abs(np.array(angles1) - np.array(angles2))
    angle_diff = [np.min(x, np.abs(x - 2*np.pi)) for x in angle_diff]
    return np.linalg.norm(angle_diff)

def angles_deg2rad(angles):
    return [np.deg2rad(x) for x in angles]

def angles_rad2deg(angles):
    return [np.rad2deg(x) for x in angles]

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            print(notification)
            e.set()
    return check

def SendCallWithRetry(call, retry,  *args):
    i = 0
    arg_out = []
    while i < retry:
        try:
            arg_out = call(*args)
            break
        except:
            i = i + 1
            continue
    if i == retry:
        print("Failed to communicate")
    return arg_out

def parseConnectionArguments(parser = argparse.ArgumentParser()):
    parser.add_argument("--ip", type=str, help="IP address of destination", default="192.168.1.10")
    parser.add_argument("-u", "--username", type=str, help="username to login", default="admin")
    parser.add_argument("-p", "--password", type=str, help="password to login", default="admin")
    return parser.parse_args()

class DeviceConnection:
    
    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection(args): 
        """
        returns RouterClient required to create services and send requests to device or sub-devices,
        """

        return DeviceConnection(args.ip, port=DeviceConnection.TCP_PORT, credentials=(args.username, args.password))

    @staticmethod
    def createUdpConnection(args): 
        """        
        returns RouterClient that allows to create services and send requests to a device or its sub-devices @ 1khz.
        """

        return DeviceConnection(args.ip, port=DeviceConnection.UDP_PORT, credentials=(args.username, args.password))

    def __init__(self, ipAddress, port=TCP_PORT, credentials = ("","")):

        self.ipAddress = ipAddress
        self.port = port
        self.credentials = credentials

        self.sessionManager = None

        # Setup API
        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def connect(self):
        self.transport.connect(self.ipAddress, self.port)

        if (self.credentials[0] != ""):
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000   # (milliseconds)
            session_info.connection_inactivity_timeout = 2000 # (milliseconds)

            self.sessionManager = SessionManager(self.router)
            print("Logging as", self.credentials[0], "on device", self.ipAddress)
            self.sessionManager.CreateSession(session_info)

    # Called when entering 'with' statement
    def __enter__(self):
        self.connect()
        return self.router

    def disconnect(self):
        if self.sessionManager != None:

            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000 
            
            self.sessionManager.CloseSession(router_options)

        self.transport.disconnect()

    # Called when exiting 'with' statement
    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

class KortexAsyncInterface:
    def __init__(self, interface_args, arm_model=None):
        self.args = interface_args
        self.tcp_connection = DeviceConnection(self.args.ip, port=DeviceConnection.TCP_PORT, credentials=(self.args.username, self.args.password))
        self.udp_connection = DeviceConnection(self.args.ip, port=DeviceConnection.UDP_PORT, credentials=(self.args.username, self.args.password))

        # "Slow" client; for things like cartesian move command
        self.base_client = BaseClient(self.tcp_connection.router)
        # "Slow" client; for things like robot config
        self.control_config_client = ControlConfigClient(self.tcp_connection.router)
        # "Fast" client; for things like closed loop control (supposedly 1kHz)
        self.base_cyclic_client = BaseCyclicClient(self.udp_connection.router)
        self.run_thread = None
        self.run_flag = False

        self.actuator_count = 7 

        # TEMP freq testing
        self.loop_count = 0
        self.start_time = -1 

        # will be numpy arrays.
        self.angles = None
        self.velocities = None
        self.torques = None
        self.filter_torques = None
        self.arm_model = arm_model

    def connect(self):
        self.tcp_connection.connect()
        self.udp_connection.connect()

    def __enter__(self):
        self.connect()
        self.run()
        return self

    def run(self):
        self.run_flag = True
        self.run_thread = threading.Thread(group=None, target=self._run, 
                                           name="kortex_interface_run",args=(), kwargs={})
        self.run_thread.start()

    def _run(self):
        assert self.actuator_count == self.base_client.GetActuatorCount().count
        self.start_time = time.time()

        while self.run_flag:
            try:
                base_feedback = self.base_cyclic_client.RefreshFeedback()
            except Exception as e:
                print("failed to refresh feedback...")
                continue
            
            self.loop_count += 1
            angles = np.empty(7)
            velocities = np.empty(7)
            torques = np.empty(7)
            for joint_idx in range(self.actuator_count):
                angles[joint_idx] = base_feedback.actuators[joint_idx].position
                velocities[joint_idx] = base_feedback.actuators[joint_idx].velocity
                torques[joint_idx] = base_feedback.actuators[joint_idx].torque
                # other fields (not recorded): voltage, current_motor

            self.angles = np.deg2rad(angles)
            ########################## Testing ##########################
            self.angles = np.where(self.angles > np.pi, self.angles - 2*np.pi, self.angles)
            ########################## Testing ##########################
            if self.arm_model is not None:
                self.arm_model.setJointAngles(self.angles)
                sim_torques = self.arm_model.getInternalTorques()
                torques -= sim_torques

            if self.filter_torques is not None:
                alpha = 0.01
                self.filter_torques = alpha*torques + (1-alpha)*self.filter_torques
            else:
                self.filter_torques = torques

            self.velocities = np.deg2rad(velocities)
            self.torques = torques
    
    def stop(self):
        self.run_flag = False
        # technically can cause race condition and throw exception but w/e
        self.run_thread.join()

    def disconnect(self):
        self.tcp_connection.disconnect()
        self.udp_connection.disconnect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self.disconnect()
    
    def safety_check(self, max_tau=20.0, sim_torques=None):
        if sim_torques is not None:
            torques = self.torques - sim_torques
        else:
            torques = self.torques
        print('obs torques:', self.torques)
        print('sim torques:', sim_torques)

        if np.any(np.abs(torques) > max_tau):
            print(f"INFO: reach maximum torques {max_tau}!")
            return False
        return True
    
    def stop_action(self):
        self.base_client.StopAction()
    
    def set_joint_softlimits(self, joint_speed_limit, joint_accel_limit=None, verbose=False):
        # NOTE: Set joint soft limits may prevent cartesian movement from reaching target pose.
        curr_speed_limits = ControlConfig_pb2.JointSpeedSoftLimits()
        if verbose:
            print('current speed limits:', curr_speed_limits)
        curr_speed_limits.control_mode = ControlConfig_pb2.ANGULAR_TRAJECTORY
        for i in range(self.actuator_count) : 
            if i < 4: # Actuators 0 to 3 are bigs on 7DOF
                curr_speed_limits.joint_speed_soft_limits.append(joint_speed_limit)
            else : # small actuators
                curr_speed_limits.joint_speed_soft_limits.append(joint_speed_limit)

        self.control_config_client.SetJointSpeedSoftLimits(curr_speed_limits)

        if joint_accel_limit is not None:
            curr_accel_limits = ControlConfig_pb2.JointAccelerationSoftLimits()
            if verbose:
                print('current accel limits:', curr_accel_limits)
            curr_accel_limits.control_mode = ControlConfig_pb2.ANGULAR_TRAJECTORY
            for i in range(self.actuator_count) : 
                if i < 4: # Actuators 0 to 3 are bigs on 7DOF
                    curr_accel_limits.joint_acceleration_soft_limits.append(joint_accel_limit)
                else : # small actuators
                    curr_accel_limits.joint_acceleration_soft_limits.append(joint_accel_limit)

            self.control_config_client.SetJointAccelerationSoftLimits(curr_accel_limits)
    
    def set_twist_linear_softlimits(self, twist_linear_limits):
        twist_speed_limits = ControlConfig_pb2.TwistLinearSoftLimit()
        twist_speed_limits.control_mode = ControlConfig_pb2.CARTESIAN_TRAJECTORY

        twist_speed_limits.twist_linear_soft_limit = twist_linear_limits
        self.control_config_client.SetTwistLinearSoftLimit(twist_speed_limits)

    def cartesian_read(self):
        feedback = self.base_cyclic_client.RefreshFeedback()

        x = feedback.base.tool_pose_x             # (meters)
        y = feedback.base.tool_pose_y             # (meters)
        z = feedback.base.tool_pose_z             # (meters)
        theta_x = feedback.base.tool_pose_theta_x # (degrees)
        theta_y = feedback.base.tool_pose_theta_y # (degrees)
        theta_z = feedback.base.tool_pose_theta_z # (degrees)
        return [x, y, z, theta_x, theta_y, theta_z]

    def cartesian_move(self, target_ee_pose):
        x, y, z, theta_x, theta_y, theta_z = target_ee_pose

        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base_client.SetServoingMode(base_servo_mode)

        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = ""

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = x # (meters)
        cartesian_pose.y = y # (meters)
        cartesian_pose.z = z # (meters)
        cartesian_pose.theta_x = theta_x # (degrees)
        cartesian_pose.theta_y = theta_y # (degrees)
        cartesian_pose.theta_z = theta_z # (degrees)

        e = threading.Event()
        self.notification_handle = self.base_client.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base_client.ExecuteAction(action)
        return e

    def persistent_cartesian_move(self, target_ee_pose, reach_tol=0.01,  max_time=30.0, max_tau=20.0, arm_env=None):
        start_time = time.time()
        while True:
            curr_ee_pose = self.cartesian_read()
            print('current ee pose:', curr_ee_pose)
            print('target ee pose:', target_ee_pose)
            if (ee_pose_dist(curr_ee_pose, target_ee_pose) < reach_tol):
                print('current ee pose:', curr_ee_pose)
                print('target ee pose:', target_ee_pose)
                break
            e = self.cartesian_move(target_ee_pose)
            while not e.isSet():
                time.sleep(0.01)
                if time.time()-start_time > max_time:
                    return False
                arm_env.setJointAngles(self.angles)
                if not self.safety_check(max_tau=max_tau, sim_torques=arm_env.getInternalTorques('npARY')):
                    return False
            self.unsubscribe_notification_handle()
        return True

    def joint_move(self, joint_angles):
        print("Starting angular action movement ...")
        action = Base_pb2.Action()
        action.name = "Example angular action movement"
        action.application_data = ""

        actuator_count = self.base_client.GetActuatorCount()

        # Place arm straight up
        for joint_id in range(actuator_count.count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = np.rad2deg(joint_angles[joint_id])

        e = threading.Event()
        self.notification_handle = self.base_client.OnNotificationActionTopic(
            check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        
        print("Executing action")
        self.base_client.ExecuteAction(action)
        return e

    def persistent_joint_move(self, target_angles, max_time=30.0, max_tau=20.0):
        start_time = time.time()
        try:
            e = self.joint_move(target_angles)
            while not e.isSet():
                time.sleep(0.01)
                if time.time()-start_time > max_time:
                    return False
                if not self.safety_check(max_tau=max_tau):
                    raise RuntimeError(f"Maximum torque {max_tau:.3f}Nm reached!")
            return True
        finally:
            self.unsubscribe_notification_handle()

    def waypoint_move(self, trajectory, duration=1):
        # Copy from kinova codebase
        print("Starting waypoint movement ...")
        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0
        waypoints.use_optimal_blending = False

        def populateAngularPose(jointPose, durationFactor):
            waypoint = Base_pb2.AngularWaypoint()
            waypoint.angles.extend(jointPose)
            waypoint.duration = durationFactor
            return waypoint

        for i, q in enumerate(trajectory):
            q = np.rad2deg(q)
            for j in range(7):
                if q[j] < 0:
                    q[j] += 360
            waypoint = waypoints.waypoints.add()
            waypoint.name = f"waypoint_{i}"
            durationFactor = duration
            waypoint.angular_waypoint.CopyFrom(populateAngularPose(q, durationFactor))

        res = self.base_client.ValidateWaypointList(waypoints)
        print(res)
        print(res.trajectory_error_report)
        if len(res.trajectory_error_report.trajectory_error_elements) == 0:
            e = threading.Event()
            self.notification_handle = self.base_client.OnNotificationActionTopic(
                check_for_end_or_abort(e), Base_pb2.NotificationOptions()
            )
        
            print("Executing action")
            self.base_client.ExecuteWaypointTrajectory(waypoints)
            return e
        return None

    def persistent_waypoint_move(self, trajectory, max_time=30.0, max_tau=30.0, error_cb=None, duration=1):
        start_time = time.time()
        try:
            e = self.waypoint_move(trajectory, duration=duration)
            while not e.isSet():
                time.sleep(0.01)
                if error_cb is None:
                    if time.time()-start_time > max_time:
                        self.stop_action()
                        return False
                    if not self.safety_check(max_tau=max_tau):
                        raise RuntimeError(f"Maximum torque {max_tau:.3f}Nm reached!")
                else:
                    if not error_cb():
                        self.stop_action()
                        return False

            return True
        finally:
            self.unsubscribe_notification_handle()

    
    def forward_kinematics(self, angles, max_try=5):
        angles = angles_rad2deg(angles)
        ang_msg = self.base_client.GetMeasuredJointAngles()
        for angle, in_val in zip(ang_msg.joint_angles, angles):
            angle.value = in_val

        # Computing Foward Kinematics (Angle -> cartesian convert) from arm's current joint angles
        for _ in range(max_try):
            try:
                pose = self.base_client.ComputeForwardKinematics(ang_msg)
                return [pose.x, pose.y, pose.z, pose.theta_x, pose.theta_y, pose.theta_z]
            except KServerException as ex:
                print("Unable to compute forward kinematics")
                print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
                print("Caught expected error: {}".format(ex))
                time.sleep(0.5)
                continue
        
        raise RuntimeError("Failed to compute forward kinematics")

    def unsubscribe_notification_handle(self):
        self.base_client.Unsubscribe(self.notification_handle)
