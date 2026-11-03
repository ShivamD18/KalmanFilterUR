"""
Move the sensor bracket towards and away from the object, looking straight at it.

The scan visits a list of distances (default 0.90 -> 0.15 -> 0.45 -> 0.90 m)
with minimum-jerk moves and a pause at each. Every waypoint is turned into joint
angles with inverse kinematics on the robot's own URDF, the whole scan is sent
to the joint_trajectory_controller as one trajectory, and then it repeats.
"""

import math

from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import ListControllers
import numpy as np
import rclpy
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .kinematics import make_transform, rpy_to_matrix, SerialChain
from .trajectory import SweepProfile


def to_duration(seconds: float) -> Duration:
    sec = math.floor(seconds)
    return Duration(sec=int(sec), nanosec=int(round((seconds - sec) * 1e9)))


class ScanMotionNode(Node):

    def __init__(self):
        super().__init__('scan_motion_node')
        param = self.declare_parameter
        self.base_frame = param('base_frame', 'base_link').value
        self.sensor_frame = param('sensor_frame', 'sensor_boresight').value
        # Point on the object the sensors look at, and the look direction (x axis of
        # look_rpy), both in base_frame. The sensors sit `distance` back along it.
        self.surface_point = np.array(param('surface_point', [0.55, 0.0, -0.45]).value)
        self.look_rotation = rpy_to_matrix(*param('look_rpy', [0.0, math.pi / 2, 0.0]).value)
        self.profile = SweepProfile(list(param('distances', [0.90, 0.15, 0.45, 0.90]).value),
                                    param('average_speed', 0.08).value,
                                    param('dwell_time', 3.0).value)
        self.move_to_start_time = param('move_to_start_time', 5.0).value
        self.waypoint_period = param('waypoint_period', 0.1).value
        self.loop = param('loop', True).value
        self.controller = param('controller', 'joint_trajectory_controller').value

        self.urdf = None
        self.joint_state = None
        self.plan = None
        self.trajectory_end = None
        self.controllers_future = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, 'robot_description', self.on_description, latched)
        self.create_subscription(JointState, 'joint_states', self.on_joint_state, 10)
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, f'{self.controller}/joint_trajectory', 10)
        self.list_controllers = self.create_client(
            ListControllers, 'controller_manager/list_controllers')
        self.create_timer(0.5, self.tick)

    def on_description(self, msg: String):
        self.urdf = msg.data

    def on_joint_state(self, msg: JointState):
        self.joint_state = msg

    def tick(self):
        if self.plan is None:
            if self.urdf is None or self.joint_state is None or not self.controller_active():
                self.get_logger().info('Waiting for the robot and its trajectory controller ...',
                                       throttle_duration_sec=5.0)
                return
            try:
                self.plan = self.compute_plan()
            except (RuntimeError, ValueError, KeyError) as error:
                self.get_logger().error(f'Cannot plan the scan: {error}')
                self.loop = False
                self.plan = ()
                return
            self.send(first=True)
        elif self.plan and self.loop and self.get_clock().now() >= self.trajectory_end:
            self.send(first=False)

    def controller_active(self) -> bool:
        """Poll the controller manager (without blocking) until the controller is active."""
        future = self.controllers_future
        if future is not None and future.done():
            self.controllers_future = None
            result = future.result()
            if result is not None and any(c.name == self.controller and c.state == 'active'
                                          for c in result.controller):
                return True
        if self.controllers_future is None and self.list_controllers.service_is_ready():
            self.controllers_future = self.list_controllers.call_async(ListControllers.Request())
        return False

    def compute_plan(self):
        chain = SerialChain.from_urdf(self.urdf, self.base_frame, self.sensor_frame)
        current = dict(zip(self.joint_state.name, self.joint_state.position))
        q = np.array([current[name] for name in chain.joint_names])
        look = self.look_rotation[:, 0]

        times = np.arange(0.0, self.profile.duration, self.waypoint_period)
        times = np.append(times, self.profile.duration)
        positions, velocities = [], []
        for t in times:
            distance, rate = self.profile.sample(t)
            target = make_transform(self.look_rotation, self.surface_point - distance * look)
            result = chain.inverse(target, q)   # warm start from the previous waypoint
            if not result.success:
                raise RuntimeError(
                    f'no IK solution {distance:.2f} m from the object (off by '
                    f'{1000 * result.position_error:.1f} mm); is surface_point reachable?')
            q = result.joint_positions
            # Joint velocities that move the sensors along the look axis: J qdot = twist
            twist = np.concatenate([-rate * look, np.zeros(3)])
            q_dot = np.linalg.lstsq(chain.jacobian(q), twist, rcond=None)[0]
            positions.append(q.copy())
            velocities.append(q_dot)

        self.get_logger().info(
            f'Planned a {self.profile.duration:.1f} s scan through '
            f'{self.profile.waypoints} m ({len(times)} waypoints)')
        return chain.joint_names, times, positions, velocities

    def send(self, first: bool):
        names, times, positions, velocities = self.plan
        msg = JointTrajectory()
        msg.joint_names = names
        if first:
            # Glide from wherever the arm is to the first scan pose.
            start = JointTrajectoryPoint(positions=list(positions[0]),
                                         velocities=[0.0] * len(names),
                                         time_from_start=to_duration(self.move_to_start_time))
            msg.points.append(start)
            offset, first_index = self.move_to_start_time, 1
        else:
            offset, first_index = self.waypoint_period, 0
        for t, q, q_dot in zip(times[first_index:], positions[first_index:],
                               velocities[first_index:]):
            msg.points.append(JointTrajectoryPoint(
                positions=list(q), velocities=list(q_dot),
                time_from_start=to_duration(offset + t)))

        self.trajectory_pub.publish(msg)
        total = offset + times[-1]
        self.trajectory_end = self.get_clock().now() + RclpyDuration(seconds=total)
        if first:
            self.get_logger().info('Scanning started')


def main(args=None):
    try:
        with rclpy.init(args=args):
            rclpy.spin(ScanMotionNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
