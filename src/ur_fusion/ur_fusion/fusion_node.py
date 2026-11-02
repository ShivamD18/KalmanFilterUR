"""
Kalman-filter the ultrasonic and IR readings: each one alone, and fused.

Subscribes  ultrasonic/range, infrared/range                 (sensor_msgs/Range)
Publishes   fusion/fused, fusion/ultrasonic_only,
            fusion/infrared_only               (ur_fusion_interfaces/DistanceEstimate)
"""

from functools import partial
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Range
from ur_fusion_interfaces.msg import DistanceEstimate

from .fusion import DistanceFusion, Estimate, FILTER_SENSORS, FusionConfig, ReorderBuffer
from .ros_params import declare_dataclass, declare_sensor_models, stamp_to_seconds

CLOCK_JUMP = 1.0  # [s] a reading this much older than the newest one means the sim was reset


class FusionNode(Node):

    def __init__(self):
        super().__init__('fusion_node')
        models = declare_sensor_models(self)
        config = declare_dataclass(self, FusionConfig)
        self.fusion = DistanceFusion(models, config)
        self.buffer = ReorderBuffer(self.declare_parameter('reorder_latency', 0.05).value)
        self.frame_id = self.declare_parameter('estimate_frame', 'sensor_boresight').value
        self.latest = {}

        self.estimate_pubs = {name: self.create_publisher(DistanceEstimate, f'fusion/{name}', 10)
                              for name in FILTER_SENSORS}
        for sensor in models:
            self.create_subscription(Range, f'{sensor}/range', partial(self.on_range, sensor), 50)
        self.create_timer(5.0, self.log_status)
        self.get_logger().info(
            f'Kalman filters ready: process noise q={config.accel_psd} m^2/s^3, '
            f'outlier gate {config.gate_sigma} sigma')

    def on_range(self, sensor: str, msg: Range):
        stamp = stamp_to_seconds(msg.header.stamp)
        if stamp < self.buffer.newest - CLOCK_JUMP:
            self.get_logger().warning('Time jumped backwards (simulation reset?), '
                                      'restarting the filters')
            self.buffer.reset()
            self.fusion.reset()
        for stamp, (sensor, reading, header_stamp) in self.buffer.push(
                stamp, (sensor, msg.range, msg.header.stamp)):
            for estimate in self.fusion.process(stamp, sensor, reading):
                self.latest[estimate.filter_name] = estimate
                self.estimate_pubs[estimate.filter_name].publish(
                    self.to_msg(estimate, header_stamp))

    def to_msg(self, est: Estimate, stamp) -> DistanceEstimate:
        msg = DistanceEstimate()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.filter_name = est.filter_name
        msg.trigger_sensor = est.sensor
        msg.distance = est.distance
        msg.distance_variance = est.distance_variance
        msg.range_rate = est.range_rate
        msg.range_rate_variance = est.range_rate_variance
        msg.measurement = est.measurement
        msg.measurement_variance = est.measurement_variance
        msg.innovation = est.innovation
        msg.innovation_variance = est.innovation_variance
        msg.nis = est.nis
        msg.kalman_gain = est.gain
        msg.measurement_used = est.used
        msg.measurement_rejected = est.rejected
        return msg

    def log_status(self):
        if 'fused' not in self.latest:
            self.get_logger().info('Waiting for ultrasonic/range and infrared/range ...')
            return
        parts = []
        for name in FILTER_SENSORS:
            est = self.latest.get(name)
            if est is not None:
                kf = self.fusion.filters[name]
                parts.append(f'{name} {est.distance:.3f} m ± '
                             f'{1000 * math.sqrt(est.distance_variance):.1f} mm '
                             f'({kf.rejected} gated)')
        self.get_logger().info(' | '.join(parts))


def main(args=None):
    try:
        with rclpy.init(args=args):
            rclpy.spin(FusionNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
