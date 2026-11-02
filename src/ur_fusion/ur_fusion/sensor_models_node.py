"""
Turn Gazebo's ideal ray casts into realistic ultrasonic and IR readings.

Gazebo ray-casts each beam on the sensor bracket and publishes exact
distances. This node plays the part of the sensor electronics:

  ideal/ultrasonic/scan    nearest echo in the cone  + UltrasonicModel -> ultrasonic/range
  ideal/infrared/scan      centroid of the IR spot   + InfraredModel   -> infrared/range
  ideal/ground_truth/scan  exact centre ray, no noise                  -> ground_truth/range
"""

from functools import partial
import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Range

from .ros_params import declare_sensor_models


def _reduce(ranges, reducer) -> float:
    """Collapse a beam's rays into one distance; +-inf if nothing usable (REP-117)."""
    ranges = np.asarray(ranges, dtype=float)
    if np.any(np.isneginf(ranges)):
        return -math.inf   # something sits closer than the beam's minimum range
    hits = ranges[np.isfinite(ranges)]
    return float(reducer(hits)) if hits.size else math.inf


def nearest_echo(ranges) -> float:
    """Sonar: the first echo comes from the closest surface anywhere in the cone."""
    return _reduce(ranges, np.min)


def spot_centroid(ranges) -> float:
    """IR triangulation: the sensor sees where its light spot lands, on average."""
    return _reduce(ranges, np.mean)


def centre_ray(ranges) -> float:
    return _reduce(ranges[len(ranges) // 2:][:1], np.min)


BEAMS = {
    # beam: (rays -> ideal distance, sensor_msgs/Range radiation type)
    'ultrasonic': (nearest_echo, Range.ULTRASOUND),
    'infrared': (spot_centroid, Range.INFRARED),
    'ground_truth': (centre_ray, Range.INFRARED),
}


class SensorModelsNode(Node):

    def __init__(self):
        super().__init__('sensor_models_node')
        self.models = declare_sensor_models(self)
        seed = self.declare_parameter('random_seed', -1).value
        self.rng = np.random.default_rng(None if seed < 0 else seed)
        self.frames = {
            'ultrasonic': self.declare_parameter('ultrasonic_frame', 'ultrasonic_link').value,
            'infrared': self.declare_parameter('infrared_frame', 'infrared_link').value,
            'ground_truth': self.declare_parameter('ground_truth_frame', 'sensor_boresight').value,
        }
        self.range_pubs = {beam: self.create_publisher(Range, f'{beam}/range', 10)
                           for beam in BEAMS}
        for beam in BEAMS:
            self.create_subscription(LaserScan, f'ideal/{beam}/scan',
                                     partial(self.on_scan, beam), qos_profile_sensor_data)
        self.get_logger().info(
            'Simulating sensors: ultrasonic sigma %.1f mm, IR sigma %.1f-%.1f mm (10-80 cm)' % (
                1000 * self.models['ultrasonic'].sigma(1.0),
                1000 * self.models['infrared'].sigma(0.10),
                1000 * self.models['infrared'].sigma(0.80)))

    def on_scan(self, beam: str, scan: LaserScan):
        reduce, radiation = BEAMS[beam]
        ideal = reduce(scan.ranges)
        msg = Range()
        msg.header.stamp = scan.header.stamp
        msg.header.frame_id = self.frames[beam]
        msg.radiation_type = radiation

        model = self.models.get(beam)
        if model is None:   # ground truth: a perfect ranger
            msg.range = ideal
            msg.min_range = scan.range_min
            msg.max_range = scan.range_max
        else:
            reading = model.measure(ideal, self.rng)
            msg.range = reading
            msg.field_of_view = model.field_of_view
            msg.min_range = model.min_range
            msg.max_range = model.max_range
            if math.isfinite(reading):
                msg.variance = model.sigma(reading) ** 2   # what the sensor thinks of itself
        self.range_pubs[beam].publish(msg)


def main(args=None):
    try:
        with rclpy.init(args=args):
            rclpy.spin(SensorModelsNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
