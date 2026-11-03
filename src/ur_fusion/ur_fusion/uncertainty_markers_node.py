"""
Show each filter's estimate in RViz as a slab that is +-2 sigma thick.

The three slabs hang side by side at the estimated distance along the sensor
boresight, next to a thin dark plate at the true distance (the object's
surface). The thinner the slab, the more certain the filter. A column of
labels beside them gives each filter's 2-sigma in millimetres.
"""

from functools import partial
import math

from builtin_interfaces.msg import Duration
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Range
from ur_fusion_interfaces.msg import DistanceEstimate
from visualization_msgs.msg import Marker, MarkerArray

FILTERS = {
    # filter: (offset along the boresight y axis [m], RGB, label)
    'ultrasonic_only': (0.11, (0.922, 0.408, 0.204), 'Ultrasonic only'),
    'fused': (0.0, (0.165, 0.471, 0.839), 'Fused (US + IR)'),
    'infrared_only': (-0.11, (0.106, 0.686, 0.478), 'IR only'),
}
SLAB_SIZE = 0.08        # [m] footprint of each slab
MIN_THICKNESS = 0.001   # [m] keep very certain estimates visible
MAX_THICKNESS = 0.40    # [m] keep a lost filter from filling the screen
LABEL_COLUMN = -0.45    # [m] labels sit in a column this far along boresight y
LINE_SPACING = 0.045    # [m] between label lines
TEXT_HEIGHT = 0.03      # [m]


class UncertaintyMarkersNode(Node):

    def __init__(self):
        super().__init__('uncertainty_markers_node')
        self.latest = {}
        self.truth = None
        for name in FILTERS:
            self.create_subscription(DistanceEstimate, f'fusion/{name}',
                                     partial(self.on_estimate, name), 10)
        self.create_subscription(Range, 'ground_truth/range', self.on_truth, 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'fusion/markers', 10)
        self.create_timer(0.1, self.publish)

    def on_estimate(self, name: str, msg: DistanceEstimate):
        self.latest[name] = msg

    def on_truth(self, msg: Range):
        self.truth = msg

    def publish(self):
        markers = MarkerArray()
        anchor = self.label_anchor()
        if anchor is not None:
            markers.markers.append(self.label(
                -2, anchor, '2-sigma uncertainty', (1.0, 1.0, 1.0)))
        for i, (name, (offset, rgb, label)) in enumerate(FILTERS.items()):
            est = self.latest.get(name)
            if est is None:
                continue
            two_sigma = 2.0 * math.sqrt(est.distance_variance)
            thickness = min(max(2.0 * two_sigma, MIN_THICKNESS), MAX_THICKNESS)
            slab = self.marker(est.header.frame_id, 'estimate', i, Marker.CUBE, rgb, 0.6)
            slab.pose.position.x = est.distance
            slab.pose.position.y = offset
            slab.scale.x, slab.scale.y, slab.scale.z = thickness, SLAB_SIZE, SLAB_SIZE
            markers.markers.append(slab)
            # Plain ASCII: RViz's text renderer has no glyphs for symbols like
            # plus-minus or sigma and garbles the whole line around them.
            if anchor is not None:
                markers.markers.append(self.label(
                    i - 1, anchor, f'{label}: +/- {1000 * two_sigma:.1f} mm', rgb))

        if self.truth is not None and math.isfinite(self.truth.range):
            plate = self.marker(self.truth.header.frame_id, 'truth', 0, Marker.CUBE,
                                (0.15, 0.15, 0.15), 0.9)
            plate.pose.position.x = self.truth.range
            plate.scale.x, plate.scale.y, plate.scale.z = 0.003, 0.34, 0.10
            markers.markers.append(plate)
        self.marker_pub.publish(markers)

    def label_anchor(self) -> tuple[str, float] | None:
        """Frame and distance the label column is centred on."""
        if self.truth is not None and math.isfinite(self.truth.range):
            return self.truth.header.frame_id, self.truth.range
        fused = self.latest.get('fused')
        return (fused.header.frame_id, fused.distance) if fused else None

    def label(self, line: int, anchor, text: str, rgb) -> Marker:
        """One line of the label column; line 0 is level with the anchor."""
        frame_id, distance = anchor
        marker = self.marker(frame_id, 'label', line + 2, Marker.TEXT_VIEW_FACING, rgb, 1.0)
        marker.pose.position.x = distance + line * LINE_SPACING
        marker.pose.position.y = LABEL_COLUMN
        marker.scale.z = TEXT_HEIGHT
        marker.text = text
        return marker

    @staticmethod
    def marker(frame_id: str, ns: str, marker_id: int, marker_type: int, rgb, alpha: float):
        marker = Marker()
        marker.header.frame_id = frame_id   # zero stamp: use the latest transform
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b = rgb
        marker.color.a = alpha
        marker.lifetime = Duration(sec=1)
        return marker


def main(args=None):
    try:
        with rclpy.init(args=args):
            rclpy.spin(UncertaintyMarkersNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
