"""
Live dashboard: how much less uncertain the fused estimate is than each sensor alone.

Draws the same figure as the offline demo (see plotting.py) from
ground_truth/range, ultrasonic/range, infrared/range and fusion/*.

In the window: press `s` to save a PNG, `r` to reset the statistics.
With headless:=true it draws off-screen and saves a PNG every save_period seconds.
"""

from functools import partial
import os
import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Range
from ur_fusion_interfaces.msg import DistanceEstimate

from .fusion import FILTER_SENSORS
from .plotting import FusionFigure, FusionRecorder
from .ros_params import declare_sensor_models, stamp_to_seconds

TITLE = 'UR arm with ultrasonic + IR sensors: Kalman filter fusion (live)'


class DashboardNode(Node):

    def __init__(self):
        super().__init__('dashboard_node')
        self.models = declare_sensor_models(self)
        self.headless = self.declare_parameter('headless', False).value
        self.save_period = self.declare_parameter('save_period', 10.0).value
        output_dir = self.declare_parameter('output_dir', '').value
        self.output_dir = output_dir or os.path.join(os.path.expanduser('~'), '.ros', 'ur_fusion')
        window = self.declare_parameter('window', 40.0).value
        self.lock = threading.Lock()
        self.recorder = FusionRecorder(self.models, window=window)
        self.last_stamp = None

        self.create_subscription(Range, 'ground_truth/range', self.on_truth, 50)
        for sensor in self.models:
            self.create_subscription(Range, f'{sensor}/range',
                                     partial(self.on_reading, sensor), 50)
        for name in FILTER_SENSORS:
            self.create_subscription(DistanceEstimate, f'fusion/{name}',
                                     partial(self.on_estimate, name), 50)

    def stamp(self, header) -> float:
        t = stamp_to_seconds(header.stamp)
        if self.last_stamp is not None and t < self.last_stamp - 1.0:
            self.recorder.reset()   # simulation was reset
        self.last_stamp = t
        return t

    def on_truth(self, msg: Range):
        with self.lock:
            self.recorder.add_truth(self.stamp(msg.header), msg.range)

    def on_reading(self, sensor: str, msg: Range):
        with self.lock:
            self.recorder.add_reading(sensor, self.stamp(msg.header), msg.range)

    def on_estimate(self, name: str, msg: DistanceEstimate):
        with self.lock:
            self.recorder.add_estimate(name, self.stamp(msg.header), msg.distance,
                                       msg.distance_variance, msg.measurement_rejected)

    def draw(self, view: FusionFigure):
        with self.lock:
            self.recorder.update()
            view.draw(self.recorder)

    def save(self, figure, name: str = 'fusion_dashboard.png') -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, name)
        figure.savefig(path, dpi=110, facecolor=figure.get_facecolor())
        return path

    def reset(self):
        with self.lock:
            self.recorder.reset()


def main(args=None):
    try:
        with rclpy.init(args=args):
            run(DashboardNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


def run(node: DashboardNode):
    import matplotlib
    if node.headless:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    figure = plt.figure(figsize=(13, 8.5))
    view = FusionFigure(figure, node.models, TITLE)

    if node.headless:
        def save_snapshot():
            node.draw(view)
            node.save(figure)

        node.create_timer(node.save_period, save_snapshot)
        node.get_logger().info(f'Saving the dashboard to {node.output_dir} '
                               f'every {node.save_period:.0f} s')
        rclpy.spin(node)
        return

    def spin():
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass

    # ROS callbacks run in the background; matplotlib owns the main thread.
    threading.Thread(target=spin, daemon=True).start()

    def update(_frame):
        if not rclpy.ok():
            plt.close(figure)
            return
        node.draw(view)

    def on_key(event):
        if event.key == 's':
            node.get_logger().info(f'Saved {node.save(figure)}')
        elif event.key == 'r':
            node.reset()

    figure.canvas.manager.set_window_title('UR sensor fusion dashboard')
    figure.canvas.mpl_connect('key_press_event', on_key)
    animation = FuncAnimation(figure, update, interval=300, cache_frame_data=False)  # noqa: F841
    plt.show()
