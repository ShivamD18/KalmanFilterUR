"""
Run the sensing + fusion pipeline without ROS or Gazebo.

Replays the same approach/retreat sweep the robot performs, samples both
sensor models at their own rates, runs the three Kalman filters and draws
the dashboard figure. Handy for experimenting with the filter in seconds:

    ros2 run ur_fusion offline_demo --output fusion_offline.png
    ros2 run ur_fusion offline_demo --accel-psd 0.5 --show
"""

import argparse
import math

import numpy as np

from .fusion import DistanceFusion, FILTER_SENSORS, FusionConfig
from .plotting import FusionFigure, FusionRecorder
from .sensor_models import InfraredModel, UltrasonicModel
from .trajectory import SweepProfile

DEFAULT_WAYPOINTS = (0.90, 0.15, 0.45, 0.90)


def simulate(models: dict, config: FusionConfig, profile: SweepProfile, duration: float,
             seed: int, truth_rate: float = 50.0) -> tuple[FusionRecorder, DistanceFusion]:
    rng = np.random.default_rng(seed)

    def true_distance(t: float) -> float:
        return profile.sample(t % profile.duration)[0]

    events = [(t, 'truth') for t in np.arange(0.0, duration, 1.0 / truth_rate)]
    for sensor, model in models.items():
        period = 1.0 / model.update_rate
        phase = rng.uniform(0.0, period)
        events += [(t, sensor) for t in np.arange(phase, duration, period)]
    events.sort()

    fusion = DistanceFusion(models, config)
    recorder = FusionRecorder(models, window=duration)
    for t, source in events:
        d = true_distance(t)
        if source == 'truth':
            recorder.add_truth(t, d)
            continue
        z = models[source].measure(d, rng)
        recorder.add_reading(source, t, z)
        for est in fusion.process(t, source, z):
            recorder.add_estimate(est.filter_name, est.stamp, est.distance,
                                  est.distance_variance, est.rejected)
    recorder.update()
    return recorder, fusion


def print_summary(recorder: FusionRecorder, fusion: DistanceFusion):
    print(f'{"filter":<18}{"RMSE [mm]":>11}{"RMS σ [mm]":>12}{"NEES":>8}'
          f'{"gated":>8}{"restarts":>10}')
    for name in FILTER_SENSORS:
        s = recorder.filter_stats(name)
        print(f'{name:<18}{1000 * s["rmse"]:>11.2f}{1000 * s["rms_sigma"]:>12.2f}'
              f'{s["nees"]:>8.2f}{s["gated"]:>8d}{fusion.filters[name].restarts:>10d}')
    for sensor, model in recorder.models.items():
        noise = recorder.raw_noise(sensor)
        print(f'raw {sensor:<11} error std {1000 * noise:6.2f} mm over its range, '
              f'{recorder.outliers[sensor]} outliers '
              f'(model: {1000 * model.sigma(model.min_range):.1f}-'
              f'{1000 * model.sigma(min(model.max_range, 1.0)):.1f} mm)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default='fusion_offline.png', help='figure to write')
    parser.add_argument('--show', action='store_true', help='also open the figure in a window')
    parser.add_argument('--cycles', type=int, default=2,
                        help='full sweeps to simulate before stopping at the crossover dwell')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--speed', type=float, default=0.08, help='average speed [m/s]')
    parser.add_argument('--dwell', type=float, default=3.0, help='pause at each waypoint [s]')
    parser.add_argument('--accel-psd', type=float, default=FusionConfig.accel_psd)
    parser.add_argument('--gate-sigma', type=float, default=FusionConfig.gate_sigma)
    parser.add_argument('--outlier-probability', type=float,
                        default=UltrasonicModel.outlier_probability)
    args = parser.parse_args(argv)

    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = {
        'ultrasonic': UltrasonicModel(outlier_probability=args.outlier_probability),
        'infrared': InfraredModel(),
    }
    config = FusionConfig(accel_psd=args.accel_psd, gate_sigma=args.gate_sigma)
    profile = SweepProfile(list(DEFAULT_WAYPOINTS), args.speed, args.dwell)
    # Stop half-way through the dwell at 0.45 m, where both sensors are about
    # equally good, so the "right now" numbers show fusion at its best.
    duration = (args.cycles * profile.duration + profile.arrival_times[2]
                + 0.5 * profile.dwell_time)
    recorder, fusion = simulate(models, config, profile, duration, args.seed)
    print_summary(recorder, fusion)

    fig = plt.figure(figsize=(13, 8.5))
    FusionFigure(fig, models, 'Ultrasonic + IR distance fusion with a Kalman filter '
                 '(offline replay)').draw(recorder)
    fig.savefig(args.output, dpi=110, facecolor=fig.get_facecolor())
    print(f'figure written to {args.output}')
    if args.show:
        plt.show()
    return 0 if math.isfinite(recorder.filter_stats('fused')['rmse']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
