import math

import numpy as np
import pytest

from ur_fusion.fusion import DistanceFusion, FILTER_SENSORS, FusionConfig, ReorderBuffer
from ur_fusion.offline_demo import simulate
from ur_fusion.sensor_models import InfraredModel, UltrasonicModel
from ur_fusion.trajectory import SweepProfile


@pytest.fixture(scope='module')
def run():
    models = {'ultrasonic': UltrasonicModel(), 'infrared': InfraredModel()}
    profile = SweepProfile([0.90, 0.15, 0.45, 0.90], 0.08, 3.0)
    return simulate(models, FusionConfig(), profile, 2 * profile.duration, seed=3)


def test_fused_filter_is_more_accurate_than_either_sensor_alone(run):
    recorder, _ = run
    stats = {name: recorder.filter_stats(name) for name in FILTER_SENSORS}
    assert stats['fused']['rmse'] < stats['ultrasonic_only']['rmse']
    assert stats['fused']['rmse'] < stats['infrared_only']['rmse']


def test_filters_report_honest_uncertainty(run):
    recorder, _ = run
    for name in FILTER_SENSORS:
        assert 0.5 < recorder.filter_stats(name)['nees'] < 1.6, name


def test_fused_uncertainty_is_never_above_the_best_single_sensor(run):
    recorder, _ = run
    sigma = {}
    for name in FILTER_SENSORS:
        est = recorder.estimate_array(name)
        sigma[name] = dict(zip(est[:, 0], np.sqrt(est[:, 2])))
    common = set(sigma['fused']) & set(sigma['ultrasonic_only']) & set(sigma['infrared_only'])
    assert len(common) > 1000
    best_single = {t: min(sigma['ultrasonic_only'][t], sigma['infrared_only'][t])
                   for t in common}
    worse = [t for t in common if sigma['fused'][t] > 1.01 * best_single[t]]
    assert len(worse) / len(common) < 0.01
    average, _ = recorder.uncertainty_reduction()
    assert average > 0.0


def test_outliers_are_gated(run):
    recorder, fusion = run
    assert recorder.outliers['ultrasonic'] > 0
    assert fusion.filters['ultrasonic_only'].rejected >= 0.8 * recorder.outliers['ultrasonic']


def test_filter_restarts_after_losing_the_target():
    models = {'ultrasonic': UltrasonicModel(), 'infrared': InfraredModel()}
    fusion = DistanceFusion(models, FusionConfig())
    t = 0.0
    for _ in range(30):
        t += 0.05
        fusion.process(t, 'ultrasonic', 0.50)
    for _ in range(10):     # the object suddenly moved
        t += 0.05
        estimates = fusion.process(t, 'ultrasonic', 0.20)
    fused = next(e for e in estimates if e.filter_name == 'fused')
    assert fusion.filters['fused'].restarts == 1
    assert fused.distance == pytest.approx(0.20, abs=0.02)


def test_filters_that_ignore_a_sensor_still_report_at_its_timestamp():
    models = {'ultrasonic': UltrasonicModel(), 'infrared': InfraredModel()}
    fusion = DistanceFusion(models, FusionConfig())
    fusion.process(0.00, 'ultrasonic', 0.5)
    fusion.process(0.01, 'infrared', 0.5)
    estimates = {e.filter_name: e for e in fusion.process(0.05, 'infrared', 0.5)}
    assert set(estimates) == set(FILTER_SENSORS)
    assert not estimates['ultrasonic_only'].used
    assert math.isnan(estimates['ultrasonic_only'].measurement)
    assert estimates['infrared_only'].used and estimates['fused'].used


def test_reorder_buffer_releases_in_time_order():
    buffer = ReorderBuffer(latency=0.05)
    released = []
    for stamp in [0.00, 0.02, 0.01, 0.04, 0.03, 0.10, 0.20]:
        released += [s for s, _ in buffer.push(stamp, None)]
    assert released == [0.00, 0.01, 0.02, 0.03, 0.04, 0.10]


def test_reorder_buffer_drops_readings_that_arrive_too_late():
    buffer = ReorderBuffer(latency=0.05)
    buffer.push(0.00, 'a')
    buffer.push(0.20, 'b')      # releases 'a'
    assert buffer.push(-0.01, 'late') == []
    assert buffer.dropped == 1
