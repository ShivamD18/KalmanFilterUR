import math

import numpy as np
import pytest

from ur_fusion.sensor_models import InfraredModel, UltrasonicModel


@pytest.mark.parametrize('distance', [0.15, 0.4, 0.75])
def test_ir_noise_matches_its_sigma(distance):
    model = InfraredModel()
    rng = np.random.default_rng(0)
    readings = np.array([model.measure(distance, rng) for _ in range(5000)])
    assert np.std(readings) == pytest.approx(model.sigma(distance), rel=0.1)
    assert np.mean(readings) == pytest.approx(distance, abs=0.2 * model.sigma(distance))


@pytest.mark.parametrize('distance', [0.1, 1.0, 3.0])
def test_ultrasonic_noise_is_constant_and_matches_its_sigma(distance):
    model = UltrasonicModel(outlier_probability=0.0)
    rng = np.random.default_rng(0)
    readings = np.array([model.measure(distance, rng) for _ in range(5000)])
    assert np.std(readings) == pytest.approx(model.sigma(distance), rel=0.1)
    assert model.sigma(distance) == pytest.approx(model.sigma(0.5))


def test_sensors_are_complementary():
    us, ir = UltrasonicModel(), InfraredModel()
    assert ir.sigma(0.15) < us.sigma(0.15) / 5    # IR far better up close
    assert ir.sigma(0.75) > us.sigma(0.75) * 2    # sonar better far away


def test_out_of_range_readings_follow_rep117():
    rng = np.random.default_rng(0)
    ir, us = InfraredModel(), UltrasonicModel()
    assert ir.measure(0.05, rng) == -math.inf
    assert ir.measure(1.2, rng) == math.inf
    assert ir.measure(math.inf, rng) == math.inf
    assert us.measure(-math.inf, rng) == -math.inf
    assert us.measure(math.nan, rng) == math.inf


def test_ultrasonic_outlier_rate():
    model = UltrasonicModel(outlier_probability=0.05)
    rng = np.random.default_rng(0)
    readings = np.array([model.measure(0.5, rng) for _ in range(8000)])
    rate = np.mean(np.abs(readings - 0.5) > 5 * model.sigma(0.5))
    assert 0.035 < rate < 0.065


def test_wrong_temperature_biases_the_sonar():
    model = UltrasonicModel(outlier_probability=0.0, air_temperature=35.0,
                            assumed_temperature=20.0)
    rng = np.random.default_rng(0)
    readings = [model.measure(1.0, rng) for _ in range(2000)]
    assert np.mean(readings) < 0.98   # sound is faster in warm air: reads short
