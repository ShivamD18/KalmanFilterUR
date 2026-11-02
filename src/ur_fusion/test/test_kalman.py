import numpy as np
import pytest

from ur_fusion.kalman import ConstantVelocityKF, inverse_variance_fusion


def test_static_fusion_is_more_certain_than_either_reading():
    mean, var = inverse_variance_fusion([1.00, 1.02], [0.01 ** 2, 0.02 ** 2])
    assert var < 0.01 ** 2
    assert var == pytest.approx(1.0 / (1.0 / 0.01 ** 2 + 1.0 / 0.02 ** 2))
    assert mean == pytest.approx(1.004)   # pulled towards the more precise reading


def test_kalman_update_reproduces_static_fusion():
    """With the first reading as the prior, one update is inverse-variance weighting."""
    kf = ConstantVelocityKF(accel_psd=0.0)
    kf.initialize(1.00, 0.01 ** 2, velocity_variance=1.0, time=0.0)
    kf.update(1.02, 0.02 ** 2)
    mean, var = inverse_variance_fusion([1.00, 1.02], [0.01 ** 2, 0.02 ** 2])
    assert kf.distance == pytest.approx(mean)
    assert kf.distance_variance == pytest.approx(var)


def test_predict_grows_and_update_shrinks_uncertainty():
    kf = ConstantVelocityKF(accel_psd=0.01)
    kf.initialize(0.5, 1e-4, 0.25, time=0.0)
    for k in range(1, 50):
        before = kf.distance_variance
        kf.predict(k * 0.04)
        predicted = kf.distance_variance
        kf.update(0.5, 1e-4)
        assert predicted >= before
        assert kf.distance_variance <= predicted


def test_gate_rejects_an_outlier_without_touching_the_state():
    kf = ConstantVelocityKF(accel_psd=0.01)
    kf.initialize(0.5, 1e-4, 0.01, time=0.0)
    x, p = kf.x.copy(), kf.P.copy()
    result = kf.update(0.9, 1e-4, gate=9.0)
    assert not result.accepted
    assert result.nis > 9.0
    np.testing.assert_array_equal(kf.x, x)
    np.testing.assert_array_equal(kf.P, p)


def test_filter_is_consistent_on_a_constant_velocity_target():
    """Monte Carlo check: the mean NEES of a correctly modelled filter is about 1."""
    rng = np.random.default_rng(1)
    sigma, dt = 0.01, 0.05
    nees = []
    for _ in range(300):
        kf = ConstantVelocityKF(accel_psd=1e-8)
        kf.initialize(1.0 + rng.normal(0, sigma), sigma ** 2, 0.2 ** 2, time=0.0,
                      velocity=-0.1 + rng.normal(0, 0.2))
        for k in range(1, 80):
            truth = 1.0 - 0.1 * k * dt
            kf.predict(k * dt)
            kf.update(truth + rng.normal(0, sigma), sigma ** 2)
        nees.append((kf.distance - truth) ** 2 / kf.distance_variance)
    assert 0.8 < np.mean(nees) < 1.25
