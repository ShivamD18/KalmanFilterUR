import pytest

from ur_fusion.trajectory import minimum_jerk, SweepProfile


def test_minimum_jerk_starts_and_ends_at_rest():
    assert minimum_jerk(0.0) == (0.0, 0.0)
    assert minimum_jerk(1.0) == (1.0, 0.0)
    position, rate = minimum_jerk(0.5)
    assert position == pytest.approx(0.5)
    assert rate == pytest.approx(1.875)


def test_sweep_visits_each_waypoint_at_rest():
    profile = SweepProfile([0.90, 0.15, 0.45, 0.90], average_speed=0.08, dwell_time=3.0)
    assert profile.duration == pytest.approx(3 * 3.0 + 1.5 / 0.08)
    for waypoint, t in zip(profile.waypoints, profile.arrival_times):
        distance, rate = profile.sample(t)
        assert distance == pytest.approx(waypoint)
        assert rate == pytest.approx(0.0, abs=1e-12)
