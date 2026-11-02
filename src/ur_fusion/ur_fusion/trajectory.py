"""Distance profile of the scan: how far the sensors are from the object over time."""

from dataclasses import dataclass


def minimum_jerk(s: float) -> tuple[float, float]:
    """
    Minimum-jerk blend from 0 to 1 and its derivative, for s in [0, 1].

    It starts and ends with zero velocity and acceleration, so the arm moves
    smoothly and the constant-velocity Kalman model is only mildly violated.
    """
    s = min(max(s, 0.0), 1.0)
    return (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5,
            30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4)


@dataclass
class Segment:
    start_time: float
    duration: float
    start: float
    end: float


class SweepProfile:
    """Visit `waypoints` (distances [m]) in order: dwell at each, then glide to the next."""

    def __init__(self, waypoints: list[float], average_speed: float, dwell_time: float):
        if len(waypoints) < 2:
            raise ValueError('need at least two distance waypoints')
        if average_speed <= 0.0:
            raise ValueError('average_speed must be positive')
        self.waypoints = list(waypoints)
        self.dwell_time = dwell_time
        self.segments = []
        self.arrival_times = [0.0]
        t = 0.0
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            if dwell_time > 0.0:
                self.segments.append(Segment(t, dwell_time, start, start))
                t += dwell_time
            move_time = abs(end - start) / average_speed
            if move_time > 0.0:
                self.segments.append(Segment(t, move_time, start, end))
                t += move_time
            self.arrival_times.append(t)
        self.duration = t

    def sample(self, t: float) -> tuple[float, float]:
        """Distance [m] and its rate [m/s] at time t (clamped to the profile)."""
        t = min(max(t, 0.0), self.duration)
        for seg in self.segments:
            if t <= seg.start_time + seg.duration:
                blend, blend_rate = minimum_jerk((t - seg.start_time) / seg.duration)
                span = seg.end - seg.start
                return seg.start + span * blend, span * blend_rate / seg.duration
        return self.waypoints[-1], 0.0
