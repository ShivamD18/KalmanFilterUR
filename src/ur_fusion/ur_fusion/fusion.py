"""
Three Kalman filters run side by side so their uncertainty can be compared.

* fused            - updated by both the ultrasonic and the IR readings
* ultrasonic_only  - updated by the ultrasonic readings only
* infrared_only    - updated by the IR readings only

All three share the same motion model and the same sensor noise models, so
any difference in their uncertainty comes purely from the data they receive.

Every reading, from either sensor, produces an estimate from *all three*
filters at the reading's timestamp. A filter that does not use that sensor
just runs its prediction step, so the three uncertainties can always be
compared at identical instants.
"""

from dataclasses import dataclass
import heapq
import itertools
import math

from .kalman import ConstantVelocityKF, UpdateResult

FILTER_SENSORS = {
    'fused': ('ultrasonic', 'infrared'),
    'ultrasonic_only': ('ultrasonic',),
    'infrared_only': ('infrared',),
}


@dataclass
class FusionConfig:
    accel_psd: float = 0.005              # [m^2/s^3] process noise of the CV model
    gate_sigma: float = 3.0               # reject readings > gate_sigma away from prediction
    initial_velocity_std: float = 0.5     # [m/s] velocity uncertainty at start-up
    max_consecutive_rejections: int = 5   # then assume the track is lost and restart
    ultrasonic_noise_scale: float = 1.0   # multiplies R; != 1 models a mis-tuned filter
    infrared_noise_scale: float = 1.0


@dataclass
class Estimate:
    filter_name: str
    sensor: str                  # sensor whose reading triggered this estimate
    stamp: float                 # [s]
    distance: float              # [m]
    distance_variance: float     # [m^2]
    range_rate: float            # [m/s]
    range_rate_variance: float   # [(m/s)^2]
    measurement: float           # [m] raw reading (nan if this filter ignores the sensor)
    measurement_variance: float  # [m^2] R used for it (nan when not used)
    innovation: float            # [m]
    innovation_variance: float   # [m^2]
    nis: float
    gain: float
    used: bool                   # the reading was fused into this filter
    rejected: bool               # the reading was gated out as an outlier


class TrackedFilter:
    """
    One Kalman filter plus the bookkeeping around it.

    It initialises itself from the first valid reading, picks R for every
    reading from the sensor model, gates outliers, and restarts if it keeps
    rejecting readings (which means it lost the target, not that every
    sensor suddenly went bad).
    """

    def __init__(self, name: str, sensors: tuple, models: dict, config: FusionConfig):
        self.name = name
        self.sensors = sensors
        self.models = models
        self.config = config
        self.noise_scale = {
            'ultrasonic': config.ultrasonic_noise_scale,
            'infrared': config.infrared_noise_scale,
        }
        self.kf = ConstantVelocityKF(config.accel_psd)
        self.consecutive_rejections = 0
        self.rejected = 0
        self.restarts = 0

    def measurement_variance(self, sensor: str, distance: float) -> float:
        """R for a reading taken at `distance` (clamped to the rated range)."""
        model = self.models[sensor]
        distance = min(max(distance, model.min_range), model.max_range)
        return self.noise_scale[sensor] * model.sigma(distance) ** 2

    def step(self, stamp: float, sensor: str, z: float) -> Estimate | None:
        if not self.kf.initialized:
            if not math.isfinite(z):
                return None
            self._restart(stamp, sensor, z)
            r = self.kf.distance_variance
            return self._estimate(sensor, z, r, UpdateResult(0.0, r, 0.0, 1.0, True))

        if not math.isfinite(z):
            # Out of range: nothing to fuse, but time passed, so the prediction
            # step alone grows the uncertainty.
            return self.predict_only(stamp, sensor, z)

        self.kf.predict(stamp)

        # Evaluate R at the predicted distance rather than at z, otherwise
        # readings that happen to come out short would get more weight.
        r = self.measurement_variance(sensor, self.kf.distance)
        result = self.kf.update(z, r, gate=self.config.gate_sigma ** 2)
        if result.accepted:
            self.consecutive_rejections = 0
        else:
            self.rejected += 1
            self.consecutive_rejections += 1
            if self.consecutive_rejections >= self.config.max_consecutive_rejections:
                self._restart(stamp, sensor, z)
                self.restarts += 1
        return self._estimate(sensor, z, r, result, rejected=not result.accepted)

    def predict_only(self, stamp: float, sensor: str, z: float = math.nan) -> Estimate | None:
        """Time update without a measurement."""
        if not self.kf.initialized:
            return None
        self.kf.predict(stamp)
        nan = math.nan
        return self._estimate(sensor, z, nan, UpdateResult(nan, nan, nan, 0.0, False))

    def _restart(self, stamp: float, sensor: str, z: float):
        self.kf.initialize(z, self.measurement_variance(sensor, z),
                           self.config.initial_velocity_std ** 2, stamp)
        self.consecutive_rejections = 0

    def _estimate(self, sensor: str, z: float, r: float, result: UpdateResult,
                  rejected: bool = False) -> Estimate:
        kf = self.kf
        return Estimate(
            filter_name=self.name, sensor=sensor, stamp=kf.time,
            distance=kf.distance, distance_variance=kf.distance_variance,
            range_rate=kf.velocity, range_rate_variance=kf.velocity_variance,
            measurement=z, measurement_variance=r,
            innovation=result.innovation, innovation_variance=result.innovation_variance,
            nis=result.nis, gain=result.gain, used=result.accepted, rejected=rejected,
        )


class DistanceFusion:
    """Feeds every reading to the filters that use it and advances the others."""

    def __init__(self, models: dict, config: FusionConfig):
        self.models = models
        self.config = config
        self.filters = {name: TrackedFilter(name, sensors, models, config)
                        for name, sensors in FILTER_SENSORS.items()}

    def process(self, stamp: float, sensor: str, z: float) -> list[Estimate]:
        estimates = []
        for kf in self.filters.values():
            if sensor in kf.sensors:
                estimate = kf.step(stamp, sensor, z)
            else:
                estimate = kf.predict_only(stamp, sensor)
            if estimate is not None:
                estimates.append(estimate)
        return estimates

    def reset(self):
        self.__init__(self.models, self.config)


class ReorderBuffer:
    """
    Releases items in timestamp order, `latency` seconds after they were taken.

    The two sensors arrive on different topics at different rates, so their
    messages can be delivered slightly out of order. A Kalman filter can only
    move forward in time, so each reading is held back briefly and released
    sorted by its timestamp.
    """

    def __init__(self, latency: float):
        self.latency = latency
        self.reset()

    def reset(self):
        self._heap = []
        self._order = itertools.count()
        self.newest = -math.inf
        self.released_until = -math.inf
        self.dropped = 0

    def push(self, stamp: float, item) -> list[tuple[float, object]]:
        if stamp < self.released_until:
            self.dropped += 1   # too late: newer readings were already fused
            return []
        heapq.heappush(self._heap, (stamp, next(self._order), item))
        self.newest = max(self.newest, stamp)
        released = []
        while self._heap and self._heap[0][0] <= self.newest - self.latency:
            stamp, _, item = heapq.heappop(self._heap)
            self.released_until = stamp
            released.append((stamp, item))
        return released
