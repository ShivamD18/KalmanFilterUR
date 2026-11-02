"""
A two-state Kalman filter for distance, written out step by step.

State           x = [d, v]^T        distance to the object [m] and its rate [m/s]
Motion model    d' = d + v * dt     (constant velocity)
                v' = v + w          (w: white-noise acceleration, PSD q)
Measurement     z  = d + n          n ~ N(0, R); both sensors measure d directly

    predict:  x = F x              P = F P F^T + Q
    update:   nu = z - H x         S = H P H^T + R       (innovation, its variance)
              K  = P H^T / S                             (Kalman gain)
              x  = x + K nu        P = (I - K H) P (I - K H)^T + K R K^T

Every update can only shrink P, and the less noisy a sensor is (small R),
the larger its gain K and the more it shrinks P. Feeding the same filter
with two sensors therefore gives a smaller P than either sensor alone.
"""

from dataclasses import dataclass
import math

import numpy as np

H = np.array([[1.0, 0.0]])  # both sensors observe the distance state


@dataclass
class UpdateResult:
    innovation: float           # nu = z - H x_pred [m]
    innovation_variance: float  # S = H P_pred H^T + R [m^2]
    nis: float                  # normalised innovation squared nu^2 / S
    gain: float                 # distance component of the Kalman gain
    accepted: bool              # False if the measurement was gated out


class ConstantVelocityKF:
    """Kalman filter for 1-D distance with a constant-velocity motion model."""

    def __init__(self, accel_psd: float):
        self.q = accel_psd   # [m^2/s^3] continuous white-noise acceleration PSD
        self.x = np.zeros(2)
        self.P = np.eye(2)
        self.time = 0.0
        self.initialized = False

    def initialize(self, distance: float, distance_variance: float,
                   velocity_variance: float, time: float, velocity: float = 0.0):
        self.x = np.array([distance, velocity])
        self.P = np.diag([distance_variance, velocity_variance])
        self.time = time
        self.initialized = True

    @staticmethod
    def transition(dt: float) -> np.ndarray:
        return np.array([[1.0, dt],
                         [0.0, 1.0]])

    def process_noise(self, dt: float) -> np.ndarray:
        """Q for a continuous white-noise acceleration model over dt."""
        return self.q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                                  [dt ** 2 / 2.0, dt]])

    def predict(self, time: float):
        """Propagate the state and covariance forward to `time` [s]."""
        dt = time - self.time
        if dt <= 0.0:
            return
        f = self.transition(dt)
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + self.process_noise(dt)
        self.time = time

    def update(self, z: float, r: float, gate: float = math.inf) -> UpdateResult:
        """
        Fuse one distance reading z with variance r.

        The reading is rejected (state untouched) when its normalised
        innovation squared exceeds `gate` - e.g. 9.0 rejects anything further
        than 3 sigma from where the filter expected it.
        """
        nu = z - (H @ self.x)[0]
        s = (H @ self.P @ H.T)[0, 0] + r
        k = (self.P @ H.T)[:, 0] / s
        nis = nu * nu / s
        if nis > gate:
            return UpdateResult(nu, s, nis, k[0], accepted=False)

        self.x = self.x + k * nu
        i_kh = np.eye(2) - np.outer(k, H)
        self.P = i_kh @ self.P @ i_kh.T + r * np.outer(k, k)  # Joseph form
        return UpdateResult(nu, s, nis, k[0], accepted=True)

    @property
    def distance(self) -> float:
        return float(self.x[0])

    @property
    def distance_variance(self) -> float:
        return float(self.P[0, 0])

    @property
    def velocity(self) -> float:
        return float(self.x[1])

    @property
    def velocity_variance(self) -> float:
        return float(self.P[1, 1])


def inverse_variance_fusion(values, variances) -> tuple[float, float]:
    """
    Optimal static fusion of independent readings of the same quantity.

        1/var_f = sum(1/var_i)       mean_f = var_f * sum(z_i / var_i)

    This is exactly what one Kalman update per reading computes when the
    filter starts with no prior information, and it shows the core idea:
    var_f is always smaller than the smallest var_i.
    """
    values = np.asarray(values, dtype=float)
    weights = 1.0 / np.asarray(variances, dtype=float)
    fused_variance = 1.0 / weights.sum()
    return float(fused_variance * (weights * values).sum()), float(fused_variance)
