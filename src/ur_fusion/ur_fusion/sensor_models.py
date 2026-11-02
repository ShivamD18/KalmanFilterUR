"""
Noise models for the two range sensors on the bracket.

Gazebo only knows geometry: it ray-casts the beams and returns the ideal
distance to whatever they hit. These classes turn that ideal number into what
a real sensor would report, and they also tell the Kalman filter how much to
trust each reading (its standard deviation as a function of distance).

Both models are deliberately simple, physically motivated approximations of
two very common hobby/lab sensors:

* UltrasonicModel - HC-SR04 style time-of-flight sonar
* InfraredModel   - Sharp GP2Y0A21YK0F style IR triangulation sensor

They are complementary, which is what makes fusing them worthwhile: the sonar
error is constant with distance and it sees up to 4 m, while the IR sensor is
very precise up close but its error grows with the square of the distance and
it is blind beyond 80 cm.
"""

from dataclasses import dataclass
import math

import numpy as np


def speed_of_sound(temperature_c: float) -> float:
    """Speed of sound in dry air [m/s] at the given temperature [deg C]."""
    return 331.3 * math.sqrt(1.0 + temperature_c / 273.15)


@dataclass
class UltrasonicModel:
    """
    HC-SR04 style ultrasonic time-of-flight ranger.

    The module times the echo and converts it with an *assumed* speed of sound:

        t_echo  = 2 d / c_air + timing_jitter
        reading = c_assumed * t_echo / 2

    so the noise is the same at every distance: sigma = c * sigma_t / 2.
    A small fraction of pings report a spurious echo (cross-talk, multipath,
    side lobes) - a uniformly random value - which the filter must reject.
    """

    min_range: float = 0.02                  # [m]
    max_range: float = 4.0                   # [m]
    field_of_view: float = 0.2618            # [rad] ~15 deg cone
    update_rate: float = 15.0                # [Hz] ~60 ms measurement cycle
    timing_jitter: float = 60e-6             # [s] 1-sigma echo timing jitter
    outlier_probability: float = 0.02        # chance of a spurious echo
    air_temperature: float = 20.0            # [deg C] actual air temperature
    assumed_temperature: float = 20.0        # [deg C] used by the firmware

    def sigma(self, distance: float) -> float:
        """1-sigma reading noise [m]; independent of distance for sonar."""
        return speed_of_sound(self.assumed_temperature) * self.timing_jitter / 2.0

    def measure(self, true_distance: float, rng: np.random.Generator) -> float:
        """Simulate one reading. Returns +inf / -inf when out of range (REP-117)."""
        if math.isnan(true_distance) or true_distance > self.max_range:
            return math.inf
        if true_distance < self.min_range:
            return -math.inf
        if rng.random() < self.outlier_probability:
            return float(rng.uniform(self.min_range, self.max_range))
        t_echo = (2.0 * true_distance / speed_of_sound(self.air_temperature)
                  + rng.normal(0.0, self.timing_jitter))
        return speed_of_sound(self.assumed_temperature) * t_echo / 2.0


@dataclass
class InfraredModel:
    """
    Sharp GP2Y0A21YK0F style IR triangulation ranger (10-80 cm).

    The sensor outputs an analog voltage that falls off roughly as 1/distance:

        V(d) = k / (d + d0) + v_offset

    The voltage is noisy and is sampled by an ADC; the driver inverts the curve.
    Because the curve flattens out with distance, the same voltage noise turns
    into a much larger distance error far away (first-order propagation):

        sigma_d(d) = |dd/dV| * sigma_V = (d + d0)^2 / k * sigma_V
    """

    min_range: float = 0.10                  # [m]
    max_range: float = 0.80                  # [m]
    field_of_view: float = 0.0524            # [rad] ~3 deg spot
    update_rate: float = 25.0                # [Hz] ~38 ms measurement cycle
    k: float = 0.229                         # [V m] curve fit
    d0: float = 0.0042                       # [m]   curve fit
    v_offset: float = 0.10                   # [V]   curve fit
    voltage_noise: float = 0.0125            # [V] 1-sigma output noise
    adc_bits: int = 10
    adc_reference: float = 5.0               # [V]

    @property
    def adc_step(self) -> float:
        return self.adc_reference / 2 ** self.adc_bits

    def voltage(self, distance: float) -> float:
        return self.k / (distance + self.d0) + self.v_offset

    def distance_from_voltage(self, voltage: float) -> float:
        if voltage <= self.v_offset:
            return math.inf
        return self.k / (voltage - self.v_offset) - self.d0

    def sigma(self, distance: float) -> float:
        """1-sigma reading noise [m] at the given distance."""
        # Uniform ADC quantisation adds step^2 / 12 of variance to the voltage.
        sigma_v = math.sqrt(self.voltage_noise ** 2 + self.adc_step ** 2 / 12.0)
        return (distance + self.d0) ** 2 / self.k * sigma_v

    def measure(self, true_distance: float, rng: np.random.Generator) -> float:
        """Simulate one reading. Returns +inf / -inf when out of range (REP-117)."""
        if math.isnan(true_distance) or true_distance > self.max_range:
            return math.inf
        if true_distance < self.min_range:
            return -math.inf
        voltage = self.voltage(true_distance) + rng.normal(0.0, self.voltage_noise)
        voltage = round(voltage / self.adc_step) * self.adc_step
        return self.distance_from_voltage(voltage)


SENSOR_MODEL_TYPES = {
    'ultrasonic': UltrasonicModel,
    'infrared': InfraredModel,
}
