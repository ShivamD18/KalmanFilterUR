"""
Load ROS parameters straight into the dataclasses used by the core library.

Both the node that simulates the readings and the Kalman filter node load the
same `ultrasonic.*` / `infrared.*` parameters (see config/fusion_params.yaml),
so the filter's idea of how noisy each sensor is matches the simulation.
"""

from dataclasses import fields

from rcl_interfaces.msg import ParameterDescriptor

from .sensor_models import SENSOR_MODEL_TYPES

# In the ROS graph Gazebo decides the update rates (see the URDF), so the
# rate fields of the models are only used by the offline demo.
NOT_ROS_PARAMETERS = {'update_rate'}


def declare_dataclass(node, cls, prefix: str = ''):
    """Declare one parameter per dataclass field (defaults from the class) and build it."""
    defaults = cls()
    values = {}
    for field in fields(cls):
        if field.name in NOT_ROS_PARAMETERS:
            continue
        name = prefix + field.name
        # dynamic typing lets the YAML say `1` for a float field
        node.declare_parameter(name, getattr(defaults, field.name),
                               ParameterDescriptor(dynamic_typing=True))
        values[field.name] = field.type(node.get_parameter(name).value)
    return cls(**values)


def declare_sensor_models(node) -> dict:
    """Build both sensors' noise models from the `<sensor>.<field>` parameters."""
    return {sensor: declare_dataclass(node, model_type, prefix=f'{sensor}.')
            for sensor, model_type in SENSOR_MODEL_TYPES.items()}


def stamp_to_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9
