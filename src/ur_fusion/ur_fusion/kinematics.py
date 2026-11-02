"""
Minimal forward / inverse kinematics for a serial chain read from a URDF.

Only numpy and the standard library are used, so the motion node does not
need MoveIt or KDL just to keep the sensor bracket pointing at the object.
"""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw -> 3x3 rotation matrix (Rz @ Ry @ Rx)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula for a rotation of `angle` about unit vector `axis`."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def rotation_error(r_target: np.ndarray, r_current: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) that takes r_current onto r_target."""
    r_err = r_target @ r_current.T
    cos_angle = np.clip((np.trace(r_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    vee = np.array([r_err[2, 1] - r_err[1, 2],
                    r_err[0, 2] - r_err[2, 0],
                    r_err[1, 0] - r_err[0, 1]])
    if angle < 1e-9:
        return 0.5 * vee
    if np.pi - angle < 1e-6:
        # Near 180 deg the skew part vanishes; recover the axis from the diagonal.
        axis = np.sqrt(np.maximum((np.diag(r_err) + 1.0) / 2.0, 0.0))
        axis *= np.where(vee >= 0.0, 1.0, -1.0)
        return angle * axis / np.linalg.norm(axis)
    return angle / (2.0 * np.sin(angle)) * vee


def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


@dataclass
class ChainJoint:
    name: str
    joint_type: str          # 'revolute', 'continuous', 'prismatic' or 'fixed'
    origin: np.ndarray       # 4x4 transform parent link -> joint frame
    axis: np.ndarray         # unit axis in the joint frame
    lower: float = -np.inf
    upper: float = np.inf

    @property
    def movable(self) -> bool:
        return self.joint_type != 'fixed'


@dataclass
class IkResult:
    joint_positions: np.ndarray
    success: bool
    position_error: float     # [m]
    orientation_error: float  # [rad]
    iterations: int


class SerialChain:
    """Kinematic chain between two links of a URDF."""

    def __init__(self, joints: list[ChainJoint]):
        self.joints = joints
        self.movable_joints = [j for j in joints if j.movable]
        self.joint_names = [j.name for j in self.movable_joints]
        self.lower = np.array([j.lower for j in self.movable_joints])
        self.upper = np.array([j.upper for j in self.movable_joints])

    @classmethod
    def from_urdf(cls, urdf_xml: str, base_link: str, tip_link: str) -> 'SerialChain':
        root = ET.fromstring(urdf_xml)
        joint_by_child = {}
        # Only top-level joints: <ros2_control> and <transmission> blocks also
        # contain <joint> tags, without parent/child links.
        for joint in root.findall('joint'):
            joint_by_child[joint.find('child').get('link')] = joint

        chain = []
        link = tip_link
        while link != base_link:
            if link not in joint_by_child:
                raise ValueError(f'No chain from "{base_link}" to "{tip_link}" in the URDF')
            joint = joint_by_child[link]
            chain.append(cls._parse_joint(joint))
            link = joint.find('parent').get('link')
        chain.reverse()
        return cls(chain)

    @staticmethod
    def _parse_joint(joint: ET.Element) -> ChainJoint:
        origin = joint.find('origin')
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
            rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
        axis_el = joint.find('axis')
        axis = np.array([1.0, 0.0, 0.0])
        if axis_el is not None:
            axis = np.array([float(v) for v in axis_el.get('xyz').split()])
        axis = axis / np.linalg.norm(axis)

        lower, upper = -np.inf, np.inf
        limit = joint.find('limit')
        if joint.get('type') in ('revolute', 'prismatic') and limit is not None:
            lower = float(limit.get('lower', '-inf'))
            upper = float(limit.get('upper', 'inf'))
        return ChainJoint(
            name=joint.get('name'),
            joint_type=joint.get('type'),
            origin=make_transform(rpy_to_matrix(*rpy), xyz),
            axis=axis,
            lower=lower,
            upper=upper,
        )

    def forward(self, q) -> np.ndarray:
        """Pose of the tip link in the base link frame (4x4)."""
        return self._forward_with_axes(q)[0]

    def _forward_with_axes(self, q):
        q = np.asarray(q, dtype=float)
        t = np.eye(4)
        axes, points = [], []
        i = 0
        for joint in self.joints:
            t = t @ joint.origin
            if not joint.movable:
                continue
            axis_world = t[:3, :3] @ joint.axis
            axes.append(axis_world)
            points.append(t[:3, 3].copy())
            if joint.joint_type == 'prismatic':
                t = t @ make_transform(np.eye(3), joint.axis * q[i])
            else:
                t = t @ make_transform(axis_angle_to_matrix(joint.axis, q[i]), [0, 0, 0])
            i += 1
        return t, axes, points

    def jacobian(self, q) -> np.ndarray:
        """6xN geometric Jacobian [linear; angular] expressed in the base frame."""
        t_tip, axes, points = self._forward_with_axes(q)
        p_tip = t_tip[:3, 3]
        jac = np.zeros((6, len(self.movable_joints)))
        for i, joint in enumerate(self.movable_joints):
            if joint.joint_type == 'prismatic':
                jac[:3, i] = axes[i]
            else:
                jac[:3, i] = np.cross(axes[i], p_tip - points[i])
                jac[3:, i] = axes[i]
        return jac

    def inverse(self, target: np.ndarray, q_seed, *, tolerance: float = 1e-6,
                max_iterations: int = 200, damping: float = 0.02,
                max_step: float = 0.2) -> IkResult:
        """Damped-least-squares IK for a full 6-DoF pose target (4x4)."""
        q = np.array(q_seed, dtype=float)
        pos_err = rot_err = np.inf
        for iteration in range(1, max_iterations + 1):
            current = self.forward(q)
            dp = target[:3, 3] - current[:3, 3]
            dr = rotation_error(target[:3, :3], current[:3, :3])
            pos_err, rot_err = np.linalg.norm(dp), np.linalg.norm(dr)
            if pos_err < tolerance and rot_err < tolerance:
                return IkResult(q, True, pos_err, rot_err, iteration)

            jac = self.jacobian(q)
            error = np.concatenate([dp, dr])
            jjt = jac @ jac.T + (damping ** 2) * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, error)
            step = np.max(np.abs(dq))
            if step > max_step:
                dq *= max_step / step
            q = np.clip(q + dq, self.lower, self.upper)
        return IkResult(q, False, pos_err, rot_err, max_iterations)
