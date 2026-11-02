import math
import shutil
import subprocess

import numpy as np
import pytest

from ur_fusion.kinematics import rotation_error, rpy_to_matrix, SerialChain

LIMIT = '<limit lower="-6.28" upper="6.28" effort="1" velocity="1"/>'
PLANAR_URDF = f"""<robot name="planar">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="tip"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/>{LIMIT}</joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>{LIMIT}</joint>
  <joint name="tip_joint" type="fixed"><parent link="l2"/><child link="tip"/>
    <origin xyz="0.5 0 0"/></joint>
</robot>"""


def test_joint_tags_outside_the_kinematic_tree_are_ignored():
    urdf = PLANAR_URDF.replace('</robot>', """
  <ros2_control name="hw" type="system">
    <joint name="j1"><command_interface name="position"/></joint>
  </ros2_control>
</robot>""")
    assert SerialChain.from_urdf(urdf, 'base', 'tip').joint_names == ['j1', 'j2']


def test_forward_kinematics_of_a_planar_arm():
    chain = SerialChain.from_urdf(PLANAR_URDF, 'base', 'tip')
    assert chain.joint_names == ['j1', 'j2']
    np.testing.assert_allclose(chain.forward([0, 0])[:3, 3], [1.5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(chain.forward([math.pi / 2, 0])[:3, 3], [0, 1.5, 0], atol=1e-12)
    np.testing.assert_allclose(chain.forward([0, math.pi / 2])[:3, 3], [1, 0.5, 0], atol=1e-12)


def test_rotation_error_recovers_the_rotation_vector():
    for rotvec in ([0.3, -0.2, 0.1], [0.0, 0.0, 3.0], [1.5, 1.5, 0.0]):
        rotvec = np.array(rotvec)
        angle = np.linalg.norm(rotvec)
        axis = rotvec / angle
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
        np.testing.assert_allclose(rotation_error(rot, np.eye(3)), rotvec, atol=1e-9)


@pytest.fixture(scope='module')
def ur5e():
    xacro = shutil.which('xacro')
    if xacro is None:
        pytest.skip('xacro not available')
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('ur_description')
    except Exception:
        pytest.skip('ur_description not installed')
    urdf = subprocess.run([xacro, f'{share}/urdf/ur.urdf.xacro', 'ur_type:=ur5e', 'name:=ur'],
                          check=True, capture_output=True, text=True).stdout
    return SerialChain.from_urdf(urdf, 'base_link', 'tool0')


def test_ur5e_zero_pose_matches_its_dh_parameters(ur5e):
    # a2 + a3 = 0.8172 out, d4 + d6 = 0.2329 sideways, d1 - d5 = 0.0628 up
    np.testing.assert_allclose(ur5e.forward(np.zeros(6))[:3, 3], [0.8172, 0.2329, 0.0628],
                               atol=1e-4)


def test_ur5e_jacobian_matches_finite_differences(ur5e):
    q = np.array([-0.3, -1.7, 1.7, -1.5, -1.57, 0.5])
    jac = ur5e.jacobian(q)
    eps = 1e-6
    t0 = ur5e.forward(q)
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        t1 = ur5e.forward(q + dq)
        np.testing.assert_allclose((t1[:3, 3] - t0[:3, 3]) / eps, jac[:3, i], atol=1e-5)
        np.testing.assert_allclose(rotation_error(t1[:3, :3], t0[:3, :3]) / eps, jac[3:, i],
                                   atol=1e-5)


def test_ur5e_inverse_kinematics_round_trip(ur5e):
    q_true = np.array([-0.3, -1.7, 1.7, -1.5, -1.57, 0.5])
    target = ur5e.forward(q_true)
    result = ur5e.inverse(target, q_true + 0.15)
    assert result.success
    np.testing.assert_allclose(ur5e.forward(result.joint_positions), target, atol=1e-6)


def test_ur5e_can_point_straight_down(ur5e):
    target = np.eye(4)
    target[:3, :3] = rpy_to_matrix(0.0, math.pi, 0.0)   # tool0 z down
    target[:3, 3] = [0.45, 0.1, 0.0]
    result = ur5e.inverse(target, [0, -1.57, 1.57, -1.57, -1.57, 0])
    assert result.success
