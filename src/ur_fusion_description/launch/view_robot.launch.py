"""Inspect the arm and its sensor bracket in RViz, with sliders for the joints.

    ros2 launch ur_fusion_description view_robot.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('ur_fusion_description')
    robot_description = ParameterValue(Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([pkg, 'urdf', 'ur_fusion.urdf.xacro']),
        ' ur_type:=', LaunchConfiguration('ur_type'), ' sim_gazebo:=false',
    ]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('ur_type', default_value='ur5e'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui'),
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', PathJoinSubstitution([pkg, 'rviz', 'fusion.rviz'])]),
    ])
