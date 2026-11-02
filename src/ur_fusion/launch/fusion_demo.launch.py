"""
Launch the whole demo.

Gazebo with the UR5e and its sensor bracket, the sensor models, the Kalman
filters, the scanning motion, RViz and the live dashboard:

    ros2 launch ur_fusion fusion_demo.launch.py
    ros2 launch ur_fusion fusion_demo.launch.py gui:=false rviz:=false dashboard_headless:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = PathJoinSubstitution([FindPackageShare('ur_fusion'), 'config', 'fusion_params.yaml'])
    description = FindPackageShare('ur_fusion_description')

    def fusion_node(executable, extra_params=None, condition=None):
        return Node(package='ur_fusion', executable=executable, output='screen',
                    parameters=[params, {'use_sim_time': True, **(extra_params or {})}],
                    condition=condition)

    return LaunchDescription([
        DeclareLaunchArgument('ur_type', default_value='ur5e'),
        DeclareLaunchArgument('gui', default_value='true', description='Show the Gazebo GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='Start RViz'),
        DeclareLaunchArgument('dashboard', default_value='true',
                              description='Start the live matplotlib dashboard'),
        DeclareLaunchArgument('dashboard_headless', default_value='false',
                              description='Draw the dashboard off-screen and save PNGs'),
        DeclareLaunchArgument('dashboard_output_dir', default_value='',
                              description='Where the dashboard saves PNGs (~/.ros/ur_fusion)'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([description, 'launch', 'gazebo_sim.launch.py'])),
            launch_arguments={'ur_type': LaunchConfiguration('ur_type'),
                              'gui': LaunchConfiguration('gui')}.items(),
        ),
        fusion_node('sensor_models_node'),
        fusion_node('fusion_node'),
        fusion_node('scan_motion_node'),
        fusion_node('uncertainty_markers_node'),
        fusion_node('dashboard_node',
                    extra_params={'headless': LaunchConfiguration('dashboard_headless'),
                                  'output_dir': LaunchConfiguration('dashboard_output_dir')},
                    condition=IfCondition(LaunchConfiguration('dashboard'))),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', PathJoinSubstitution([description, 'rviz', 'fusion.rviz'])],
             parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
