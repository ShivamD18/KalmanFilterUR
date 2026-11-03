"""
Start Gazebo with the UR arm, its sensor bracket and the work cell.

Also starts ros2_control and the Gazebo -> ROS bridge:

    ros2 launch ur_fusion_description gazebo_sim.launch.py             # with the Gazebo GUI
    ros2 launch ur_fusion_description gazebo_sim.launch.py gui:=false  # headless server
"""

import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def gazebo(context):
    world = LaunchConfiguration('world').perform(context)
    gui = LaunchConfiguration('gui').perform(context).lower() in ('true', '1', 'yes')
    # Headless still needs off-screen rendering: the sensor beams are GPU ray casts.
    gz_args = f'-r -v 2 {world}' if gui else f'-r -s --headless-rendering -v 2 {world}'
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items(),
    )]


def generate_launch_description():
    pkg = FindPackageShare('ur_fusion_description')

    robot_description = ParameterValue(Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([pkg, 'urdf', 'ur_fusion.urdf.xacro']),
        ' ur_type:=', LaunchConfiguration('ur_type'),
    ]), value_type=str)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'ur_fusion'],
        output='screen',
    )

    # The work cell is static scenery with its own description, so RViz can draw it
    # (from /workcell/robot_description) exactly as Gazebo simulates it.
    workcell_description = ParameterValue(Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([pkg, 'urdf', 'workcell.urdf.xacro']),
    ]), value_type=str)
    workcell_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='workcell',
        parameters=[{'robot_description': workcell_description, 'use_sim_time': True}],
        output='screen',
    )
    spawn_workcell = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/workcell/robot_description', '-name', 'workcell'],
        output='screen',
    )
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': PathJoinSubstitution([pkg, 'config', 'gz_bridge.yaml'])}],
        output='screen',
    )
    controllers_file = PathJoinSubstitution([pkg, 'config', 'ur_controllers.yaml'])
    controllers = [
        Node(package='controller_manager', executable='spawner', output='screen',
             arguments=[name, '--controller-manager', '/controller_manager',
                        '--param-file', controllers_file])
        for name in ('joint_state_broadcaster', 'joint_trajectory_controller')
    ]

    return LaunchDescription([
        DeclareLaunchArgument('ur_type', default_value='ur5e',
                              description='UR model, e.g. ur5e (the scan is tuned for it)'),
        DeclareLaunchArgument('gui', default_value='true', description='Show the Gazebo GUI'),
        DeclareLaunchArgument('world', description='World file',
                              default_value=PathJoinSubstitution(
                                  [pkg, 'worlds', 'fusion_lab.sdf'])),
        # Keep Gazebo's transport on loopback: otherwise its multicast discovery runs
        # over Wi-Fi, and a network change silently cuts the sensor data and /clock.
        SetEnvironmentVariable('GZ_IP', os.environ.get('GZ_IP', '127.0.0.1')),
        OpaqueFunction(function=gazebo),
        robot_state_publisher,
        spawn_robot,
        workcell_state_publisher,
        spawn_workcell,
        bridge,
        # The controller manager lives inside Gazebo, so wait until the robot is in.
        RegisterEventHandler(OnProcessExit(target_action=spawn_robot, on_exit=controllers)),
    ])
