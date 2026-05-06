import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    vehicle_name = os.environ.get('VEHICLE_NAME', 'e4')
    pkg_share = get_package_share_directory('basic_launch')

    default_cfg = os.path.join(pkg_share, 'config', vehicle_name,
                               'robot_localization', 'ekf.yaml')

    cfg_arg = DeclareLaunchArgument(
        'ekf_config',
        default_value=default_cfg,
        description='Path to robot_localization ekf.yaml')

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[LaunchConfiguration('ekf_config')],
        remappings=[('odometry/filtered', '/odometry/filtered')],
    )

    return LaunchDescription([cfg_arg, ekf_node])
