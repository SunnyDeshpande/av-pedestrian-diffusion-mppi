# launch/lidar_preprocessing_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to the config file
    config_file = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'lidar_params.yaml'
    ])

    return LaunchDescription([
        Node(
            package='lidar_person_detection',
            executable='lidar_preprocessor',
            name='lidar_preprocessor',
            output='screen',
            parameters=[config_file],
            remappings=[],
        )
    ])
