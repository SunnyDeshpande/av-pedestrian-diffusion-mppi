from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    vehicle_name_arg = DeclareLaunchArgument(
        'vehicle_name',
        default_value='',
        description='Vehicle identifier (e.g., e2, e4)',
    )
    matching_threshold_arg = DeclareLaunchArgument(
        'matching_threshold',
        default_value='2.0',
        description='Maximum distance (meters) for matching Lidar and Camera detections',
    )

    fusion_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'), 'config', 'sensor_fusion_params.yaml',
    ])
    lidar_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'), 'config', 'lidar_params.yaml',
    ])

    return LaunchDescription([
        vehicle_name_arg,
        matching_threshold_arg,
        Node(
            package='adapt_full',
            executable='lidar_processing',
            name='lidar_processing',
            output='screen',
            parameters=[lidar_config],
        ),
        Node(
            package='yolo_person_detector',
            executable='rgbd_pedestrian_detector',
            name='rgbd_pedestrian_detector',
            output='screen',
        ),
        Node(
            package='adapt_full',
            executable='lidar_camera_fusion',
            name='sensor_fusion_node',
            output='screen',
            parameters=[
                fusion_config,
                {
                    'vehicle_name': LaunchConfiguration('vehicle_name'),
                    'matching_threshold': LaunchConfiguration('matching_threshold'),
                },
            ],
        ),
    ])
