from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mppi_controller',
            executable='adapt_mppi_node',
            name='adapt_mppi_node',
            output='screen',
            parameters=[{
                'v_ref': 3.0,
                'K': 600,
                'H': 30,
                'dt': 0.1,
                'sigma_steer': 0.05,
                'sigma_accel': 0.8,
                'lambda_': 1.0,
                'rate_hz': 10.0,
            }],
        ),
    ])
