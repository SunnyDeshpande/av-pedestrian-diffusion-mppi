#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray


class CameraPositionSpoof(Node):
    def __init__(self):
        super().__init__('camera_position_spoof')

        # Declare parameters
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('test_distance', 0)
        self.declare_parameter('test_angle', 0)

        # Get parameters
        publish_rate = self.get_parameter('publish_rate').value
        self.test_distance = self.get_parameter('test_distance').value
        self.test_angle = self.get_parameter('test_angle').value

        # Publisher
        self.pub = self.create_publisher(
            Int32MultiArray,
            '/camera_pedestrian_position',
            10
        )

        # Timer to publish at regular intervals
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('Camera Position Spoof Node initialized')
        self.get_logger().info(f'Publishing to /camera_pedestrian_position at {publish_rate} Hz')
        self.get_logger().info(f'Test data: distance={self.test_distance}, angle={self.test_angle}')

    def timer_callback(self):
        """Publish spoof camera position data"""
        msg = Int32MultiArray()
        msg.data = [self.test_distance, self.test_angle]
        self.pub.publish(msg)

        self.get_logger().debug(
            f'Published: distance={self.test_distance}, angle={self.test_angle}'
        )


def main(args=None):
    rclpy.init(args=args)

    try:
        node = CameraPositionSpoof()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in camera position spoof node: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
