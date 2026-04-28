#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int32MultiArray, Bool, Float64, String
from geometry_msgs.msg import Twist
import numpy as np
from rclpy.time import Time


class HighLevelDecisionNode(Node):
    """
    Central safety executive for autonomous vehicle decision-making.

    Subscribes to:
        - /fusion_pedestrian_position (Int32MultiArray) - Pedestrian detections
        - /pedestrian_sign_present (Bool) - Regulatory sign detection
        - /pedestrian_motion (Twist) - Pedestrian velocity
        - /pedestrian_ttc (Float64) - Time to collision

    Publishes to:
        - /safety_decision (String) - Safety state command

    Decision States:
        - CRUISE: Normal driving, no pedestrian threat
        - STOP_YIELD: Immediate stop required
        - SLOW_CAUTION: Slow down, pedestrian standing still
        - STOP_WATCH: Stop and watch moving pedestrian
        - CREEP_PASS: Carefully pass after waiting
    """

    # Safety thresholds (configurable via ROS parameters)
    STALE_DATA_TIMEOUT = 0.5  # seconds
    TTC_CRITICAL_MIN = 0.0     # seconds
    TTC_CRITICAL_MAX = 2.5     # seconds
    SPEED_STANDING_THRESHOLD = 0.1  # m/s
    WAIT_PATIENCE_TIMEOUT = 2.0     # seconds

    def __init__(self):
        super().__init__('high_level_decision_node')

        # Declare ROS parameters
        self.declare_parameter('decision_rate_hz', 20.0)
        self.declare_parameter('stale_data_timeout', 0.5)
        self.declare_parameter('ttc_critical_min', 0.0)
        self.declare_parameter('ttc_critical_max', 2.5)
        self.declare_parameter('speed_standing_threshold', 0.1)
        self.declare_parameter('wait_patience_timeout', 2.0)

        # Get parameters
        rate_hz = self.get_parameter('decision_rate_hz').value
        self.STALE_DATA_TIMEOUT = self.get_parameter('stale_data_timeout').value
        self.TTC_CRITICAL_MIN = self.get_parameter('ttc_critical_min').value
        self.TTC_CRITICAL_MAX = self.get_parameter('ttc_critical_max').value
        self.SPEED_STANDING_THRESHOLD = self.get_parameter('speed_standing_threshold').value
        self.WAIT_PATIENCE_TIMEOUT = self.get_parameter('wait_patience_timeout').value

        # QoS Profile for sensor data (best effort, keep last)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Initialize data storage with timestamps
        self.fusion_data = []
        self.fusion_timestamp = None

        self.sign_present = False
        self.sign_timestamp = None

        self.pedestrian_velocity = Twist()
        self.motion_timestamp = None

        self.ttc_value = float('inf')
        self.ttc_timestamp = None

        # Internal state management
        self.wait_start_time = None  # For patience logic
        self.current_state = "CRUISE"  # Track current state

        # Subscribers with callbacks
        self.fusion_sub = self.create_subscription(
            Int32MultiArray,
            '/fusion_pedestrian_position',
            self.fusion_callback,
            sensor_qos
        )

        self.sign_sub = self.create_subscription(
            Bool,
            '/pedestrian_sign_present',
            self.sign_callback,
            sensor_qos
        )

        self.motion_sub = self.create_subscription(
            Twist,
            '/pedestrian_motion',
            self.motion_callback,
            sensor_qos
        )

        self.ttc_sub = self.create_subscription(
            Float64,
            '/pedestrian_ttc',
            self.ttc_callback,
            sensor_qos
        )

        # Publisher for safety decision
        self.decision_pub = self.create_publisher(
            String,
            '/safety_decision',
            10
        )

        # Timer for decision loop (20 Hz)
        timer_period = 1.0 / rate_hz
        self.timer = self.create_timer(timer_period, self.decision_loop)

        # Statistics
        self.decision_counts = {
            "CRUISE": 0,
            "STOP_YIELD": 0,
            "SLOW_CAUTION": 0,
            "STOP_WATCH": 0,
            "CREEP_PASS": 0
        }

        self.get_logger().info('High-Level Decision Node initialized')
        self.get_logger().info(f'Decision rate: {rate_hz} Hz')
        self.get_logger().info(f'Stale data timeout: {self.STALE_DATA_TIMEOUT}s')
        self.get_logger().info(f'TTC critical range: {self.TTC_CRITICAL_MIN}-{self.TTC_CRITICAL_MAX}s')

    def fusion_callback(self, msg):
        """
        Callback for pedestrian fusion data.
        Stores the latest detection array and timestamp.
        """
        self.fusion_data = list(msg.data)
        self.fusion_timestamp = self.get_clock().now()
        self.get_logger().debug(f'Fusion data updated: {len(self.fusion_data)//2} detections')

    def sign_callback(self, msg):
        """
        Callback for regulatory sign detection.
        Stores whether pedestrian is holding a STOP/Yield sign.
        """
        self.sign_present = msg.data
        self.sign_timestamp = self.get_clock().now()
        if self.sign_present:
            self.get_logger().debug('Regulatory sign detected!')

    def motion_callback(self, msg):
        """
        Callback for pedestrian motion (velocity vector).
        Stores the Twist message for speed calculation.
        """
        self.pedestrian_velocity = msg
        self.motion_timestamp = self.get_clock().now()
        speed = self.calculate_speed_magnitude()
        self.get_logger().debug(f'Pedestrian speed: {speed:.2f} m/s')

    def ttc_callback(self, msg):
        """
        Callback for Time To Collision.
        Stores the TTC value in seconds.
        """
        self.ttc_value = msg.data
        self.ttc_timestamp = self.get_clock().now()
        if 0.0 < self.ttc_value < self.TTC_CRITICAL_MAX:
            self.get_logger().debug(f'TTC: {self.ttc_value:.2f}s (CRITICAL)')
        else:
            self.get_logger().debug(f'TTC: {self.ttc_value:.2f}s')

    def calculate_speed_magnitude(self):
        """
        Calculate the speed magnitude from velocity vector.

        Math:
            speed = sqrt(vx^2 + vy^2)

        Returns:
            float: Speed magnitude in m/s
        """
        vx = self.pedestrian_velocity.linear.x
        vy = self.pedestrian_velocity.linear.y
        speed = np.sqrt(vx**2 + vy**2)
        return speed

    def is_data_stale(self, timestamp):
        """
        Check if data timestamp is too old (stale).

        Args:
            timestamp: rclpy.time.Time object or None

        Returns:
            bool: True if data is stale or None
        """
        if timestamp is None:
            return True

        current_time = self.get_clock().now()
        age = (current_time - timestamp).nanoseconds / 1e9  # Convert to seconds

        return age > self.STALE_DATA_TIMEOUT

    def check_any_stale_data(self):
        """
        Check if any critical sensor data is stale.

        Returns:
            bool: True if any data is stale
        """
        if self.is_data_stale(self.fusion_timestamp):
            self.get_logger().warn('Fusion data is stale')
            return True
        if self.is_data_stale(self.sign_timestamp):
            self.get_logger().warn('Sign data is stale')
            return True
        if self.is_data_stale(self.motion_timestamp):
            self.get_logger().warn('Motion data is stale')
            return True
        if self.is_data_stale(self.ttc_timestamp):
            self.get_logger().warn('TTC data is stale')
            return True

        return False

    def reset_patience_timer(self):
        """
        Reset the patience timer used for waiting logic.
        """
        if self.wait_start_time is not None:
            self.get_logger().debug('Resetting patience timer')
            self.wait_start_time = None

    def publish_decision(self, state, reason):
        """
        Publish the safety decision and log the reasoning.

        Args:
            state (str): One of the decision states
            reason (str): Human-readable reason for this decision
        """
        msg = String()
        msg.data = state
        self.decision_pub.publish(msg)

        # Update statistics
        self.decision_counts[state] += 1

        # Log state changes only
        if state != self.current_state:
            self.get_logger().info(f'State: {state} - {reason}')
            self.current_state = state
        else:
            self.get_logger().debug(f'State: {state} - {reason}')

    def decision_loop(self):
        """
        Main decision loop (20 Hz).
        Implements the priority-based decision hierarchy.

        Decision Priority (top-down):
            0. Fail-Safe: Stale data � STOP_YIELD
            1. No Pedestrian: Empty fusion array � CRUISE
            2. Critical TTC: 0.0 < TTC < 2.5 � STOP_YIELD
            3. Regulatory Sign: Sign present � STOP_YIELD
            4. Behavioral Analysis:
                4a. Standing Still: speed < 0.1 � SLOW_CAUTION
                4b. Moving: speed >= 0.1 � Patience logic
        """

        # ========== Priority 0: Fail-Safe Check (Stale Data) ==========
        if self.check_any_stale_data():
            self.publish_decision("STOP_YIELD", "Stale data - executing fail-safe stop")
            self.reset_patience_timer()
            return

        # ========== Priority 1: No Pedestrian Check ==========
        if len(self.fusion_data) == 0:
            self.publish_decision("CRUISE", "No pedestrian detected")
            self.reset_patience_timer()  # Important: Reset timer when clear
            return

        # ========== Priority 2: Critical Safety Override (TTC) ==========
        if self.TTC_CRITICAL_MIN < self.ttc_value < self.TTC_CRITICAL_MAX:
            self.publish_decision(
                "STOP_YIELD",
                f"TTC Critical override: {self.ttc_value:.2f}s"
            )
            self.reset_patience_timer()
            return

        # ========== Priority 3: Regulatory Check (Signs) ==========
        if self.sign_present:
            self.publish_decision("STOP_YIELD", "Pedestrian holding regulatory sign")
            self.reset_patience_timer()
            return

        # ========== Priority 4: Behavioral Analysis (Motion & Patience) ==========
        pedestrian_speed = self.calculate_speed_magnitude()

        # Sub-case 4a: Standing Still
        if pedestrian_speed < self.SPEED_STANDING_THRESHOLD:
            self.publish_decision(
                "SLOW_CAUTION",
                f"Pedestrian standing still (speed: {pedestrian_speed:.2f} m/s)"
            )
            self.reset_patience_timer()
            return

        # Sub-case 4b: Active Moving (Patience Logic)
        # Pedestrian is moving (speed >= 0.1 m/s)
        current_time = self.get_clock().now()

        # Start patience timer if not already running
        if self.wait_start_time is None:
            self.wait_start_time = current_time
            self.publish_decision(
                "STOP_WATCH",
                f"Pedestrian moving (speed: {pedestrian_speed:.2f} m/s) - Starting patience timer"
            )
            return

        # Check elapsed time on patience timer
        elapsed_time = (current_time - self.wait_start_time).nanoseconds / 1e9

        if elapsed_time > self.WAIT_PATIENCE_TIMEOUT:
            self.publish_decision(
                "CREEP_PASS",
                f"Waited {elapsed_time:.1f}s - Carefully passing"
            )
        else:
            self.publish_decision(
                "STOP_WATCH",
                f"Watching moving pedestrian ({elapsed_time:.1f}s/{self.WAIT_PATIENCE_TIMEOUT}s)"
            )

    def print_statistics(self):
        """
        Print decision statistics for debugging.
        """
        total = sum(self.decision_counts.values())
        if total > 0:
            self.get_logger().info('=== Decision Statistics ===')
            for state, count in self.decision_counts.items():
                percentage = (count / total) * 100
                self.get_logger().info(f'{state}: {count} ({percentage:.1f}%)')

    def destroy_node(self):
        """
        Cleanup before shutting down.
        """
        self.print_statistics()
        super().destroy_node()


def main(args=None):
    """
    Main entry point for the high-level decision node.
    """
    rclpy.init(args=args)

    try:
        node = HighLevelDecisionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in high-level decision node: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
