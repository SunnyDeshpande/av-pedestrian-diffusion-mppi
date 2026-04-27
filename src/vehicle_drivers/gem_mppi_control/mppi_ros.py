#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import torch
import numpy as np
import math
from pytorch_mppi import mppi

# PACMod & Sensor Messages
from pacmod2_msgs.msg import PositionWithSpeed, VehicleSpeedRpt, GlobalCmd, SystemCmdFloat
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod
import pymap3d as pm

class TorchMPPINode(Node):
    def __init__(self):
        super().__init__('gem_torch_mppi_node')

        # --- Device Configuration ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # --- MPPI Parameters ---
        self.dt = 0.1
        self.horizon = 30
        self.num_samples = 1000
        
        # --- Bicycle Model Constants ---
        self.L = 2.57
        self.max_steer = math.radians(35)
        self.max_accel = 2.0

        # --- Cost Weights (Easily Tunable) ---
        self.w_pos = 15.0
        self.w_vel = 5.0
        self.w_curv = 2.0  # Penalty for high steering at high speeds

        # Initialize State: [x, y, yaw, v]
        self.state = torch.zeros(4, device=self.device)
        self.goal = torch.tensor([0.0, 0.0, 0.0, 2.0], device=self.device) # [x, y, yaw, v_target]

        # Setup MPPI
        self.ctrl = mppi.MPPI(
            dynamics=self.dynamics_model,
            running_cost=self.running_cost,
            nx=4,
            terminal_state_cost=None,
            num_samples=self.num_samples,
            horizon=self.horizon,
            device=self.device,
            u_min=torch.tensor([-1.0, -self.max_steer], device=self.device),
            u_max=torch.tensor([self.max_accel, self.max_steer], device=self.device),
            noise_sigma=torch.tensor([[0.5, 0.0], [0.0, 0.2]], device=self.device), # Variance
            lambda_=0.1
        )

        self.init_comms()
        self.create_timer(self.dt, self.control_loop)

    def dynamics_model(self, state, u):
        """
        Bicycle Model
        state: (N, 4) -> [x, y, yaw, v]
        u: (N, 2)     -> [accel, delta]
        """
        # Unpack for readability
        yaw = state[:, 2]
        v = state[:, 3]
        accel = u[:, 0]
        delta = u[:, 1]

        new_state = torch.zeros_like(state)
        new_state[:, 0] = state[:, 0] + v * torch.cos(yaw) * self.dt
        new_state[:, 1] = state[:, 1] + v * torch.sin(yaw) * self.dt
        new_state[:, 2] = state[:, 2] + (v / self.L) * torch.tan(delta) * self.dt
        new_state[:, 3] = state[:, 3] + accel * self.dt
        
        return new_state

    def running_cost(self, state, u):
        """
        Vectorized cost function
        state: (N, 4)
        """
        # Distance to goal (x, y)
        pos_err = torch.norm(state[:, :2] - self.goal[:2], dim=1)
        
        # Velocity error
        vel_err = torch.abs(state[:, 3] - self.goal[3])
        
        # Stability penalty: discourage high steering at high speeds
        stability_penalty = torch.abs(u[:, 1]) * state[:, 3]

        return (self.w_pos * pos_err) + (self.w_vel * vel_err) + (self.w_curv * stability_penalty)

    def control_loop(self):
        # Calculate optimal control
        # Command returns the first action in the optimized sequence
        action = self.ctrl.command(self.state)
        
        # Convert torch action to PACMod commands
        self.publish_pacmod(action.cpu().numpy())

    def publish_pacmod(self, u):
        # u[0] = accel, u[1] = steering angle (rad)
        accel_cmd = SystemCmdFloat(command=float(np.clip(u[0], 0.0, self.max_accel)))
        
        # Steering Polynomial (Front wheel deg to steering wheel rad)
        f_deg = math.degrees(u[1])
        sw_deg = -0.1084 * abs(f_deg)**2 + 21.775 * abs(f_deg)
        sw_rad = math.radians(sw_deg if f_deg >= 0 else -sw_deg)
        
        steer_cmd = PositionWithSpeed(angular_position=sw_rad, angular_velocity_limit=4.0)
        
        self.accel_pub.publish(accel_cmd)
        self.steer_pub.publish(steer_cmd)

    # --- Standard ROS Callbacks (GNSS/INS/Speed) ---
    def init_comms(self):
        self.create_subscription(NavSatFix, '/navsatfix', self.gnss_cb, 10)
        self.create_subscription(INSNavGeod, '/insnavgeod', self.ins_cb, 10)
        self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt', self.speed_cb, 10)
        self.accel_pub = self.create_publisher(SystemCmdFloat, '/pacmod/accel_cmd', 10)
        self.steer_pub = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 10)

    def gnss_cb(self, msg):
        olat, olon = 40.0927422, -88.2359639
        x, y, _ = pm.geodetic2enu(msg.latitude, msg.longitude, 0, olat, olon, 0)
        self.state[0] = x
        self.state[1] = y

    def ins_cb(self, msg):
        # Convert heading to Yaw
        self.state[2] = np.radians(90 - msg.heading) if msg.heading < 270 else np.radians(450 - msg.heading)

    def speed_cb(self, msg):
        self.state[3] = msg.vehicle_speed

def main():
    rclpy.init()
    node = TorchMPPINode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()