import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pytorch_mppi import mppi

class MPPIConfidenceGrowthSim:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Sim Params ---
        self.dt = 0.1
        self.horizon = 40 
        self.num_samples = 600
        self.L = 2.57
        
        # Cones: [x, y, radius]
        self.cones = torch.tensor([[10.0, 7.0, 1.2], [16.0, 11.0, 0.7], [10.0, 15.0, 0.7], [16.0, 19.0, 0.7], [22.0, 15.0, 0.7], [28.0, 19.0, 0.7]], device=self.device)

        # Pedestrians: [x, y, vx, vy, initial_confidence]
        self.peds = torch.tensor([[4.0, 10.0, 0.4, -0.8, 0.8]], device=self.device)

        self.state = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        self.goal = torch.tensor([25.0, 20.0, 0.0, 0.0], device=self.device)

        self.ctrl = mppi.MPPI(
            dynamics=self.dynamics_model,
            running_cost=self.running_cost,
            nx=4,
            num_samples=self.num_samples,
            horizon=self.horizon,
            device=self.device,
            u_min=torch.tensor([-1.5, -math.radians(35)], device=self.device),
            u_max=torch.tensor([2.0, math.radians(35)], device=self.device),
            noise_sigma=torch.tensor([[0.5, 0.0], [0.0, 0.15]], device=self.device),
            lambda_=0.1
        )

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def dynamics_model(self, state, u):
        yaw, v, accel, delta = state[:, 2], state[:, 3], u[:, 0], u[:, 1]
        new_state = torch.zeros_like(state)
        new_state[:, 0] = state[:, 0] + v * torch.cos(yaw) * self.dt
        new_state[:, 1] = state[:, 1] + v * torch.sin(yaw) * self.dt
        new_state[:, 2] = state[:, 2] + (v / self.L) * torch.tan(delta) * self.dt
        new_state[:, 3] = state[:, 3] + accel * self.dt
        return new_state

    def running_cost(self, state, u):
        """
        state: (N, 4) - Note: MPPI passes the state for a specific timestep 't'
        To handle the temporal change, we need to know WHICH timestep we are in.
        pytorch-mppi doesn't explicitly pass 't', so we approximate the 
        pedestrian's future ground truth based on the car's distance/velocity 
        or simply track the internal rollout if using a custom loop. 
        
        For this implementation, we calculate the cost based on the 
        Current State + Ground Truth Prediction.
        """
        pos_err = torch.norm(state[:, :2] - self.goal[:2], dim=1)
        cost_obst = torch.zeros(state.shape[0], device=self.device)

        # 1. Cone Costs
        for cone in self.cones:
            dist = torch.norm(state[:, :2] - cone[:2], dim=1)
            cost_obst += torch.where(dist < cone[2], 250.0, 0.0) + 40.0 * torch.exp(-dist)

        # 2. Pedestrian Ground Truth Projection with Confidence Growth
        # Note: In a real 'running_cost' that MPPI calls, it evaluates all samples 
        # for a specific step. Here we use the pedestrian's current velocity.
        for ped in self.peds:
            px, py, vx, vy, start_conf = ped
            
            # Estimate which 't' we are at based on car's state (simplified)
            # A more robust way is to pass 't' if using a custom MPPI wrapper
            t_eff = torch.mean(torch.norm(state[:, :2] - self.state[:2], dim=1)) / (self.state[3] + 1e-3)
            t_eff = torch.clamp(t_eff, 0, self.horizon * self.dt)

            # Ground Truth position at time t
            gt_x = px + vx * t_eff
            gt_y = py + vy * t_eff
            
            # Confidence grows over the horizon
            # start_conf at t=0, approaching 1.0 at the end of horizon
            current_conf = torch.clamp(start_conf + (t_eff / (self.horizon * self.dt)) * (1.0 - start_conf), 0, 1)

            base_sigma = 1.0 + (1.0 - current_conf) * 3.0
            sigma_long = base_sigma * (1.0 + (1.0 - current_conf) * 2.0)
            sigma_lat = base_sigma
            
            mu_x = sigma_long * 1.5 * current_conf 

            dx, dy = state[:, 0] - gt_x, state[:, 1] - gt_y
            angle = torch.atan2(vy, vx)
            dx_rot = dx * torch.cos(angle) + dy * torch.sin(angle)
            dy_rot = -dx * torch.sin(angle) + dy * torch.cos(angle)
            
            exponent = -((dx_rot - mu_x)**2 / (2 * sigma_long**2) + (dy_rot)**2 / (2 * sigma_lat**2))
            cost_obst += 150.0 * torch.exp(exponent)

        return (15.0 * pos_err) + cost_obst

    def plot_scene(self):
        self.ax.cla()
        
        # Plot Sampled Trajectories
        if self.ctrl.perturbed_action is not None:
            samples = self.ctrl.perturbed_action[::20] 
            for sample_u in samples:
                traj = self.rollout_trajectory(self.state, sample_u)
                self.ax.plot(traj[:, 0], traj[:, 1], color='cyan', alpha=0.4, linewidth=0.5)

        # Plot Cones
        for cone in self.cones:
            self.ax.add_patch(plt.Circle((cone[0].item(), cone[1].item()), cone[2].item(), color='orange'))

        # Plot Pedestrian and Temporal Risk Map
        for ped in self.peds:
            px, py, vx, vy, conf = ped.tolist()
            self.ax.scatter(px, py, color='red', s=100, zorder=11)
            
            # Show "Now" Risk (Smashed)
            self.draw_ped_ellipse(px, py, vx, vy, conf, alpha=0.3, label="Current Risk")
            
            # Show "Future" Risk (Converged Ground Truth at Horizon End)
            future_x = px + vx * self.horizon * self.dt
            future_y = py + vy * self.horizon * self.dt
            self.draw_ped_ellipse(future_x, future_y, vx, vy, 1.0, alpha=0.1, label="Future GT")

        best_u = self.ctrl.get_action_sequence()
        best_traj = self.rollout_trajectory(self.state, best_u)
        self.ax.plot(best_traj[:, 0], best_traj[:, 1], color='blue', linewidth=3)
        self.ax.scatter(self.goal[0].item(), self.goal[1].item(), color='green', marker='X', s=200)
        
        self.ax.set_xlim(-2, 30); self.ax.set_ylim(-2, 30)
        self.ax.set_title(f"Temporal Confidence Growth | Start Conf: {conf:.2f}")
        plt.grid()
        plt.pause(0.001)

    def draw_ped_ellipse(self, x, y, vx, vy, conf, alpha, label=""):
        base_s = 1.0 + (1.0 - conf) * 3.0
        s_long = base_s * (1.0 + (1.0 - conf) * 2.0)
        angle_deg = math.degrees(math.atan2(vy, vx))
        mu_x = s_long * 1.5 * conf
        cx = x + mu_x * math.cos(math.radians(angle_deg))
        cy = y + mu_x * math.sin(math.radians(angle_deg))
        ellipse = Ellipse((cx, cy), s_long*2.5, base_s*2.5, angle=angle_deg, color='red', alpha=alpha)
        self.ax.add_patch(ellipse)

    def rollout_trajectory(self, start_state, u_seq):
        states = [start_state[:2].cpu().numpy()]
        curr_s = start_state.clone().unsqueeze(0)
        for t in range(u_seq.shape[0]):
            curr_s = self.dynamics_model(curr_s, u_seq[t].unsqueeze(0))
            states.append(curr_s[0, :2].cpu().detach().numpy())
        return np.array(states)

    def run(self):
        try:
            while torch.norm(self.state[:2] - self.goal[:2]) > 0.8:
                action = self.ctrl.command(self.state)
                self.plot_scene()
                self.state = self.dynamics_model(self.state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
                # Update Ground Truth in real-time
                self.peds[:, 0:2] += self.peds[:, 2:4] * self.dt
            plt.show(block=True)
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    sim = MPPIConfidenceGrowthSim()
    sim.run()