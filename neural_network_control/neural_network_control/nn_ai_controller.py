import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import pickle
import time
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker 
from geometry_msgs.msg import Point 
from tf_transformations import euler_from_quaternion

class F1TENTH_PINN(nn.Module):
    def __init__(self):
        super(F1TENTH_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

class PINNInferenceNode(Node):
    def __init__(self):
        super().__init__('pinn_inference_node')
        
        # Paths
        self.base_path = os.path.expanduser('~/sim_ws/src/neural_network_control/neural_network_control/')
        model_path = os.path.join(self.base_path, 'nn_model_v2.pth')
        scaler_path = os.path.join(self.base_path, 'scaler_v2.pkl')
        waypoints_path = os.path.join(self.base_path, 'waypoints.csv')
        self.pp_reference_path = os.path.join(self.base_path, 'pp_reference_lap.csv')

        # Load environment and model data
        self.waypoints_df = pd.read_csv(waypoints_path)
        self.waypoints = self.waypoints_df[['x', 'y']].values
        
        self.pp_data = None
        if os.path.exists(self.pp_reference_path):
            self.pp_data = pd.read_csv(self.pp_reference_path)
            self.get_logger().info("🔵 Loaded Pure Pursuit Reference data for fair comparison.")
        else:
            self.get_logger().warn("⚠️ pp_reference_lap.csv not found.")

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.device = torch.device("cuda" if torch.torch.cuda.is_available() else "cpu")
        self.model = F1TENTH_PINN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.get_logger().info(f"🚀 PINN Model Loaded on {self.device}.")

        # Metrics and Buffers
        self.lap_count = 0
        self.lap_start_time = time.time()
        self.total_dist_traveled = 0.0
        self.prev_pos = None
        self.start_line = self.waypoints[0]
        
        self.actual_x, self.actual_y, self.actual_yaw = [], [], []
        self.desired_x, self.desired_y, self.desired_yaw, self.cte_history = [], [], [], []

        # ROS 2 Pubs/Subs
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(Marker, '/global_waypoints_marker', 10) 
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.viz_timer = self.create_timer(2.0, self.publish_waypoints_marker)
        self.curr_x, self.curr_y, self.yaw, self.v_curr = 0.0, 0.0, 0.0, 0.0

    def save_lap_plot(self, lap_time):
        """Generates a comparison plot including Desired Path, PINN, and PP against Lap Progress."""
        fig, axs = plt.subplots(4, 1, figsize=(10, 18))
        fig.suptitle(f'Spatial Tracking Accuracy | PINN Lap Time: {lap_time:.3f}s', fontsize=16)

        # 1. Generate Progress Axes (0 to 100%)
        nn_p = np.linspace(0, 100, len(self.actual_x))
        ref_p = np.linspace(0, 100, len(self.desired_x))
        
        pp_p, pp_x, pp_y, pp_yaw, pp_cte = None, None, None, None, None
        if self.pp_data is not None:
            pp_p = self.pp_data['progress'].values if 'progress' in self.pp_data.columns else np.linspace(0, 100, len(self.pp_data))
            pp_x, pp_y = self.pp_data['pp_x'].values, self.pp_data['pp_y'].values
            pp_yaw, pp_cte = self.pp_data['pp_yaw'].values, self.pp_data['pp_cte'].values

        # 2. Plotting Logic
        labels = ['X (m)', 'Y (m)', 'Yaw (rad)', 'CTE (m)']
        nn_data = [self.actual_x, self.actual_y, self.actual_yaw, self.cte_history]
        ref_data = [self.desired_x, self.desired_y, self.desired_yaw, None]
        pp_data = [pp_x, pp_y, pp_yaw, pp_cte]

        for i in range(4):
            # Plot Desired Path (Reference) - Ground Truth
            if ref_data[i] is not None:
                axs[i].plot(ref_p, ref_data[i], 'g', label='Desired Path', alpha=0.3, linewidth=3)
            
            # Plot PINN Data
            axs[i].plot(nn_p, nn_data[i], 'r--', label='PINN (Proposed)')
            
            # Plot Pure Pursuit Data
            if pp_data[i] is not None:
                axs[i].plot(pp_p, pp_data[i], 'b:', label='Pure Pursuit (Baseline)')
            
            axs[i].set_ylabel(labels[i])
            axs[i].grid(True)
            if i == 0:
                axs[i].legend(loc='upper right')
            if i == 3:
                axs[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axs[i].set_xlabel('Lap Progress (%)')

        save_name = os.path.join(self.base_path, f'lap_{self.lap_count}_full_comparison.png')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_name)
        plt.close(fig)
        self.get_logger().info(f"✅ Full comparison plot saved: {save_name}")

    def compute_and_save_metrics(self, lap_duration):
        """Calculate quantitative performance metrics for comparison"""
        if len(self.actual_x) < 2:
            return
        
        # Convert to numpy arrays
        actual_x = np.array(self.actual_x)
        actual_y = np.array(self.actual_y)
        desired_x = np.array(self.desired_x)
        desired_y = np.array(self.desired_y)
        cte = np.array(self.cte_history)
        
        # Compute min length for alignment
        min_len = min(len(actual_x), len(desired_x))
        actual_x = actual_x[:min_len]
        actual_y = actual_y[:min_len]
        desired_x = desired_x[:min_len]
        desired_y = desired_y[:min_len]
        cte = cte[:min_len]
        
        # **Positional Errors**
        x_error = np.abs(actual_x - desired_x)
        y_error = np.abs(actual_y - desired_y)
        positional_distance = np.sqrt(x_error**2 + y_error**2)
        
        rms_x_error = np.sqrt(np.mean(x_error**2))
        rms_y_error = np.sqrt(np.mean(y_error**2))
        rms_positional_error = np.sqrt(np.mean(positional_distance**2))
        
        mean_x_error = np.mean(x_error)
        mean_y_error = np.mean(y_error)
        max_x_error = np.max(x_error)
        max_y_error = np.max(y_error)
        
        # **Cross-Track Error Statistics**
        rms_cte = np.sqrt(np.mean(cte**2))
        mean_cte = np.mean(np.abs(cte))
        max_cte = np.max(np.abs(cte))
        
        # **Steering smoothness** (rate of change)
        if len(self.actual_yaw) > 1:
            yaw_rates = np.diff(np.array(self.actual_yaw))
            steering_smoothness = np.std(yaw_rates)  # Lower is smoother
        else:
            steering_smoothness = 0.0
        
        # **Sequential consistency** (correlation)
        if len(self.desired_yaw) > 10:
            desired_yaw = np.array(self.desired_yaw[:min_len])
            actual_yaw = np.array(self.actual_yaw[:min_len])
            correlation = np.corrcoef(desired_yaw, actual_yaw)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Save metrics to CSV
        metrics_dict = {
            'controller': 'PINN_v2',
            'lap': self.lap_count,
            'lap_time_s': lap_duration,
            'samples': min_len,
            'rms_x_error_m': rms_x_error,
            'rms_y_error_m': rms_y_error,
            'rms_positional_error_m': rms_positional_error,
            'mean_x_error_m': mean_x_error,
            'mean_y_error_m': mean_y_error,
            'max_x_error_m': max_x_error,
            'max_y_error_m': max_y_error,
            'rms_cte_m': rms_cte,
            'mean_cte_m': mean_cte,
            'max_cte_m': max_cte,
            'steering_smoothness_rad_per_sample': steering_smoothness,
            'trajectory_correlation': correlation
        }
        
        # Append to metrics file
        metrics_file = os.path.join(self.base_path, 'performance_metrics.csv')
        if os.path.exists(metrics_file):
            df_existing = pd.read_csv(metrics_file)
            df_new = pd.DataFrame([metrics_dict])
            df_metrics = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_metrics = pd.DataFrame([metrics_dict])
        
        df_metrics.to_csv(metrics_file, index=False)
        
        # Log summary
        self.get_logger().info(f"📊 PINN_v2 Lap {self.lap_count} Metrics:")
        self.get_logger().info(f"   RMS Error: X={rms_x_error:.4f}m, Y={rms_y_error:.4f}m, Total={rms_positional_error:.4f}m")
        self.get_logger().info(f"   CTE: Mean={mean_cte:.4f}m, RMS={rms_cte:.4f}m, Max={max_cte:.4f}m")
        self.get_logger().info(f"   Steering Smoothness: {steering_smoothness:.4f} rad/sample")
        self.get_logger().info(f"   Trajectory Correlation: {correlation:.4f}")

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.v_curr = msg.twist.twist.linear.x
        orient = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

        curr_pos = np.array([self.curr_x, self.curr_y])
        if self.prev_pos is not None:
            self.total_dist_traveled += np.linalg.norm(curr_pos - self.prev_pos)
            dist_to_start = np.linalg.norm(curr_pos - self.start_line)
            
            if dist_to_start < 1.2 and self.total_dist_traveled > 15.0:
                self.lap_count += 1
                lap_duration = time.time() - self.lap_start_time
                self.save_lap_plot(lap_duration)
                self.compute_and_save_metrics(lap_duration)
                
                # Reset buffers for next lap
                self.actual_x, self.actual_y, self.actual_yaw = [], [], []
                self.desired_x, self.desired_y, self.desired_yaw, self.cte_history = [], [], [], []
                self.lap_start_time = time.time()
                self.total_dist_traveled = 0.0
        
        self.prev_pos = curr_pos

    def scan_callback(self, scan_msg):
        # Look-ahead logic for reference comparison
        Ld_feat = 1.5 
        dists = np.linalg.norm(self.waypoints - np.array([self.curr_x, self.curr_y]), axis=1)
        closest_idx = np.argmin(dists)
        
        target_idx = closest_idx
        for i in range(closest_idx, closest_idx + 100):
            idx = i % len(self.waypoints)
            if dists[idx] >= Ld_feat:
                target_idx = idx
                break
        
        target_pt = self.waypoints[target_idx]
        dx_g, dy_g = target_pt[0] - self.curr_x, target_pt[1] - self.curr_y
        dx_local = dx_g * np.cos(self.yaw) + dy_g * np.sin(self.yaw)
        dy_local = -dx_g * np.sin(self.yaw) + dy_g * np.cos(self.yaw)
        path_yaw = np.arctan2(dy_g, dx_g)
        err_yaw = (path_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi

        # Data Recording
        self.actual_x.append(self.curr_x); self.actual_y.append(self.curr_y); self.actual_yaw.append(self.yaw)
        self.desired_x.append(target_pt[0]); self.desired_y.append(target_pt[1]); self.desired_yaw.append(path_yaw)
        
        current_cte = np.min(dists)
        side = (target_pt[0] - self.curr_x) * np.sin(self.yaw) - (target_pt[1] - self.curr_y) * np.cos(self.yaw)
        self.cte_history.append(current_cte if side > 0 else -current_cte)

        # Inference
        scan_ranges = np.nan_to_num(np.array(scan_msg.ranges), nan=0.0, posinf=10.0)
        idx_samples = np.round(np.linspace(0, len(scan_ranges) - 1, 20)).astype(int)
        lidar_samples = scan_ranges[idx_samples]

        features = np.array([self.v_curr, dx_local, dy_local, err_yaw] + list(lidar_samples)).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            prediction = self.model(features_tensor).cpu().numpy()[0]
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed, drive_msg.drive.steering_angle = float(prediction[1]), float(prediction[0])
        self.drive_pub.publish(drive_msg)

    def publish_waypoints_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"; marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "global_path"; marker.id = 0; marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD; marker.scale.x = 0.1; marker.color.a, marker.color.g = 1.0, 1.0
        for x, y in self.waypoints:
            p = Point(); p.x, p.y, p.z = float(x), float(y), 0.05 
            marker.points.append(p)
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args); node = PINNInferenceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()