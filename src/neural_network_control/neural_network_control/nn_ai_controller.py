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


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

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
        self.nn_reference_path = os.path.join(self.base_path, 'nn_ai_reference_lap.csv')
        self.step_response_log_path = os.path.join(self.base_path, 'nn_ai_step_response.csv')

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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # Single-target step response configuration.
        self.single_target_mode = False
        self.target_left_offset_m = 5.0
        self.target_initialized = True
        self.target_init_x = None
        self.target_init_y = None
        self.target_init_yaw = None
        self.target_x =0.0
        self.target_y = 0.0
        self.target_yaw_deg = 0.0
        self.target_yaw = float(np.deg2rad(self.target_yaw_deg))
        self.target_tolerance_m = 0.20
        self.target_yaw_tolerance_rad = 0.12
        self.heading_align_radius_m = 0.8
        self.step_timeout_s = 30.0
        self.step_start_time = time.time()
        self.step_finished = False
        self.step_rows = []
        self.step_target_reached = False
        self.step_target_reached_time = None
        self.step_stop_speed_threshold = 0.15
        self.step_stop_hold_s = 0.4
        self.step_stop_wait_timeout_s = 5.0
        self.step_below_speed_since = None

        # ROS 2 Pubs/Subs
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(Marker, '/global_waypoints_marker', 10) 
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.viz_timer = self.create_timer(2.0, self.publish_waypoints_marker)
        self.curr_x, self.curr_y, self.yaw, self.v_curr = 0.0, 0.0, 0.0, 0.0
        if self.single_target_mode:
            self.get_logger().info(
                f"🎯 Single-target mode active: x={self.target_x:.2f}, y={self.target_y:.2f}, yaw={self.target_yaw:.2f}"
            )

    def initialize_relative_target(self):
        if self.target_initialized:
            return

        self.target_init_x = float(self.curr_x)
        self.target_init_y = float(self.curr_y)
        self.target_init_yaw = float(self.yaw)

        # Left direction in map frame from initial heading.
        left_x = -np.sin(self.target_init_yaw)
        left_y = np.cos(self.target_init_yaw)

        self.target_x = self.target_init_x + self.target_left_offset_m * left_x
        self.target_y = self.target_init_y + self.target_left_offset_m * left_y
        self.target_yaw = self.target_init_yaw

        self.step_start_time = time.time()
        self.target_initialized = True
        self.get_logger().info(
            f"🎯 Relative target initialized from start ({self.target_init_x:.2f}, {self.target_init_y:.2f}, yaw={self.target_init_yaw:.2f}) "
            f"-> target ({self.target_x:.2f}, {self.target_y:.2f})"
        )

    def save_reference_csv(self, lap_time):
        """Save latest NN trajectory as XY for multi-controller overlay plotting."""
        if len(self.actual_x) < 2:
            return
        df = pd.DataFrame({
            'x': np.array(self.actual_x),
            'y': np.array(self.actual_y),
            'lap_time_s': np.full(len(self.actual_x), lap_time)
        })
        df.to_csv(self.nn_reference_path, index=False)
        self.get_logger().info(f"💾 Saved NN AI XY trajectory CSV: {self.nn_reference_path}")

        lap_path = os.path.join(self.base_path, f'nn_ai_reference_lap_{self.lap_count}.csv')
        df.to_csv(lap_path, index=False)
        self.get_logger().info(f"💾 Saved NN AI lap-specific XY CSV: {lap_path}")

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

        if self.single_target_mode and not self.step_finished:
            now = time.time()
            dist_to_target = float(np.hypot(self.target_x - self.curr_x, self.target_y - self.curr_y))
            yaw_error = abs(wrap_to_pi(self.target_yaw - self.yaw))
            elapsed = now - self.step_start_time

            reached_target = dist_to_target <= self.target_tolerance_m and yaw_error <= self.target_yaw_tolerance_rad
            if reached_target and not self.step_target_reached:
                self.step_target_reached = True
                self.step_target_reached_time = now
                self.step_below_speed_since = None
                self.get_logger().info("🎯 Target reached, continuing to log until vehicle fully stops.")

            if not self.step_target_reached and elapsed >= self.step_timeout_s:
                self.finish_step_response(status='timeout')
                return

            if self.step_target_reached:
                if abs(self.v_curr) <= self.step_stop_speed_threshold:
                    if self.step_below_speed_since is None:
                        self.step_below_speed_since = now
                    elif (now - self.step_below_speed_since) >= self.step_stop_hold_s:
                        self.finish_step_response(status='vehicle_stopped')
                        return
                else:
                    self.step_below_speed_since = None

                if self.step_target_reached_time is not None and (now - self.step_target_reached_time) >= self.step_stop_wait_timeout_s:
                    self.finish_step_response(status='target_reached_stop_timeout')
                    return
            return

        curr_pos = np.array([self.curr_x, self.curr_y])
        if self.prev_pos is not None:
            self.total_dist_traveled += np.linalg.norm(curr_pos - self.prev_pos)
            dist_to_start = np.linalg.norm(curr_pos - self.start_line)
            
            if dist_to_start < 1.2 and self.total_dist_traveled > 15.0:
                self.lap_count += 1
                lap_duration = time.time() - self.lap_start_time
                self.save_reference_csv(lap_duration)
                self.save_lap_plot(lap_duration)
                self.compute_and_save_metrics(lap_duration)
                
                # Reset buffers for next lap
                self.actual_x, self.actual_y, self.actual_yaw = [], [], []
                self.desired_x, self.desired_y, self.desired_yaw, self.cte_history = [], [], [], []
                self.lap_start_time = time.time()
                self.total_dist_traveled = 0.0
        
        self.prev_pos = curr_pos

    def finish_step_response(self, status='target_reached'):
        if self.step_finished:
            return
        self.step_finished = True

        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(stop_msg)

        if self.step_rows:
            pd.DataFrame(self.step_rows).to_csv(self.step_response_log_path, index=False)
            self.get_logger().info(f"💾 Step response CSV saved: {self.step_response_log_path}")

        self.get_logger().info(f"✅ NN AI step run finished ({status}).")

    def scan_callback(self, scan_msg):
        if self.single_target_mode:
            if self.step_finished:
                return

            if self.step_target_reached:
                stop_msg = AckermannDriveStamped()
                stop_msg.drive.speed = 0.0
                stop_msg.drive.steering_angle = 0.0
                self.drive_pub.publish(stop_msg)

                elapsed = time.time() - self.step_start_time
                err_yaw = wrap_to_pi(self.target_yaw - self.yaw)
                dist_to_target = float(np.hypot(self.target_x - self.curr_x, self.target_y - self.curr_y))
                self.step_rows.append({
                    'time_s': elapsed,
                    'desired_x': self.target_x,
                    'desired_y': self.target_y,
                    'desired_yaw': self.target_yaw,
                    'actual_x': float(self.curr_x),
                    'actual_y': float(self.curr_y),
                    'actual_yaw': float(self.yaw),
                    'x_error': float(self.target_x - self.curr_x),
                    'y_error': float(self.target_y - self.curr_y),
                    'yaw_error': float(err_yaw),
                    'distance_error': dist_to_target,
                    'speed_cmd': 0.0,
                    'steering_cmd': 0.0,
                    'speed_actual': float(self.v_curr),
                })
                return

            dx_g = self.target_x - self.curr_x
            dy_g = self.target_y - self.curr_y
            dist_to_target = float(np.hypot(dx_g, dy_g))
            err_yaw_target = wrap_to_pi(self.target_yaw - self.yaw)
            heading_align_active = (
                dist_to_target <= self.heading_align_radius_m
                and abs(err_yaw_target) > self.target_yaw_tolerance_rad
            )
            if heading_align_active:
                path_yaw = self.target_yaw
                err_yaw = err_yaw_target
            else:
                path_yaw = np.arctan2(dy_g, dx_g)
                err_yaw = wrap_to_pi(path_yaw - self.yaw)
            dx_local = dx_g * np.cos(self.yaw) + dy_g * np.sin(self.yaw)
            dy_local = -dx_g * np.sin(self.yaw) + dy_g * np.cos(self.yaw)

            self.actual_x.append(self.curr_x)
            self.actual_y.append(self.curr_y)
            self.actual_yaw.append(self.yaw)
            self.desired_x.append(self.target_x)
            self.desired_y.append(self.target_y)
            self.desired_yaw.append(path_yaw)
            self.cte_history.append(dist_to_target)

            scan_ranges = np.nan_to_num(np.array(scan_msg.ranges), nan=0.0, posinf=10.0)
            idx_samples = np.round(np.linspace(0, len(scan_ranges) - 1, 20)).astype(int)
            lidar_samples = scan_ranges[idx_samples]

            features = np.array([self.v_curr, dx_local, dy_local, err_yaw] + list(lidar_samples)).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                prediction = self.model(features_tensor).cpu().numpy()[0]

            steer_cmd = float(np.clip(prediction[0], -0.41, 0.41))
            speed_cmd = float(np.clip(prediction[1], 0.0, 5.5))
            if dist_to_target > self.target_tolerance_m * 2.0:
                speed_cmd = max(speed_cmd, 0.45)
            if heading_align_active:
                speed_cmd = min(speed_cmd, 1.0)
                speed_cmd = max(speed_cmd, 0.35)

            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = speed_cmd
            drive_msg.drive.steering_angle = steer_cmd
            self.drive_pub.publish(drive_msg)

            elapsed = time.time() - self.step_start_time
            self.step_rows.append({
                'time_s': elapsed,
                'desired_x': self.target_x,
                'desired_y': self.target_y,
                'desired_yaw': self.target_yaw,
                'actual_x': float(self.curr_x),
                'actual_y': float(self.curr_y),
                'actual_yaw': float(self.yaw),
                'x_error': float(self.target_x - self.curr_x),
                'y_error': float(self.target_y - self.curr_y),
                'yaw_error': float(err_yaw_target),
                'distance_error': dist_to_target,
                'speed_cmd': speed_cmd,
                'steering_cmd': steer_cmd,
                'speed_cmd_raw': float(prediction[1]),
                'steering_cmd_raw': float(prediction[0]),
                'heading_align_active': bool(heading_align_active),
                'speed_actual': float(self.v_curr),
            })
            return

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