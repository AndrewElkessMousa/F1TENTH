import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
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

class PINNDriveNode(Node):
    def __init__(self):
        super().__init__('pinn_drive')
        
        # Paths
        self.package_share = get_package_share_directory('neural_network_control')
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(self.package_share, 'pinn_model_weights.pth')
        scaler_path = os.path.join(self.package_share, 'pinn_scaler.pkl')

        default_waypoints = os.path.join(self.package_share, 'center_line_sp.csv')
        waypoints_path = self.declare_parameter('waypoints_path', default_waypoints).value
        self.waypoints_topic = self.declare_parameter('waypoints_topic', '/center_line_path').value
        self.center_line_received = False

        # Load Waypoints from file as a default fallback
        df_wp = pd.read_csv(waypoints_path, comment='#', header=None)
        if df_wp.shape[1] < 2:
            raise ValueError(f"Waypoint CSV must have at least 2 columns: {waypoints_path}")
        wp_xy = df_wp.iloc[:, :2].apply(pd.to_numeric, errors='coerce').dropna()
        if wp_xy.empty:
            raise ValueError(f"No valid waypoint rows found in: {waypoints_path}")
        self.waypoints = wp_xy.to_numpy(dtype=float)
        self.start_line = self.waypoints[0]
        self.get_logger().info(f'Loaded {len(self.waypoints)} center line waypoints from: {waypoints_path}')

        # Load Model and Scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = F1TENTH_PINN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Plotting Buffers
        self.actual_x, self.actual_y, self.actual_yaw = [], [], []
        self.desired_x, self.desired_y, self.desired_yaw = [], [], []
        self.cte_history = []
        self.timestamps = []
        
        # Lap Logic
        self.lap_count = 0
        self.lap_start_time = time.time()
        self.total_dist_traveled = 0.0
        self.prev_pos = None
        self.plot_output_dir = os.path.join(self.base_path, 'lap_plots')
        self.pinn_reference_path = os.path.join(self.base_path, 'pinn_reference_lap.csv')
        os.makedirs(self.plot_output_dir, exist_ok=True)

        self.last_estimated_pose = None
        self.last_estimated_pose_time = None

        # ROS Setup
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.odom_sub = self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.odom_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.path_sub = self.create_subscription(Path, self.waypoints_topic, self.center_line_path_cb, 10)
        self.path_viz_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        
        self.curr_x, self.curr_y, self.yaw, self.v_curr = 0.0, 0.0, 0.0, 0.0
        self.get_logger().info("🚀 PINN Drive with CTE Tracking Started.")
        self.get_logger().info(f'Listening for center line path on: {self.waypoints_topic}')
        
        # Publish Global Path Visualization (continuously on timer)
        self.path_timer = self.create_timer(1.0, self.publish_path_visualization)

    def save_performance_plot(self, lap_time):
        fig, axs = plt.subplots(4, 1, figsize=(10, 16)) # Increased size for 4th plot
        t = np.array(self.timestamps) - self.timestamps[0]
        
        # X Plot
        axs[0].plot(t, self.desired_x, 'g', label='Desired X', linewidth=2)
        axs[0].plot(t, self.actual_x, 'r--', label='Actual X (PINN)')
        axs[0].set_ylabel('X (m)')
        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        # Y Plot
        axs[1].plot(t, self.desired_y, 'g', label='Desired Y', linewidth=2)
        axs[1].plot(t, self.actual_y, 'r--', label='Actual Y (PINN)')
        axs[1].set_ylabel('Y (m)')
        axs[1].grid(True)

        # Yaw Plot
        axs[2].plot(t, self.desired_yaw, 'g', label='Desired Yaw', linewidth=2)
        axs[2].plot(t, self.actual_yaw, 'r--', label='Actual Yaw (PINN)')
        axs[2].set_ylabel('Yaw (rad)')
        axs[2].grid(True)

        # CTE Plot (Cross-Track Error)
        axs[3].plot(t, self.cte_history, 'b', label='Cross-Track Error')
        axs[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[3].set_ylabel('CTE (m)')
        axs[3].set_xlabel('Time (s)')
        axs[3].grid(True)
        axs[3].legend(loc='upper right')

        plt.suptitle(f'PINN Performance | Lap {self.lap_count} | Time: {lap_time:.3f}s', fontsize=16)
        save_name = os.path.join(self.base_path, f'PINN_performance_lap_{self.lap_count}.png')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_name)
        plt.close()
        self.get_logger().info(f"✅ Full comparison plot with CTE saved: {save_name}")
        
        # Calculate and log performance metrics
        self.compute_and_save_metrics(lap_time)

    def _lap_label(self, lap_number):
        if 10 <= lap_number % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(lap_number % 10, 'th')
        return f'{lap_number}{suffix}'

    def _lap_time_tag(self, lap_time):
        return f'{lap_time:.3f}s'.replace('.', 'p')

    def save_reference_csv(self, lap_time):
        """Save latest PINN trajectory as XY for multi-controller overlay plotting."""
        if len(self.actual_x) < 2:
            return
        df = pd.DataFrame({
            'x': np.array(self.actual_x),
            'y': np.array(self.actual_y),
            'lap_time_s': np.full(len(self.actual_x), lap_time)
        })
        df.to_csv(self.pinn_reference_path, index=False)
        self.get_logger().info(f"💾 Saved PINN XY trajectory CSV: {self.pinn_reference_path}")

        lap_path = os.path.join(self.base_path, f'pinn_reference_lap_{self.lap_count}.csv')
        df.to_csv(lap_path, index=False)
        self.get_logger().info(f"💾 Saved PINN lap-specific XY CSV: {lap_path}")

    def save_lap_path_overlay(self, lap_time):
        if len(self.actual_x) < 2 or len(self.desired_x) < 2:
            return

        actual_path = np.column_stack((np.array(self.actual_x), np.array(self.actual_y)))
        desired_path = np.vstack((self.waypoints, self.waypoints[0]))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(desired_path[:, 0], desired_path[:, 1], color='green', linewidth=2.5, label='Desired path')
        ax.plot(actual_path[:, 0], actual_path[:, 1], color='red', linestyle='--', linewidth=2, label='Actual path (PINN)')
        ax.scatter(self.waypoints[0][0], self.waypoints[0][1], color='blue', s=60, label='Start')
        ax.set_title(f'{self._lap_label(self.lap_count)} Lap Path Comparison | {lap_time:.3f}s')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        save_name = os.path.join(
            self.plot_output_dir,
            f'lap_{self._lap_label(self.lap_count)}_{self._lap_time_tag(lap_time)}_path.png'
        )
        fig.tight_layout()
        fig.savefig(save_name, dpi=150)
        plt.close(fig)
        self.get_logger().info(f"🖼️ Saved lap path overlay: {save_name}")
        
        # Also save a minimal clean version (paths only)
        self.save_lap_path_minimal(lap_time, actual_path, desired_path)

    def save_lap_path_minimal(self, lap_time, actual_path, desired_path):
        """Save a minimal, clean PNG with just the desired path (green) and actual path (blue)"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot paths - matching exact style requested
        ax.plot(desired_path[:, 0], desired_path[:, 1], color='green', linewidth=2.5, label='Desired Path')
        ax.plot(actual_path[:, 0], actual_path[:, 1], color='blue', linewidth=2, label='Actual Path')
        
        # Minimal styling
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.tick_params(labelsize=10)
        
        save_name = os.path.join(
            self.plot_output_dir,
            f'lap_{self._lap_label(self.lap_count)}_paths_clean.png'
        )
        fig.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.get_logger().info(f"📐 Saved minimal path overlay: {save_name}")


    def publish_path_visualization(self):
        """Publish waypoints as green line MarkerArray for visualization in RViz"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "waypoints_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Set lifetime (0 = infinite)
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        
        # Green color for path
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Line width
        marker.scale.x = 0.1  # Line thickness
        
        # Add all waypoints to the line
        for waypoint in self.waypoints:
            point = Point()
            point.x = float(waypoint[0])
            point.y = float(waypoint[1])
            point.z = 0.0
            marker.points.append(point)
        
        # Close the loop by adding first point at end
        point = Point()
        point.x = float(self.waypoints[0][0])
        point.y = float(self.waypoints[0][1])
        point.z = 0.0
        marker.points.append(point)

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.path_viz_pub.publish(marker_array)

    def center_line_path_cb(self, msg: Path):
        if not msg.poses:
            return

        waypoints = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses], dtype=float)
        if waypoints.size == 0:
            return

        self.waypoints = waypoints
        self.start_line = self.waypoints[0]
        self.center_line_received = True
        self.get_logger().info(f'Received {len(self.waypoints)} center line points from {self.waypoints_topic}')

    def compute_and_save_metrics(self, lap_time):
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
        
        # **Velocity profile** (if available - check from inference)
        avg_speed = np.mean(self.v_curr) if hasattr(self, 'v_curr') else 0.0
        
        # **Steering smoothness** (rate of change)
        if len(self.actual_yaw) > 1:
            yaw_rates = np.diff(np.array(self.actual_yaw))
            steering_smoothness = np.std(yaw_rates)  # Lower is smoother
        else:
            steering_smoothness = 0.0
        
        # **Sequential consistency** (lag/delay indicator)
        # Compute correlation between desired and actual to detect phase lag
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
            'controller': 'PINN',
            'lap': self.lap_count,
            'lap_time_s': lap_time,
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
        self.get_logger().info(f"📊 PINN Lap {self.lap_count} Metrics:")
        self.get_logger().info(f"   RMS Error: X={rms_x_error:.4f}m, Y={rms_y_error:.4f}m, Total={rms_positional_error:.4f}m")
        self.get_logger().info(f"   CTE: Mean={mean_cte:.4f}m, RMS={rms_cte:.4f}m, Max={max_cte:.4f}m")
        self.get_logger().info(f"   Steering Smoothness: {steering_smoothness:.4f} rad/sample")
        self.get_logger().info(f"   Trajectory Correlation: {correlation:.4f}")

    def odom_cb(self, msg):
        self.curr_x = msg.pose.position.x
        self.curr_y = msg.pose.position.y
        orient = msg.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

        now = time.time()
        if self.last_estimated_pose is not None and self.last_estimated_pose_time is not None:
            dt = now - self.last_estimated_pose_time
            if dt > 1e-3:
                dx = self.curr_x - self.last_estimated_pose[0]
                dy = self.curr_y - self.last_estimated_pose[1]
                self.v_curr = float(np.hypot(dx, dy) / dt)
        self.last_estimated_pose = (self.curr_x, self.curr_y, self.yaw)
        self.last_estimated_pose_time = now

        # Lap Detection Logic
        curr_pos = np.array([self.curr_x, self.curr_y])
        if self.prev_pos is not None:
            self.total_dist_traveled += np.linalg.norm(curr_pos - self.prev_pos)
            dist_to_start = np.linalg.norm(curr_pos - self.start_line)
            
            if dist_to_start < 1.0 and self.total_dist_traveled > 20.0:
                self.lap_count += 1
                lap_duration = time.time() - self.lap_start_time
                self.save_reference_csv(lap_duration)
                self.save_lap_path_overlay(lap_duration)
                self.save_performance_plot(lap_duration)
                
                # Reset Buffers
                self.actual_x, self.actual_y, self.actual_yaw = [], [], []
                self.desired_x, self.desired_y, self.desired_yaw = [], [], []
                self.cte_history, self.timestamps = [], []
                self.lap_start_time = time.time()
                self.total_dist_traveled = 0.0
        
        self.prev_pos = curr_pos

    def scan_cb(self, msg):
        # Look-ahead logic
        Ld = 1.5
        dists = np.linalg.norm(self.waypoints - np.array([self.curr_x, self.curr_y]), axis=1)
        closest_idx = np.argmin(dists)
        
        # Calculate CTE (Distance to closest waypoint)
        current_cte = dists[closest_idx]
        # Determine side (Left/Right) for signed CTE
        target_pt_closest = self.waypoints[closest_idx]
        side = (target_pt_closest[0] - self.curr_x) * np.sin(self.yaw) - (target_pt_closest[1] - self.curr_y) * np.cos(self.yaw)
        self.cte_history.append(current_cte if side > 0 else -current_cte)

        target_idx = closest_idx
        for i in range(closest_idx, closest_idx + 100):
            idx = i % len(self.waypoints)
            if dists[idx] >= Ld:
                target_idx = idx
                break
        
        target_pt = self.waypoints[target_idx]
        dx_g, dy_g = target_pt[0] - self.curr_x, target_pt[1] - self.curr_y
        
        # Data Recording
        self.actual_x.append(self.curr_x); self.actual_y.append(self.curr_y); self.actual_yaw.append(self.yaw)
        self.desired_x.append(target_pt[0]); self.desired_y.append(target_pt[1])
        self.desired_yaw.append(np.arctan2(dy_g, dx_g))
        self.timestamps.append(time.time())

        # Transform to Local Frame
        dx_local = dx_g * np.cos(self.yaw) + dy_g * np.sin(self.yaw)
        dy_local = -dx_g * np.sin(self.yaw) + dy_g * np.cos(self.yaw)
        err_yaw = (np.arctan2(dy_g, dx_g) - self.yaw + np.pi) % (2 * np.pi) - np.pi

        # Inference
        scan_ranges = np.nan_to_num(np.array(msg.ranges), nan=0.0, posinf=10.0)
        idx_samples = np.round(np.linspace(0, len(scan_ranges) - 1, 20)).astype(int)
        lidar_20 = scan_ranges[idx_samples]

        features = np.array([self.v_curr, dx_local, dy_local, err_yaw] + lidar_20.tolist()).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy()[0]
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(prediction[0])
        drive_msg.drive.speed = float(prediction[1])
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args); node = PINNDriveNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()