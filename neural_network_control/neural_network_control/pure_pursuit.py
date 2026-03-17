import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion

class CurriculumPerformancePlotter(Node):
    def __init__(self):
        super().__init__('curriculum_performance_plotter')
        
        # --- 1. CONFIG & PATHS ---
        # Residing in Egypt for your uni track development
        self.base_path = os.path.expanduser('~/sim_ws/src/neural_network_control/neural_network_control/')
        self.waypoints_path = os.path.join(self.base_path, 'waypoints.csv')
        self.pp_reference_path = os.path.join(self.base_path, 'pp_reference_lap.csv')
        
        self.wheelbase = 0.33
        self.MIN_SPEED = 1.5       
        self.TARGET_SPEED = 9.0    
        
        # --- 2. LAP TIMING & BUFFERS ---
        self.lap_count = 0
        self.lap_start_time = time.time()
        self.total_dist_traveled = 0.0
        self.prev_pos = None
        
        # Initializing all buffers for plotting
        self.plot_times = []
        self.act_x, self.des_x = [], []
        self.act_y, self.des_y = [], []
        self.act_yaw, self.des_yaw = [], []
        self.cte_hist = []

        # ROS 2 Humble setup
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.load_waypoints()
        self.start_line = self.waypoints[0] 
        self.curr_x, self.curr_y, self.yaw, self.v_curr = 0.0, 0.0, 0.0, 0.0
        self.get_logger().info(f"🏁 PP PLOTTER START: Target Speed {self.TARGET_SPEED}m/s")

    def load_waypoints(self):
        self.waypoints = pd.read_csv(self.waypoints_path)[['x', 'y']].values

    def save_reference_csv(self, progress_axis):
        """Saves PP data for the PINN node comparison."""
        df = pd.DataFrame({
            'progress': progress_axis,
            'time': np.array(self.plot_times),
            'pp_x': np.array(self.act_x),
            'pp_y': np.array(self.act_y),
            'pp_yaw': np.array(self.act_yaw),
            'pp_cte': np.array(self.cte_hist)
        })
        df.to_csv(self.pp_reference_path, index=False)
        self.get_logger().info(f"💾 Saved PP Reference CSV.")

    def compute_and_save_metrics(self, lap_duration):
        """Calculate quantitative performance metrics for comparison"""
        if len(self.act_x) < 2:
            return
        
        # Convert to numpy arrays
        actual_x = np.array(self.act_x)
        actual_y = np.array(self.act_y)
        desired_x = np.array(self.des_x)
        desired_y = np.array(self.des_y)
        cte = np.array(self.cte_hist)
        
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
        if len(self.act_yaw) > 1:
            yaw_rates = np.diff(np.array(self.act_yaw))
            steering_smoothness = np.std(yaw_rates)  # Lower is smoother
        else:
            steering_smoothness = 0.0
        
        # **Sequential consistency** (correlation)
        if len(self.des_yaw) > 10:
            desired_yaw = np.array(self.des_yaw[:min_len])
            actual_yaw = np.array(self.act_yaw[:min_len])
            correlation = np.corrcoef(desired_yaw, actual_yaw)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Save metrics to CSV
        metrics_dict = {
            'controller': 'Pure_Pursuit',
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
        self.get_logger().info(f"📊 Pure Pursuit Lap {self.lap_count} Metrics:")
        self.get_logger().info(f"   RMS Error: X={rms_x_error:.4f}m, Y={rms_y_error:.4f}m, Total={rms_positional_error:.4f}m")
        self.get_logger().info(f"   CTE: Mean={mean_cte:.4f}m, RMS={rms_cte:.4f}m, Max={max_cte:.4f}m")
        self.get_logger().info(f"   Steering Smoothness: {steering_smoothness:.4f} rad/sample")
        self.get_logger().info(f"   Trajectory Correlation: {correlation:.4f}")

    def save_lap_plots(self, lap_duration):
        """Generates PNGs by Time and Progress using explicit NumPy conversion."""
        times_axis = np.array(self.plot_times)
        progress_axis = np.linspace(0, 100, len(self.act_x))
        
        # Convert lists to numpy arrays to avoid 'Multi-dimensional indexing' errors
        ax_np, dx_np = np.array(self.act_x), np.array(self.des_x)
        ay_np, dy_np = np.array(self.act_y), np.array(self.des_y)
        ayaw_np, dyaw_np = np.array(self.act_yaw), np.array(self.des_yaw)
        cte_np = np.array(self.cte_hist)

        plot_configs = [
            (times_axis, "Time (s)", "time_tracking.png"),
            (progress_axis, "Lap Progress (%)", "progress_tracking.png")
        ]

        for x_data, x_label, suffix in plot_configs:
            fig, axs = plt.subplots(4, 1, figsize=(10, 14))
            fig.suptitle(f'Pure Pursuit vs Desired Path ({x_label}) | Lap: {self.lap_count}', fontsize=16)

            axs[0].plot(x_data, dx_np, 'g', label='Desired Path', alpha=0.3, linewidth=2)
            axs[0].plot(x_data, ax_np, 'b:', label='Pure Pursuit')
            axs[0].set_ylabel('X (m)'); axs[0].legend(); axs[0].grid(True)

            axs[1].plot(x_data, dy_np, 'g', alpha=0.3, linewidth=2)
            axs[1].plot(x_data, ay_np, 'b:')
            axs[1].set_ylabel('Y (m)'); axs[1].grid(True)

            axs[2].plot(x_data, dyaw_np, 'g', alpha=0.3, linewidth=2)
            axs[2].plot(x_data, ayaw_np, 'b:')
            axs[2].set_ylabel('Yaw (rad)'); axs[2].grid(True)

            axs[3].plot(x_data, cte_np, 'r', label='Cross Track Error')
            axs[3].set_ylabel('CTE (m)'); axs[3].set_xlabel(x_label); axs[3].grid(True)

            save_path = os.path.join(self.base_path, f'lap_{self.lap_count}_{suffix}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_path)
            plt.close(fig)
        
        return progress_axis

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
            
            # Detect lap completion
            if dist_to_start < 1.2 and self.total_dist_traveled > 15.0:
                self.lap_count += 1
                lap_duration = time.time() - self.lap_start_time
                
                prog_ax = self.save_lap_plots(lap_duration)
                self.save_reference_csv(prog_ax)
                self.compute_and_save_metrics(lap_duration)
                
                # Fixed unpacking: 4 variables assigned to 4 lists
                self.plot_times, self.act_x, self.des_x = [], [], []
                self.act_y, self.des_y, self.act_yaw, self.des_yaw = [], [], [], []
                self.cte_hist = []
                self.lap_start_time = time.time()
                self.total_dist_traveled = 0.0
        
        self.prev_pos = curr_pos

    def scan_callback(self, scan_msg):
        # Pure Pursuit Look-ahead logic
        Ld = np.clip(self.v_curr * 0.15 + 0.8, 1.0, 3.5)
        dists = np.linalg.norm(self.waypoints - np.array([self.curr_x, self.curr_y]), axis=1)
        closest_idx = np.argmin(dists)
        
        target_idx = closest_idx
        for i in range(closest_idx, closest_idx + 100):
            idx = i % len(self.waypoints)
            if dists[idx] >= Ld:
                target_idx = idx
                break
        
        target_pt = self.waypoints[target_idx]
        dx_g, dy_g = target_pt[0] - self.curr_x, target_pt[1] - self.curr_y
        dx_local = dx_g * np.cos(self.yaw) + dy_g * np.sin(self.yaw)
        dy_local = -dx_g * np.sin(self.yaw) + dy_g * np.cos(self.yaw)
        
        steer = np.clip(np.arctan((2 * self.wheelbase * dy_local) / (Ld**2)), -0.41, 0.41)
        
        # Adaptive speed logic
        future_idx = (target_idx + 12) % len(self.waypoints)
        f_pt = self.waypoints[future_idx]
        fl_y = -(f_pt[0]-self.curr_x) * np.sin(self.yaw) + (f_pt[1]-self.curr_y) * np.cos(self.yaw)
        f_steer = abs(np.arctan((2 * self.wheelbase * fl_y) / (Ld**2)))
        curve_severity = max(abs(steer), f_steer) / 0.41
        speed_label = np.clip(self.TARGET_SPEED - (4.5 * curve_severity) - np.clip((abs(dy_local) - 0.2) * 5.0, 0.0, 6.5), self.MIN_SPEED, self.TARGET_SPEED)

        # Recording data for comparison
        path_yaw = np.arctan2(dy_g, dx_g)
        self.plot_times.append(time.time() - self.lap_start_time)
        self.act_x.append(self.curr_x); self.des_x.append(target_pt[0])
        self.act_y.append(self.curr_y); self.des_y.append(target_pt[1])
        self.act_yaw.append(self.yaw); self.des_yaw.append(path_yaw)
        self.cte_hist.append(np.min(dists))

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed, drive_msg.drive.steering_angle = float(speed_label), float(steer)
        self.drive_pub.publish(drive_msg)

def main():
    rclpy.init(); node = CurriculumPerformancePlotter()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()