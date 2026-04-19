import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from tf_transformations import euler_from_quaternion

# --- THE MODEL ARCHITECTURE ---
class F1TENTH_PINN(nn.Module):
    def __init__(self, input_dim):
        super(F1TENTH_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

class PINNFinalController(Node):
    def __init__(self):
        super().__init__('pinn_final_controller')
        
        # 1. PATHS & LOGGING CONFIG
        base_path = os.path.expanduser('~/sim_ws/src/controller/controller/')
        self.waypoints_path = os.path.join(base_path, 'waypoints.csv')
        self.model_path = os.path.join(base_path, 'pinn_model.pth')
        self.scaler_path = os.path.join(base_path, 'scaler.pkl')
        self.output_csv_path = os.path.join(base_path, 'pinn_training_data_new.csv')
        self.plot_output_dir = os.path.join(base_path, 'lap_plots')
        os.makedirs(self.plot_output_dir, exist_ok=True)
        
        # 2. VEHICLE & RACING PARAMS
        self.wheelbase, self.MAX_STEER = 0.33, 0.41
        self.TARGET_RACE_SPEED, self.MIN_SPEED_FLOOR = 11.0, 3.5    
        
        # 3. LOGGER STATE
        self.data_buffer = []
        self.total_dist_traveled = 0.0
        self.last_pos = None
        self.finish_threshold, self.cooldown_dist = 1.5, 25.0
        self.current_lap_x = []
        self.current_lap_y = []
        
        # Load AI Components
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.model = F1TENTH_PINN(input_dim=22)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.load_waypoints()
        
        # ROS Comms
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/ego_racecar/scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/ego_racecar/drive', 10)
        self.viz_pub = self.create_publisher(Marker, '/pinn_prediction_viz', 10)

        self.v_curr, self.yaw, self.curr_x, self.curr_y = 0.0, 0.0, 0.0, 0.0
        self.get_logger().info(f"✅ DRIVE & ERROR-LOGGER READY. Saving to: {self.output_csv_path}")

    def load_waypoints(self):
        self.waypoints = pd.read_csv(self.waypoints_path)[['x', 'y']].values

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.v_curr = msg.twist.twist.linear.x
        _, _, self.yaw = euler_from_quaternion([
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

        # --- LAP LOGIC ---
        curr_pos = np.array([self.curr_x, self.curr_y])
        self.current_lap_x.append(self.curr_x)
        self.current_lap_y.append(self.curr_y)
        if self.last_pos is not None:
            self.total_dist_traveled += np.linalg.norm(curr_pos - self.last_pos)
            dist_to_start = np.linalg.norm(curr_pos - self.waypoints[0])
            
            if dist_to_start < self.finish_threshold and self.total_dist_traveled > self.cooldown_dist:
                self.get_logger().info("🏁 LAP COMPLETE! Writing data to CSV...")
                self.save_lap_path_plot(self.lap_count + 1)
                self.save_to_csv()
                self.total_dist_traveled = 0.0
                self.current_lap_x = [self.curr_x]
                self.current_lap_y = [self.curr_y]
        self.last_pos = curr_pos

    def scan_callback(self, scan_msg):
        # --- 1. AI PREDICTION ---
        scan_ranges = np.array(scan_msg.ranges)
        scan_ranges = np.nan_to_num(scan_ranges, nan=0.0, posinf=10.0, neginf=0.0)
        idx = np.round(np.linspace(0, len(scan_ranges) - 1, 20)).astype(int)
        lidar_samples = scan_ranges[idx]
        
        ai_input = np.hstack(([self.v_curr, self.yaw], lidar_samples)).reshape(1, -1)
        input_scaled = self.scaler.transform(ai_input)
        
        with torch.no_grad():
            prediction = self.model(torch.tensor(input_scaled, dtype=torch.float32)).numpy()[0]
        
        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        pred_x = self.curr_x + (prediction[0] * cos_y - prediction[1] * sin_y)
        pred_y = self.curr_y + (prediction[0] * sin_y + prediction[1] * cos_y)
        self.publish_viz(pred_x, pred_y)

        # --- 2. ERROR CALCULATIONS ---
        # Find closest waypoint to the CURRENT car position to calculate errors
        dists_to_track = np.linalg.norm(self.waypoints - np.array([self.curr_x, self.curr_y]), axis=1)
        closest_idx = np.argmin(dists_to_track)
        
        # Positional Error (Target - Current)
        err_x = self.waypoints[closest_idx][0] - self.curr_x
        err_y = self.waypoints[closest_idx][1] - self.curr_y
        
        # Heading Error (Yaw Error)
        next_idx = (closest_idx + 1) % len(self.waypoints)
        path_yaw = np.arctan2(self.waypoints[next_idx][1] - self.waypoints[closest_idx][1], 
                              self.waypoints[next_idx][0] - self.waypoints[closest_idx][0])
        err_yaw = path_yaw - self.yaw
        err_yaw = (err_yaw + np.pi) % (2 * np.pi) - np.pi # Normalize

        # --- 3. PURE PURSUIT (Lateral) ---
        Ld = np.clip(self.v_curr * 0.15 + 0.8, 0.8, 3.5)
        dists_from_pred = np.linalg.norm(self.waypoints - np.array([pred_x, pred_y]), axis=1)
        target_idx = np.argmin(dists_from_pred)
        for i in range(target_idx, target_idx + 100):
            if dists_from_pred[i % len(self.waypoints)] >= Ld:
                target_idx = i % len(self.waypoints)
                break
        
        target_pt = self.waypoints[target_idx]
        tx, ty = target_pt[0] - pred_x, target_pt[1] - pred_y
        local_y = tx * np.sin(-self.yaw) + ty * np.cos(-self.yaw)
        steering_angle = np.arctan((2 * self.wheelbase * local_y) / (Ld**2))
        steering_angle = np.clip(steering_angle, -self.MAX_STEER, self.MAX_STEER)

        # --- 4. PREDICTIVE BRAKING ---
        look_ahead_idx = min(target_idx + 15, len(self.waypoints) - 1)
        future_pt = self.waypoints[look_ahead_idx]
        flx, fly = future_pt[0] - pred_x, future_pt[1] - pred_y
        future_local_y = flx * np.sin(-self.yaw) + fly * np.cos(-self.yaw)
        future_steer_needed = abs(np.arctan((2 * self.wheelbase * future_local_y) / (Ld**2)))
        
        severity = max(abs(steering_angle), future_steer_needed) / self.MAX_STEER
        target_speed = max(self.TARGET_RACE_SPEED - (self.TARGET_RACE_SPEED - self.MIN_SPEED_FLOOR) * severity, self.MIN_SPEED_FLOOR)

        # --- 5. PUBLISH & RECORD ---
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed, drive_msg.drive.steering_angle = float(target_speed), float(steering_angle)
        self.drive_pub.publish(drive_msg)

        # Log errors + labels for training
        record = {
            'v_curr': self.v_curr,
            'err_x': err_x,
            'err_y': err_y,
            'err_yaw': err_yaw,
            'steer_label': steering_angle,
            'speed_label': target_speed
        }
        for i, val in enumerate(lidar_samples):
            record[f'lidar_{i}'] = val
        self.data_buffer.append(record)

    def save_to_csv(self):
        if not self.data_buffer: return
        try:
            df = pd.DataFrame(self.data_buffer)
            file_exists = os.path.isfile(self.output_csv_path)
            df.to_csv(self.output_csv_path, mode='a', index=False, header=not file_exists)
            self.get_logger().info(f"💾 SAVED {len(self.data_buffer)} samples.")
            self.data_buffer = [] 
        except Exception as e:
            self.get_logger().error(f"❌ FAILED TO SAVE: {e}")

    def save_lap_path_plot(self, lap_number):
        if len(self.current_lap_x) < 2:
            return

        actual_path = np.column_stack((np.array(self.current_lap_x), np.array(self.current_lap_y)))
        desired_path = np.vstack((self.waypoints, self.waypoints[0]))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(desired_path[:, 0], desired_path[:, 1], color='green', linewidth=2.5, label='Desired path')
        ax.plot(actual_path[:, 0], actual_path[:, 1], color='red', linestyle='--', linewidth=2, label='Actual path')
        ax.scatter(self.waypoints[0][0], self.waypoints[0][1], color='blue', s=60, label='Start')
        ax.set_title(f'Lap {lap_number} Path Comparison')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        save_path = os.path.join(self.plot_output_dir, f'lap_{lap_number:03d}_path.png')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        self.get_logger().info(f"🖼️ Saved lap path plot: {save_path}")

    def publish_viz(self, px, py):
        m = Marker()
        m.header.frame_id = "map"
        m.id = 0
        m.type = Marker.SPHERE
        m.pose.position.x, m.pose.position.y = float(px), float(py)
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.a, m.color.r, m.color.g, m.color.b = 1.0, 0.0, 1.0, 1.0
        self.viz_pub.publish(m)

def main(args=None):
    rclpy.init(args=args)
    node = PINNFinalController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.save_to_csv() 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()