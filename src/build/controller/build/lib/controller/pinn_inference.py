import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
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
            nn.Linear(64, 2) # [dx, dy]
        )
    def forward(self, x):
        return self.net(x)

class PINNFinalController(Node):
    def __init__(self):
        super().__init__('pinn_final_controller')
        
        # Paths
        base_path = os.path.expanduser('~/sim_ws/src/controller/controller/')
        self.waypoints_path = os.path.join(base_path, 'waypoints.csv')
        self.model_path = os.path.join(base_path, 'pinn_model.pth')
        self.scaler_path = os.path.join(base_path, 'scaler.pkl')
        
        # Vehicle & Racing Params
        self.wheelbase = 0.33
        self.MAX_STEER = 0.41
        self.TARGET_RACE_SPEED = 11.0  # Your target top speed
        self.MIN_SPEED_FLOOR = 3.5    # Minimum speed for sharp hairpins
        
        # Load AI Components
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.model = F1TENTH_PINN(input_dim=22)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.load_waypoints()
        
        # ROS Comms
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.viz_pub = self.create_publisher(Marker, '/pinn_prediction_viz', 10)

        self.v_curr, self.yaw, self.curr_x, self.curr_y = 0.0, 0.0, 0.0, 0.0
        self.get_logger().info(f"🏁 Final PINN Controller Loaded. Target Speed: {self.TARGET_RACE_SPEED}m/s")

    def load_waypoints(self):
        df = pd.read_csv(self.waypoints_path)
        self.waypoints = df[['x', 'y']].values

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.v_curr = msg.twist.twist.linear.x
        _, _, self.yaw = euler_from_quaternion([
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

    def scan_callback(self, scan_msg):
        # 1. AI PREDICTION (Where will the car be?)
        scan_ranges = np.array(scan_msg.ranges)
        scan_ranges = np.nan_to_num(scan_ranges, nan=0.0, posinf=10.0, neginf=0.0)
        idx = np.round(np.linspace(0, len(scan_ranges) - 1, 20)).astype(int)
        lidar_samples = scan_ranges[idx]
        
        ai_input = np.hstack(([self.v_curr, self.yaw], lidar_samples)).reshape(1, -1)
        input_scaled = self.scaler.transform(ai_input)
        
        with torch.no_grad():
            prediction = self.model(torch.tensor(input_scaled, dtype=torch.float32)).numpy()[0]
        
        # Transform local dx, dy to Global Coordinates
        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        pred_x = self.curr_x + (prediction[0] * cos_y - prediction[1] * sin_y)
        pred_y = self.curr_y + (prediction[0] * sin_y + prediction[1] * cos_y)
        self.publish_viz(pred_x, pred_y)

        # 2. PURE PURSUIT (Lateral Control)
        Ld = np.clip(self.v_curr * 0.15 + 0.8, 0.8, 3.5)
        dists = np.linalg.norm(self.waypoints - np.array([pred_x, pred_y]), axis=1)
        closest_idx = np.argmin(dists)
        
        target_idx = closest_idx
        for i in range(closest_idx, len(self.waypoints)):
            if dists[i] >= Ld:
                target_idx = i
                break
        
        target_pt = self.waypoints[target_idx]
        tx, ty = target_pt[0] - pred_x, target_pt[1] - pred_y
        local_y = tx * np.sin(-self.yaw) + ty * np.cos(-self.yaw)
        steering_angle = np.arctan((2 * self.wheelbase * local_y) / (Ld**2))
        steering_angle = np.clip(steering_angle, -self.MAX_STEER, self.MAX_STEER)

        # 3. PREDICTIVE BRAKING (Longitudinal Control)
        # Check waypoint curvature 15 points ahead
        look_ahead_idx = min(target_idx + 15, len(self.waypoints) - 1)
        look_ahead_pt = self.waypoints[look_ahead_idx]
        
        lx, ly = look_ahead_pt[0] - pred_x, look_ahead_pt[1] - pred_y
        future_local_y = lx * np.sin(-self.yaw) + ly * np.cos(-self.yaw)
        future_steer_needed = abs(np.arctan((2 * self.wheelbase * future_local_y) / (Ld**2)))
        
        # Blend current steer and future steer for braking severity
        severity = max(abs(steering_angle), future_steer_needed) / self.MAX_STEER
        
        # Final Velocity Logic: Scale between Top Speed and Floor Speed
        target_speed = self.TARGET_RACE_SPEED - (self.TARGET_RACE_SPEED - self.MIN_SPEED_FLOOR) * severity
        target_speed = max(target_speed, self.MIN_SPEED_FLOOR)

        # 4. PUBLISH
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(target_speed)
        drive_msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(drive_msg)

    def publish_viz(self, px, py):
        m = Marker()
        m.header.frame_id = "map"
        m.id = 0
        m.type = Marker.SPHERE
        m.pose.position.x, m.pose.position.y = float(px), float(py)
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.a, m.color.r, m.color.g, m.color.b = 1.0, 0.0, 1.0, 1.0 # Cyan
        self.viz_pub.publish(m)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(PINNFinalController())
    rclpy.shutdown()

if __name__ == '__main__':
    main()