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
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion

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

class PINNHybridController(Node):
    def __init__(self):
        super().__init__('pinn_hybrid_controller')
        
        # --- CONFIGURATION ---
        base_path = os.path.expanduser('~/sim_ws/src/controller/controller/')
        self.waypoints_path = os.path.join(base_path, 'waypoints.csv')
        self.model_path = os.path.join(base_path, 'pinn_model.pth')
        self.scaler_path = os.path.join(base_path, 'scaler.pkl')
        
        self.wheelbase = 0.33
        self.lookahead_gain = 0.18   
        self.lookahead_min = 0.8     
        self.lookahead_max = 3.5     
        self.smoothing = 0.4          
        self.MAX_STEER = 0.41         
        self.MAX_CORRECTION = 0.06    
        self.MIN_SPEED = 1.5          
        self.MAX_SPEED = 8.0          

        # --- UPDATED ACCURATE LAP TIMER SETTINGS ---
        self.lap_start_time = None
        self.lap_count = 0
        self.finish_line_threshold = 1.2   # Increased for high speed (8m/s)
        self.cooldown_dist = 10.0          # Dist to travel before re-arming
        self.total_dist_traveled = 0.0
        self.last_pos = None
        self.lap_triggered = False         # Prevents double-counting
        # -------------------------------------------

        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.model = F1TENTH_PINN(input_dim=22) 
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.load_waypoints()
        
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/viz_path', 10)
        self.target_pub = self.create_publisher(Marker, '/viz_target', 10)

        self.v_curr, self.yaw, self.curr_x, self.curr_y = 0.0, 0.0, 0.0, 0.0
        self.prev_steering = 0.0
        self.create_timer(2.0, self.publish_static_path)
        
        self.get_logger().info("🚀 Hybrid PINN Active with Accurate Lap Timer!")

    def load_waypoints(self):
        df = pd.read_csv(self.waypoints_path)
        self.waypoints = df[['x', 'y']].values

    def update_lap_timer(self, x, y):
        curr_pos = np.array([x, y])
        if self.last_pos is None:
            self.last_pos = curr_pos
            self.lap_start_time = self.get_clock().now()
            return

        # Cumulative distance to ensure we've actually driven the track
        step_dist = np.linalg.norm(curr_pos - self.last_pos)
        self.total_dist_traveled += step_dist
        self.last_pos = curr_pos

        # Check distance to Finish Line (Waypoint 0)
        dist_to_finish = np.linalg.norm(curr_pos - self.waypoints[0])

        if dist_to_finish < self.finish_line_threshold:
            # Only trigger if we aren't already in the "triggered" state
            if not self.lap_triggered and self.total_dist_traveled > self.cooldown_dist:
                now = self.get_clock().now()
                if self.lap_start_time is not None:
                    lap_time_sec = (now - self.lap_start_time).nanoseconds / 1e9
                    self.lap_count += 1
                    self.get_logger().info(f"🏁 [HYBRID] LAP {self.lap_count} | TIME: {lap_time_sec:.3f}s")
                
                self.lap_start_time = now
                self.total_dist_traveled = 0.0
                self.lap_triggered = True  # Lock the trigger
        else:
            # Reset the trigger once we have moved outside the finish zone
            if self.lap_triggered:
                self.lap_triggered = False

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.v_curr = msg.twist.twist.linear.x
        
        self.update_lap_timer(self.curr_x, self.curr_y)
        
        orient = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

    def scan_callback(self, scan_msg):
        # 1. ADAPTIVE LOOKAHEAD
        Ld = np.clip(self.v_curr * self.lookahead_gain + self.lookahead_min, self.lookahead_min, self.lookahead_max)
        distances = np.linalg.norm(self.waypoints - np.array([self.curr_x, self.curr_y]), axis=1)
        closest_idx = np.argmin(distances)
        
        target_idx = closest_idx
        for i in range(closest_idx, len(self.waypoints)):
            if distances[i] >= Ld:
                target_idx = i
                break
        target_point = self.waypoints[target_idx]
        self.publish_target_point(target_point[0], target_point[1])

        dx, dy = target_point[0] - self.curr_x, target_point[1] - self.curr_y
        local_y = dx * np.sin(-self.yaw) + dy * np.cos(-self.yaw)
        pp_steering = np.arctan((2 * self.wheelbase * local_y) / (Ld**2))

        # 2. AI PREDICTION
        scan_ranges = np.array(scan_msg.ranges)
        scan_ranges = np.nan_to_num(scan_ranges, nan=0.0, posinf=10.0, neginf=0.0)
        idx = np.round(np.linspace(0, len(scan_ranges) - 1, 20)).astype(int)
        lidar_samples = scan_ranges[idx]
        ai_input = np.hstack(([self.v_curr, self.yaw], lidar_samples)).reshape(1, -1)
        
        input_scaled = self.scaler.transform(ai_input)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            prediction = self.model(input_tensor).numpy()[0]
        
        # 3. FILTERS & COMMAND
        steering_corr = np.clip(prediction[0], -self.MAX_CORRECTION, self.MAX_CORRECTION)
        v_opt = np.clip(prediction[1], self.MIN_SPEED, self.MAX_SPEED)
        
        if abs(pp_steering) > 0.15:
            v_opt *= 0.85 

        raw_final_steering = np.clip(pp_steering + steering_corr, -self.MAX_STEER, self.MAX_STEER)
        final_steering = (self.smoothing * self.prev_steering) + ((1.0 - self.smoothing) * raw_final_steering)
        self.prev_steering = final_steering

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(final_steering)
        drive_msg.drive.speed = float(v_opt)
        self.drive_pub.publish(drive_msg)

    def publish_static_path(self):
        marker_array = MarkerArray()
        for i in range(0, len(self.waypoints), 5):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y = float(self.waypoints[i][0]), float(self.waypoints[i][1])
            marker.scale.x, marker.scale.y, marker.scale.z = 0.15, 0.15, 0.15
            marker.color.a, marker.color.g = 1.0, 1.0
            marker_array.markers.append(marker)
        self.viz_pub.publish(marker_array)

    def publish_target_point(self, tx, ty):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y = float(tx), float(ty)
        marker.scale.x, marker.scale.y, marker.scale.z = 0.35, 0.35, 0.35
        marker.color.a, marker.color.r = 1.0, 1.0
        self.target_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = PINNHybridController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()