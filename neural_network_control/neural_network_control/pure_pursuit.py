import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import torch
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion

class CurriculumDataCollector(Node):
    def __init__(self):
        super().__init__('curriculum_data_collector')
        
        # --- 1. CONFIG & PATHS ---
        base_path = os.path.expanduser('~/sim_ws/src/neural_network_control/neural_network_control/')
        self.waypoints_path = os.path.join(base_path, 'waypoints.csv')
        self.output_csv_path = os.path.join(base_path, 'curriculum_training_data.csv')
        
        self.wheelbase = 0.33
        self.MIN_SPEED = 3.0
        self.MAX_SPEED_LIMIT = 9.0  # Cap it at 9 to avoid constant crashing
        self.current_max_speed = 3.0
        
        # --- 2. LAP TRACKING ---
        self.lap_count = 0
        self.last_pos = None
        self.total_dist = 0.0
        self.finish_threshold = 1.5
        self.cooldown_dist = 25.0 # Prevent double counting at start line
        
        # --- 3. DATA STORAGE ---
        self.data_buffer = []
        
        # Subscribers & Publishers
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.load_waypoints()
        self.curr_x, self.curr_y, self.yaw, self.v_curr = 0.0, 0.0, 0.0, 0.0
        self.get_logger().info(f"🏁 CURRICULUM START: Initial Speed {self.current_max_speed}m/s")

    def load_waypoints(self):
        self.waypoints = pd.read_csv(self.waypoints_path)[['x', 'y']].values

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.v_curr = msg.twist.twist.linear.x
        orient = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

        # LAP COUNTING LOGIC
        curr_pos = np.array([self.curr_x, self.curr_y])
        if self.last_pos is not None:
            self.total_dist += np.linalg.norm(curr_pos - self.last_pos)
            dist_to_start = np.linalg.norm(curr_pos - self.waypoints[0])
            
            if dist_to_start < self.finish_threshold and self.total_dist > self.cooldown_dist:
                self.lap_count += 1
                self.total_dist = 0.0
                self.get_logger().info(f"✅ LAP {self.lap_count} COMPLETE")
                
                # Update speed every 5 laps
                if self.lap_count % 5 == 0 and self.current_max_speed < self.MAX_SPEED_LIMIT:
                    self.current_max_speed += 1.0
                    self.get_logger().info(f"🚀 SPEED UP! New Max Speed: {self.current_max_speed}m/s")
        
        self.last_pos = curr_pos

    def scan_callback(self, scan_msg):
        # 1. PURE PURSUIT MATH
        Ld = np.clip(self.v_curr * 0.15 + 0.8, 1.0, 3.5)
        dists = np.linalg.norm(self.waypoints - np.array([self.curr_x, self.curr_y]), axis=1)
        closest_idx = np.argmin(dists)
        
        target_idx = closest_idx
        for i in range(closest_idx, closest_idx + 100):
            idx = i % len(self.waypoints)
            if dists[idx] >= Ld:
                target_idx = idx
                break
        
        # Local Transformation
        target_pt = self.waypoints[target_idx]
        dx_g, dy_g = target_pt[0] - self.curr_x, target_pt[1] - self.curr_y
        dx_local = dx_g * np.cos(self.yaw) + dy_g * np.sin(self.yaw)
        dy_local = -dx_g * np.sin(self.yaw) + dy_g * np.cos(self.yaw)
        
        steer = np.arctan((2 * self.wheelbase * dy_local) / (Ld**2))
        steer = np.clip(steer, -0.41, 0.41)

        # 2. PREDICTIVE BRAKING
        future_idx = (target_idx + 12) % len(self.waypoints)
        f_pt = self.waypoints[future_idx]
        fl_y = -(f_pt[0]-self.curr_x) * np.sin(self.yaw) + (f_pt[1]-self.curr_y) * np.cos(self.yaw)
        f_steer = abs(np.arctan((2 * self.wheelbase * fl_y) / (Ld**2)))
        
        severity = max(abs(steer), f_steer) / 0.41
        speed_label = max(self.current_max_speed - (self.current_max_speed - self.MIN_SPEED) * severity, self.MIN_SPEED)

        # 3. DATA LOGGING
        scan_ranges = np.nan_to_num(np.array(scan_msg.ranges), nan=0.0, posinf=10.0)
        idx_samples = np.round(np.linspace(0, len(scan_ranges) - 1, 20)).astype(int)
        lidar_samples = scan_ranges[idx_samples]
        
        path_yaw = np.arctan2(dy_g, dx_g)
        err_yaw = (path_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi

        record = {
            'v_curr': self.v_curr,
            'err_x': dx_local, 'err_y': dy_local, 'err_yaw': err_yaw,
            'steer_label': steer, 'speed_label': speed_label
        }
        for i, val in enumerate(lidar_samples): record[f'lidar_{i}'] = val
        self.data_buffer.append(record)

        # 4. PUBLISH DRIVE
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(speed_label)
        drive_msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(drive_msg)

        if len(self.data_buffer) >= 500: self.save_data()

    def save_data(self):
        df = pd.DataFrame(self.data_buffer)
        file_exists = os.path.isfile(self.output_csv_path)
        df.to_csv(self.output_csv_path, mode='a', index=False, header=not file_exists)
        self.data_buffer = []

def main():
    rclpy.init()
    node = CurriculumDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()