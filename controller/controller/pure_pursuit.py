import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion

class AutomatedCurriculumLogger(Node):
    def __init__(self):
        super().__init__('automated_curriculum_logger')
        
        # --- CONFIGURATION ---
        self.waypoints_path = os.path.expanduser('~/sim_ws/src/controller/controller/waypoints.csv')
        self.output_csv_path = os.path.expanduser('~/sim_ws/src/controller/controller/training_data.csv')
        self.wheelbase = 0.33
        
        # --- CURRICULUM SETTINGS ---
        self.speeds = [5.0, 7.0, 8.5] 
        self.speed_idx = 0
        self.max_speed = self.speeds[self.speed_idx]
        self.laps_per_speed = 5
        self.current_lap_in_phase = 0

        # --- LAP TIMER LOGIC ---
        self.lap_count = 0
        self.last_pos = None
        self.total_dist_traveled = 0.0
        self.cooldown_dist = 15.0
        self.finish_threshold = 1.2
        self.lap_triggered = False

        # Subscribers
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/ego_racecar/odom')
        self.scan_sub = message_filters.Subscriber(self, LaserScan, '/scan')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.scan_sub], 10, 0.05)
        self.ts.registerCallback(self.sync_callback)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.data_buffer = []
        self.load_waypoints()
        
        self.get_logger().info(f"🚀 LOGGER READY: Phase {self.speed_idx + 1}/3 | Max Speed: {self.max_speed} m/s")

    def load_waypoints(self):
        if not os.path.exists(self.waypoints_path):
            self.get_logger().error(f"❌ Waypoints not found at {self.waypoints_path}")
            return
        df = pd.read_csv(self.waypoints_path)
        self.waypoints = df[['x', 'y']].values

    def update_lap_counter(self, x, y):
        curr_pos = np.array([x, y])
        if self.last_pos is None:
            self.last_pos = curr_pos
            return

        self.total_dist_traveled += np.linalg.norm(curr_pos - self.last_pos)
        self.last_pos = curr_pos
        dist_to_start = np.linalg.norm(curr_pos - self.waypoints[0])

        if dist_to_start < self.finish_threshold:
            if not self.lap_triggered and self.total_dist_traveled > self.cooldown_dist:
                self.lap_count += 1
                self.current_lap_in_phase += 1
                self.total_dist_traveled = 0.0
                self.lap_triggered = True
                
                self.get_logger().info(f"🏁 LAP {self.lap_count} DONE ({self.current_lap_in_phase}/{self.laps_per_speed})")

                if self.current_lap_in_phase >= self.laps_per_speed:
                    self.speed_idx += 1
                    if self.speed_idx < len(self.speeds):
                        self.max_speed = self.speeds[self.speed_idx]
                        self.current_lap_in_phase = 0
                        self.get_logger().info(f"🔥 SPEED INCREASED TO: {self.max_speed} m/s")
                    else:
                        self.get_logger().info("✅ CURRICULUM FINISHED!")
                        self.max_speed = 0.0 
        else:
            if self.lap_triggered:
                self.lap_triggered = False

    def sync_callback(self, odom_msg, scan_msg):
        curr_x = odom_msg.pose.pose.position.x
        curr_y = odom_msg.pose.pose.position.y
        curr_v = odom_msg.twist.twist.linear.x
        
        self.update_lap_counter(curr_x, curr_y)

        # 1. Orientation
        orient = odom_msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])
        
        # 2. Pure Pursuit Logic
        Ld = np.clip(curr_v * 0.12 + 0.8, 0.8, 3.5)
        distances = np.linalg.norm(self.waypoints - np.array([curr_x, curr_y]), axis=1)
        closest_idx = np.argmin(distances)
        
        target_idx = closest_idx
        for i in range(closest_idx, len(self.waypoints)):
            if distances[i] >= Ld:
                target_idx = i
                break
        
        target_point = self.waypoints[target_idx]
        dx, dy = target_point[0] - curr_x, target_point[1] - curr_y
        local_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)
        steering_angle = np.arctan((2 * self.wheelbase * local_y) / (Ld**2))
        steering_angle = np.clip(steering_angle, -0.41, 0.41)

        # 3. Predictive Braking
        future_idx = min(target_idx + 12, len(self.waypoints) - 1)
        future_pt = self.waypoints[future_idx]
        f_dx, f_dy = future_pt[0] - curr_x, future_pt[1] - curr_y
        f_local_y = f_dx * np.sin(-yaw) + f_dy * np.cos(-yaw)
        future_steer_needed = abs(np.arctan((2 * self.wheelbase * f_local_y) / (Ld**2)))
        
        severity = max(abs(steering_angle), future_steer_needed) / 0.41
        min_allowed = min(3.5, self.max_speed)
        
        if self.max_speed > 0:
            target_speed = max(self.max_speed - (self.max_speed - min_allowed) * severity, min_allowed)
        else:
            target_speed = 0.0

        # 4. Actuate
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(target_speed)
        drive_msg.drive.steering_angle = float(steering_angle)
        self.pub_drive.publish(drive_msg)

        # --- 5. UPDATED LOGGING (Includes curr_x, curr_y) ---
        scan_ranges = np.array(scan_msg.ranges)
        scan_ranges = np.nan_to_num(scan_ranges, nan=0.0, posinf=10.0, neginf=0.0)
        idx = np.round(np.linspace(0, len(scan_ranges) - 1, 20)).astype(int)
        lidar_samples = scan_ranges[idx]

        record = {
            'timestamp': odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9,
            'curr_x': curr_x,
            'curr_y': curr_y,
            'v_curr': curr_v,
            'yaw': yaw,
            'pp_steering': steering_angle,
            'pp_speed': target_speed,
        }
        for i, val in enumerate(lidar_samples):
            record[f'lidar_{i}'] = val
        
        if self.max_speed > 0:
            self.data_buffer.append(record)

    def save_to_csv(self):
        if self.data_buffer:
            df = pd.DataFrame(self.data_buffer)
            file_exists = os.path.isfile(self.output_csv_path)
            # Append mode 'a' allows you to keep adding laps to the same file
            df.to_csv(self.output_csv_path, mode='a', index=False, header=not file_exists)
            self.get_logger().info(f"💾 SAVED {len(self.data_buffer)} SAMPLES TO {self.output_csv_path}")

def main(args=None):
    rclpy.init(args=args)
    node = AutomatedCurriculumLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_to_csv()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
