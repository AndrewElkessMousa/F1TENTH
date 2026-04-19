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

class ManualDataLogger(Node):
    def __init__(self):
        super().__init__('manual_data_logger')
        
        # --- CONFIGURATION ---
        self.output_csv_path = os.path.expanduser('~/sim_ws/src/controller/controller/training_data.csv')
        # ---------------------

        # Subscribers
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/ego_racecar/odom')
        self.scan_sub = message_filters.Subscriber(self, LaserScan, '/scan')
        self.drive_sub = message_filters.Subscriber(self, AckermannDriveStamped, '/drive')

        # Synchronize all three topics
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.scan_sub, self.drive_sub], 10, 0.05)
        self.ts.registerCallback(self.sync_callback)

        self.data_buffer = []
        self.get_logger().info("🕹️ Manual Logger Started! Drive to record dynamics + expert input.")

    def sync_callback(self, odom_msg, scan_msg, drive_msg):
        # 1. Get Current State (Critical for Dynamics)
        curr_x = odom_msg.pose.pose.position.x
        curr_y = odom_msg.pose.pose.position.y
        curr_v = odom_msg.twist.twist.linear.x
        
        orient = odom_msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

        # 2. Get Manual Input (Expert Labels)
        manual_steering = drive_msg.drive.steering_angle
        manual_speed = drive_msg.drive.speed

        # 3. Process LiDAR (Downsample to 20 rays)
        scan_ranges = np.array(scan_msg.ranges)
        scan_ranges = np.nan_to_num(scan_ranges, nan=0.0, posinf=10.0, neginf=0.0)
        idx = np.round(np.linspace(0, len(scan_ranges) - 1, 20)).astype(int)
        lidar_samples = scan_ranges[idx]

        # 4. Save Record with Physics Parameters
        record = {
            'timestamp': odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9,
            'curr_x': curr_x,
            'curr_y': curr_y,
            'v_curr': curr_v,
            'yaw': yaw,
            'pp_steering': manual_steering, # Matches column names in training script
            'pp_speed': manual_speed,
        }
        
        # Add lidar columns
        for i, val in enumerate(lidar_samples):
            record[f'lidar_{i}'] = val
        
        self.data_buffer.append(record)

        if len(self.data_buffer) % 100 == 0:
            self.get_logger().info(f"Captured {len(self.data_buffer)} manual samples...")

    def save_to_csv(self):
        if self.data_buffer:
            df = pd.DataFrame(self.data_buffer)
            # If file exists, append; otherwise, create new
            file_exists = os.path.isfile(self.output_csv_path)
            df.to_csv(self.output_csv_path, mode='a', index=False, header=not file_exists)
            self.get_logger().info(f"💾 SUCCESS: Added {len(self.data_buffer)} samples to {self.output_csv_path}")
        else:
            self.get_logger().warning("⚠️ No data recorded!")

def main(args=None):
    rclpy.init(args=args)
    node = ManualDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_to_csv()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()