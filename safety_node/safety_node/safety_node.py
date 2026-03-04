import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

class AEBNode(Node):
    def __init__(self):
        super().__init__('aeb_node')
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.car_width = 0.35        
        self.lidar_to_front = 0.15   
        self.current_v = 0.0
        self.ttc_threshold = 0.5     

    def odom_callback(self, odom_msg):
        self.current_v = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        if self.current_v <= 0.01: return
        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        
        y_dist = ranges * np.sin(angles)
        width_mask = np.abs(y_dist) < (self.car_width / 2.0)
        long_ranges = (ranges * np.cos(angles)) - self.lidar_to_front
        range_rate = self.current_v * np.cos(angles)
        
        valid = np.where(width_mask & (long_ranges > 0) & (range_rate > 0))
        if len(valid[0]) == 0: return

        ttc = long_ranges[valid] / range_rate[valid]
        if np.min(ttc) < self.ttc_threshold:
            self.brake()
            self.get_logger().warn(f"⚠️ AEB: BRAKING! TTC: {np.min(ttc):.2f}s")

    def brake(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        self.drive_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AEBNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()