import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    def __init__(self):
        super().__init__('wall_follow_node')

        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # PID Gains
        self.kp = 0.5
        self.kd = 0.08
        self.ki = 0.0
        
        self.prev_error = 0.0
        self.setpoint = 0.8    
        self.look_ahead = 0.5  
        
        # New Print to confirm startup
        self.get_logger().info("🚀 LEFT WALL FOLLOW NODE STARTED - VERSION 2.0")
        
    def get_range(self, range_data, angle):
        index = int((np.radians(angle) - range_data.angle_min) / range_data.angle_increment)
        if index < 0 or index >= len(range_data.ranges):
            return 4.0
        dist = range_data.ranges[index]
        if np.isinf(dist) or np.isnan(dist):
            return 4.0 
        return dist

    def scan_callback(self, msg):
        theta = np.radians(45) 
        a = self.get_range(msg, 45)  
        b = self.get_range(msg, 90)  

        alpha = np.arctan((a * np.cos(theta) - b) / (a * np.sin(theta)))
        dist_now = b * np.cos(alpha)
        dist_future = dist_now + self.look_ahead * np.sin(alpha)

        error = self.setpoint - dist_future
        p_val = self.kp * error
        d_val = self.kd * (error - self.prev_error)
        self.prev_error = error
        
        # Steering Logic: Turning away from LEFT wall
        steering_angle = -(p_val + d_val) 

        # --- LOGGING SECTION ---
        # This will print every ~1 second so it doesn't spam too fast
        self.get_logger().info(f"Ds1: {dist_now:.2f}m | Error: {error:.2f} | Steer: {np.degrees(steering_angle):.1f} deg", throttle_duration_sec=1.0)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 4.2 # Slow speed for testing
        drive_msg.drive.steering_angle = steering_angle
        self.publisher.publish(drive_msg)
def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
