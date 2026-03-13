import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # --- CONFIGURATION ---
        # Path to your recorded waypoints
        self.csv_path = os.path.expanduser('~/sim_ws/src/controller/controller/waypoints.csv')
        
        # Speed Limits (m/s)
        self.max_speed = 8  # Top speed on straights
        self.min_speed = 2   # Minimum speed for sharp turns
        
        # Lookahead Settings
        # Distance = (speed * lookahead_gain) + lookahead_min
        self.lookahead_gain = 0.1
        self.lookahead_min = 0.8
        self.lookahead_max = 3.0
        
        self.wheelbase = 0.33   # F1TENTH standard wheelbase
        # ---------------------

        self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.load_waypoints()
        self.get_logger().info(f"🏎️ Dynamic Velocity Node Active! Range: {self.min_speed} - {self.max_speed} m/s")

    def load_waypoints(self):
        try:
            if not os.path.exists(self.csv_path):
                self.get_logger().error(f"Could not find waypoints at {self.csv_path}")
                self.waypoints = None
                return
            df = pd.read_csv(self.csv_path)
            self.waypoints = df[['x', 'y']].values
            self.get_logger().info(f"✅ Loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"❌ CSV Load Error: {e}")
            self.waypoints = None

    def odom_callback(self, msg):
        if self.waypoints is None:
            return

        # 1. Get current state
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        curr_v = msg.twist.twist.linear.x  # Current speed of the car
        
        orient = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

        # 2. Dynamic Lookahead Distance
        # We look further ahead when going faster to maintain stability
        Ld = np.clip(curr_v * self.lookahead_gain + self.lookahead_min, 
                     self.lookahead_min, self.lookahead_max)

        # 3. Find Target Point
        distances = np.linalg.norm(self.waypoints - np.array([curr_x, curr_y]), axis=1)
        closest_idx = np.argmin(distances)
        
        target_idx = closest_idx
        for i in range(closest_idx, len(self.waypoints)):
            if distances[i] >= Ld:
                target_idx = i
                break
        target_point = self.waypoints[target_idx]

        # 4. Transform to Local Frame
        dx = target_point[0] - curr_x
        dy = target_point[1] - curr_y
        local_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)

        # 5. Calculate Steering
        steering_angle = np.arctan((2 * self.wheelbase * local_y) / (Ld**2))
        steering_angle = np.clip(steering_angle, -0.41, 0.41)

        # 6. Dynamic Speed Control
        # Calculate how "sharp" the turn is (0.0 to 1.0)
        turn_severity = abs(steering_angle) / 0.41
        
        # Calculate speed: slower in turns, faster on straights
        target_speed = self.max_speed - (self.max_speed - self.min_speed) * turn_severity

        # 7. Publish Drive Command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = float(target_speed)
        drive_msg.drive.steering_angle = float(steering_angle)
        
        self.pub_drive.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
