import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Pure Pursuit Parameters
        self.lookahead_distance = 1.0  # L (The "carrot" distance)
        self.wheelbase = 0.33          # L (Distance between axles)
        self.speed = 1.0               # Speed in m/s
        self.target_radius = 2.0       # The radius of the circle we want to follow

        self.get_logger().info(f"🚀 Pure Pursuit Active! Following a {self.target_radius}m circle.")

    def odom_callback(self, msg):
        # 1. Get current position and heading
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

        # 2. Geometry: Find a goal point on our virtual circle path
        # We calculate a point that is always 'ahead' of our current heading
        target_x = self.target_radius * np.sin(yaw + 0.5) 
        target_y = self.target_radius * (1 - np.cos(yaw + 0.5))

        # 3. Transform Goal Point to Car's Local Frame (the core of Pure Pursuit)
        dx = target_x - pos.x
        dy = target_y - pos.y
        local_x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
        local_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)

        # 4. Pure Pursuit Steering Law: delta = atan( (2 * L * sin(alpha)) / lookahead )
        # Simplified for F1TENTH: steering = (2 * local_y * wheelbase) / (lookahead^2)
        L_sq = self.lookahead_distance**2
        steering_angle = (2 * local_y * self.wheelbase) / L_sq

        # 5. Limit and Publish
        steering_angle = np.clip(steering_angle, -0.41, 0.41) # Max Ackerman limit
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = steering_angle
        self.pub_drive.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()