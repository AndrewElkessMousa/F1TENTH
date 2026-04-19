import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
import os

class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypoint_logger')
        self.subscription = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        
        # Path to save the file
        self.file_path = os.path.expanduser('~/sim_ws/src/controller/controller/waypoints.csv')
        
        # Open file and write header
        self.file = open(self.file_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['x', 'y'])
        
        self.last_x = 0.0
        self.last_y = 0.0
        self.threshold = 0.2  # Save a point every 20cm
        
        self.get_logger().info(f"📝 Logging waypoints to {self.file_path}. Drive the car!")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Only save if the car has moved enough (to avoid massive files)
        dist = ((x - self.last_x)**2 + (y - self.last_y)**2)**0.5
        
        if dist > self.threshold:
            self.writer.writerow([round(x, 4), round(y, 4)])
            self.last_x = x
            self.last_y = y
            self.get_logger().info(f"Saved: {x}, {y}")

    def __del__(self):
        self.file.close()

def main(args=None):
    rclpy.init(args=args)
    node = WaypointLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping logger and saving file...")
    finally:
        node.file.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()