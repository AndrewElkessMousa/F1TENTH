import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
from nav_msgs.msg import Odometry

class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypoint_logger')
        
        # --- CONFIGURATION ---
        # This will be your new path for the NN to test
        self.output_csv_path = os.path.expanduser('~/sim_ws/src/neural_network_control/neural_network_control/waypoints.csv')
        self.min_distance = 0.1  # Record a point every 10cm
        # ---------------------

        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        
        self.waypoints = []
        self.last_x = None
        self.last_y = None
        
        self.get_logger().info(f"📍 Waypoint Logger Started! Drive the car to map the path.")
        self.get_logger().info(f"💾 Saving to: {self.output_csv_path}")

    def odom_callback(self, msg):
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y

        # Initialize last position if this is the first point
        if self.last_x is None:
            self.add_waypoint(curr_x, curr_y)
            return

        # Calculate distance from the last recorded waypoint
        dist = np.sqrt((curr_x - self.last_x)**2 + (curr_y - self.last_y)**2)

        # Only log if the car has moved at least 0.1m
        if dist >= self.min_distance:
            self.add_waypoint(curr_x, curr_y)

    def add_waypoint(self, x, y):
        self.waypoints.append({'x': x, 'y': y})
        self.last_x = x
        self.last_y = y
        if len(self.waypoints) % 50 == 0:
            self.get_logger().info(f"Recorded {len(self.waypoints)} waypoints...")

    def save_to_csv(self):
        if len(self.waypoints) > 5:
            df = pd.DataFrame(self.waypoints)
            df.to_csv(self.output_csv_path, index=False)
            self.get_logger().info(f"✅ SUCCESS: Saved {len(self.waypoints)} waypoints to {self.output_csv_path}")
        else:
            self.get_logger().warning("⚠️ Too few points recorded. Did you move the car?")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_to_csv()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()