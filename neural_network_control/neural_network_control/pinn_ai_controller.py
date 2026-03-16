import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import pickle
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker # Added for visualization
from geometry_msgs.msg import Point # Added for visualization
from tf_transformations import euler_from_quaternion

class F1TENTH_PINN(nn.Module):
    def __init__(self):
        super(F1TENTH_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

class PINNInferenceNode(Node):
    def __init__(self):
        super().__init__('pinn_inference_node')
        
        base_path = os.path.expanduser('~/sim_ws/src/neural_network_control/neural_network_control/')
        model_path = os.path.join(base_path, 'pinn_model_v2.pth')
        scaler_path = os.path.join(base_path, 'scaler_v2.pkl')
        waypoints_path = os.path.join(base_path, 'waypoints.csv')

        # Load Waypoints
        self.waypoints_df = pd.read_csv(waypoints_path)
        self.waypoints = self.waypoints_df[['x', 'y']].values
        
        # Load Scaler & Model
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = F1TENTH_PINN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.get_logger().info(f"🚀 Model Loaded on {self.device}. Scaling verified.")

        # --- ROS COMMS ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(Marker, '/global_waypoints_marker', 10) # Added Marker Pub
        
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Timer to publish the path to RViz every 2 seconds
        self.viz_timer = self.create_timer(2.0, self.publish_waypoints_marker)

        self.curr_x, self.curr_y, self.yaw, self.v_curr = 0.0, 0.0, 0.0, 0.0

    def publish_waypoints_marker(self):
        """Publishes the CSV waypoints as a green line in RViz."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "global_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Line width and color (Green)
        marker.scale.x = 0.1 
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        for x, y in self.waypoints:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05 # Slightly above ground
            marker.points.append(p)

        self.marker_pub.publish(marker)

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.v_curr = msg.twist.twist.linear.x
        orient = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

    def scan_callback(self, scan_msg):
        # ... (Transformation logic remains identical to your working code) ...
        Ld_feat = 1.5 
        dists = np.linalg.norm(self.waypoints - np.array([self.curr_x, self.curr_y]), axis=1)
        closest_idx = np.argmin(dists)
        
        target_idx = closest_idx
        for i in range(closest_idx, closest_idx + 100):
            idx = i % len(self.waypoints)
            if dists[idx] >= Ld_feat:
                target_idx = idx
                break
        
        target_pt = self.waypoints[target_idx]
        dx_g, dy_g = target_pt[0] - self.curr_x, target_pt[1] - self.curr_y
        
        dx_local = dx_g * np.cos(self.yaw) + dy_g * np.sin(self.yaw)
        dy_local = -dx_g * np.sin(self.yaw) + dy_g * np.cos(self.yaw)
        
        path_yaw = np.arctan2(dy_g, dx_g)
        err_yaw = (path_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi

        scan_ranges = np.nan_to_num(np.array(scan_msg.ranges), nan=0.0, posinf=10.0)
        idx_samples = np.round(np.linspace(0, len(scan_ranges) - 1, 20)).astype(int)
        lidar_samples = scan_ranges[idx_samples]

        features = np.array([self.v_curr, dx_local, dy_local, err_yaw] + list(lidar_samples)).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            prediction = self.model(features_tensor).cpu().numpy()[0]
        
        # Publish commands
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(prediction[1])
        drive_msg.drive.steering_angle = float(prediction[0])
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PINNInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()