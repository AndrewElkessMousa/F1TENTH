import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion

class PurePursuitRacing(Node):
    def __init__(self):
        super().__init__('pure_pursuit_racing')
        
        # --- CONFIGURATION ---
        base_path = os.path.expanduser('~/sim_ws/src/controller/controller/')
        self.waypoints_path = os.path.join(base_path, 'waypoints.csv')
        
        self.wheelbase = 0.33
        self.MAX_SPEED = 12  
        self.MIN_SPEED = 3.5
        self.plot_output_dir = os.path.join(base_path, 'lap_plots')
        os.makedirs(self.plot_output_dir, exist_ok=True)

        # --- LAP TIMING ---
        self.lap_count = 0
        self.last_pos = None
        self.total_dist = 0.0
        self.finish_threshold = 1.2
        self.cooldown_dist = 20.0
        self.lap_start_time = None
        self.current_lap_x = []
        self.current_lap_y = []

        # Subscribers & Publishers
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/ego_racecar/drive', 10)

        self.load_waypoints()
        self.get_logger().info(f"🏎️ PURE PURSUIT ACTIVE: Speed {self.MAX_SPEED}m/s | Logging Disabled")

    def load_waypoints(self):
        if not os.path.exists(self.waypoints_path):
            self.get_logger().error(f"❌ Waypoints not found at {self.waypoints_path}")
            return
        self.waypoints = pd.read_csv(self.waypoints_path)[['x', 'y']].values

    def odom_callback(self, msg):
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        curr_v = msg.twist.twist.linear.x
        
        # Orientation
        orient = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

        # 1. LAP TIMING
        curr_pos = np.array([curr_x, curr_y])
        self.current_lap_x.append(curr_x)
        self.current_lap_y.append(curr_y)
        if self.last_pos is not None:
            self.total_dist += np.linalg.norm(curr_pos - self.last_pos)
            dist_to_start = np.linalg.norm(curr_pos - self.waypoints[0])
            
            if dist_to_start < self.finish_threshold and self.total_dist > self.cooldown_dist:
                now = self.get_clock().now().nanoseconds / 1e9
                if self.lap_start_time:
                    self.get_logger().info(f"⏱️ LAP {self.lap_count+1}: {now - self.lap_start_time:.3f}s")
                self.lap_start_time = now
                self.save_lap_path_plot(self.lap_count + 1)
                self.lap_count += 1
                self.total_dist = 0.0
                self.current_lap_x = [curr_x]
                self.current_lap_y = [curr_y]
        self.last_pos = curr_pos

        # 2. PURE PURSUIT MATH
        Ld = np.clip(curr_v * 0.12 + 0.8, 0.8, 3.5) # Lookahead distance
        
        # Find target point
        dists = np.linalg.norm(self.waypoints - curr_pos, axis=1)
        closest_idx = np.argmin(dists)
        
        target_idx = closest_idx
        for i in range(closest_idx, closest_idx + 100):
            idx = i % len(self.waypoints)
            if dists[idx] >= Ld:
                target_idx = idx
                break
        
        # Transform to local frame
        target_pt = self.waypoints[target_idx]
        dx, dy = target_pt[0] - curr_x, target_pt[1] - curr_y
        local_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)
        
        # Steering calculation
        steer = np.arctan((2 * self.wheelbase * local_y) / (Ld**2))
        steer = np.clip(steer, -0.41, 0.41)

        # 3. PREDICTIVE BRAKING
        # Look ahead even further for curvature
        future_idx = (target_idx + 12) % len(self.waypoints)
        f_pt = self.waypoints[future_idx]
        fl_y = (f_pt[0]-curr_x) * np.sin(-yaw) + (f_pt[1]-curr_y) * np.cos(-yaw)
        f_steer = abs(np.arctan((2 * self.wheelbase * fl_y) / (Ld**2)))
        
        severity = max(abs(steer), f_steer) / 0.41
        speed = max(self.MAX_SPEED - (self.MAX_SPEED - self.MIN_SPEED) * severity, self.MIN_SPEED)

        # 4. PUBLISH DRIVE
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(drive_msg)

    def save_lap_path_plot(self, lap_number):
        if len(self.current_lap_x) < 2:
            return

        actual_path = np.column_stack((np.array(self.current_lap_x), np.array(self.current_lap_y)))
        desired_path = np.vstack((self.waypoints, self.waypoints[0]))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(desired_path[:, 0], desired_path[:, 1], color='green', linewidth=2.5, label='Desired path')
        ax.plot(actual_path[:, 0], actual_path[:, 1], color='red', linestyle='--', linewidth=2, label='Actual path')
        ax.scatter(self.waypoints[0][0], self.waypoints[0][1], color='blue', s=60, label='Start')
        ax.set_title(f'Lap {lap_number} Path Comparison')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        save_path = os.path.join(self.plot_output_dir, f'lap_{lap_number:03d}_path.png')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        self.get_logger().info(f"🖼️ Saved lap path plot: {save_path}")

def main():
    rclpy.init()
    rclpy.spin(PurePursuitRacing())
    rclpy.shutdown()

if __name__ == '__main__':
    main()