import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray # For the green dots
from tf_transformations import euler_from_quaternion

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # --- CONFIGURATION ---
        self.csv_path = os.path.expanduser('~/sim_ws/src/controller/controller/waypoints.csv')
        self.max_speed = 8.0
        self.min_speed = 2.0
        self.lookahead_gain = 0.1
        self.lookahead_min = 0.8
        self.lookahead_max = 3.0
        self.wheelbase = 0.33
        
        # --- ACCURATE LAP TIMER SETTINGS ---
        self.lap_start_time = None
        self.lap_count = 0
        self.finish_line_threshold = 1.2   # Increased for accuracy at 8m/s
        self.cooldown_dist = 10.0          
        self.total_dist_traveled = 0.0
        self.last_pos = None
        self.lap_triggered = False         
        # ---------------------

        # Subscribers and Publishers
        self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # Visualization Publisher
        self.viz_pub = self.create_publisher(MarkerArray, '/viz_path', 10)

        self.load_waypoints()
        
        # Timer to publish the green dots every 2 seconds
        self.create_timer(2.0, self.publish_static_path)
        
        self.get_logger().info(f"🏎️ Pure Pursuit Active! Green path dots enabled.")

    def load_waypoints(self):
        try:
            df = pd.read_csv(self.csv_path)
            self.waypoints = df[['x', 'y']].values
            self.get_logger().info(f"✅ Loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"❌ CSV Load Error: {e}")
            self.waypoints = None

    def publish_static_path(self):
        """Publishes the green spheres to RViz."""
        if self.waypoints is None:
            return

        marker_array = MarkerArray()
        # We step by 5 to avoid overloading RViz with too many markers
        for i in range(0, len(self.waypoints), 5):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(self.waypoints[i][0])
            marker.pose.position.y = float(self.waypoints[i][1])
            marker.pose.position.z = 0.0
            
            # Scale of the dots
            marker.scale.x, marker.scale.y, marker.scale.z = 0.15, 0.15, 0.15
            
            # Color: Green
            marker.color.a = 1.0 # Alpha
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
            marker_array.markers.append(marker)
        
        self.viz_pub.publish(marker_array)

    def update_lap_timer(self, x, y):
        curr_pos = np.array([x, y])
        if self.last_pos is None:
            self.last_pos = curr_pos
            self.lap_start_time = self.get_clock().now()
            return

        step_dist = np.linalg.norm(curr_pos - self.last_pos)
        self.total_dist_traveled += step_dist
        self.last_pos = curr_pos

        dist_to_finish = np.linalg.norm(curr_pos - self.waypoints[0])

        if dist_to_finish < self.finish_line_threshold:
            if not self.lap_triggered and self.total_dist_traveled > self.cooldown_dist:
                now = self.get_clock().now()
                if self.lap_start_time is not None:
                    lap_time_sec = (now - self.lap_start_time).nanoseconds / 1e9
                    self.lap_count += 1
                    self.get_logger().info(f"🏁 [BASELINE] LAP {self.lap_count} | TIME: {lap_time_sec:.3f}s")
                
                self.lap_start_time = now
                self.total_dist_traveled = 0.0
                self.lap_triggered = True
        else:
            if self.lap_triggered:
                self.lap_triggered = False

    def odom_callback(self, msg):
        if self.waypoints is None:
            return

        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        curr_v = msg.twist.twist.linear.x
        
        self.update_lap_timer(curr_x, curr_y)

        orient = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

        Ld = np.clip(curr_v * self.lookahead_gain + self.lookahead_min, 
                     self.lookahead_min, self.lookahead_max)

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

        turn_severity = abs(steering_angle) / 0.41
        target_speed = self.max_speed - (self.max_speed - self.min_speed) * turn_severity

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