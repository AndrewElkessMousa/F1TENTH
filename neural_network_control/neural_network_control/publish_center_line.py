import os
import pandas as pd
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path

class CenterLinePublisher(Node):
    def __init__(self):
        super().__init__('center_line_publisher')

        # 1. Setup paths and load CSV
        self.package_share = get_package_share_directory('neural_network_control')
        csv_path = os.path.join(self.package_share, 'center_line_sp.csv')

        if not os.path.exists(csv_path):
            self.get_logger().error(f'Center line CSV not found: {csv_path}')
            raise FileNotFoundError(f'CSV not found at {csv_path}')

        # 2. Process waypoints
        df = pd.read_csv(csv_path, comment='#', header=None)
        # Select first two columns (x, y)
        waypoints_df = df.iloc[:, :2].apply(pd.to_numeric, errors='coerce').dropna()
        self.waypoints = waypoints_df.to_numpy(dtype=float)

        # 3. Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/center_line_marker_array', 10)
        self.path_pub = self.create_publisher(Path, '/center_line_path', 10)
        
        # 4. Timer (10Hz)
        self.timer = self.create_timer(0.1, self.publish_center_line)
        self.get_logger().info(f'Static Center Line Publisher started. Loaded {len(self.waypoints)} points.')

    def publish_center_line(self):
        now = self.get_clock().now().to_msg()
        
        # Create Path Message
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = now

        # Create Marker Message
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = now
        marker.ns = 'center_line'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.08  # Width of the line
        marker.color.r = 0.0
        marker.color.g = 0.7
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0

        for x, y in self.waypoints:
            # Add to Path
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            path_msg.poses.append(pose)

            # Add to Marker
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)

        # Close the loop if it's a circuit
        if len(self.waypoints) > 0:
            p_start = Point()
            p_start.x = float(self.waypoints[0][0])
            p_start.y = float(self.waypoints[0][1])
            marker.points.append(p_start)

        # Publish
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CenterLinePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()