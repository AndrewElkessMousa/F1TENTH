import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
import sys, select, termios, tty

class ManualTeleop(Node):
    def __init__(self):
        super().__init__('manual_teleop_node')
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # --- SETTINGS ---
        self.current_speed = 4.0      # Starting speed
        self.speed_step = 0.5         # How much W/Q changes speed
        self.max_allowable = 12.0     # Cap for your stress test
        self.steering_angle = 0.41    # Max steering lock
        # ----------------

        self.settings = termios.tcgetattr(sys.stdin)
        self.get_logger().info(f"""
🚀 Manual Drive Active!
---------------------------
Arrows : Drive & Steer
W      : Increase Speed (+0.5)
Q      : Decrease Speed (-0.5)
Space  : EMERGENCY BRAKE
CTRL+C : Quit
---------------------------
Current Target Speed: {self.current_speed} m/s
        """)

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0.1)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, sys.stdin.fileno(), self.settings)
        return key

    def run(self):
        try:
            while True:
                key = self.get_key()
                drive_msg = AckermannDriveStamped()
                drive_msg.header.stamp = self.get_clock().now().to_msg()
                
                # Check for Arrow Keys (ANSI sequences)
                if key == '\x1b': # Escape character
                    key2 = sys.stdin.read(1)
                    key3 = sys.stdin.read(1)
                    if key3 == 'A': # Up Arrow
                        drive_msg.drive.speed = float(self.current_speed)
                    elif key3 == 'B': # Down Arrow
                        drive_msg.drive.speed = float(-self.current_speed) # Reverse
                    elif key3 == 'C': # Right Arrow
                        drive_msg.drive.steering_angle = float(-self.steering_angle)
                        drive_msg.drive.speed = float(self.current_speed / 2) # Slow in turns
                    elif key3 == 'D': # Left Arrow
                        drive_msg.drive.steering_angle = float(self.steering_angle)
                        drive_msg.drive.speed = float(self.current_speed / 2)
                
                # Speed Adjustments
                elif key.lower() == 'w':
                    self.current_speed = min(self.current_speed + self.speed_step, self.max_allowable)
                    print(f"📈 Speed: {self.current_speed} m/s", end='\r')
                elif key.lower() == 'q':
                    self.current_speed = max(self.current_speed - self.speed_step, 0.0)
                    print(f"📉 Speed: {self.current_speed} m/s", end='\r')
                
                # Emergency Brake
                elif key == ' ':
                    drive_msg.drive.speed = 0.0
                    drive_msg.drive.steering_angle = 0.0
                
                # Exit
                elif key == '\x03':
                    break

                self.pub.publish(drive_msg)

        except Exception as e:
            print(e)
        finally:
            # Stop car on exit
            stop_msg = AckermannDriveStamped()
            self.pub.publish(stop_msg)

def main():
    rclpy.init()
    node = ManualTeleop()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()