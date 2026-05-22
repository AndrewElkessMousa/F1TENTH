"""
VESC (Flipsky FESC 4.12) Motor Driver for F1TENTH
Converts AckermannDriveStamped commands to VESC protocol over UART
"""

import rclpy
from rclpy.node import Node
import struct
import serial
import time
from ackermann_msgs.msg import AckermannDriveStamped


class VESCDriver(Node):
    """
    ROS 2 node for controlling Flipsky FESC 4.12 motor controller
    Subscribes to /drive (AckermannDriveStamped) and sends commands via UART
    """

    def __init__(self):
        super().__init__('vesc_driver')

        # Parameters
        self.port = self.declare_parameter('port', '/dev/ttyUSB1').value
        self.baudrate = self.declare_parameter('baudrate', 115200).value
        self.timeout = self.declare_parameter('timeout', 1.0).value
        
        # VESC Configuration
        self.max_speed = self.declare_parameter('max_speed', 50000).value  # RPM or ERPM
        self.max_steering = self.declare_parameter('max_steering', 0.5).value  # radians
        
        # Motor control scaling
        self.motor_pole_pairs = self.declare_parameter('motor_pole_pairs', 7).value
        self.gear_ratio = self.declare_parameter('gear_ratio', 1.0).value
        
        self.get_logger().info(f'VESC Driver initializing on {self.port} @ {self.baudrate} baud')
        
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.get_logger().info(f'✓ VESC connected on {self.port}')
        except Exception as e:
            self.get_logger().error(f'✗ Failed to connect to VESC: {e}')
            self.serial_port = None
        
        # Subscribe to drive commands
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped,
            '/drive',
            self.drive_callback,
            10
        )
        
        # State tracking
        self.current_speed = 0.0
        self.current_steering = 0.0
        
        self.get_logger().info('VESC Driver ready. Listening for commands on /drive')

    def drive_callback(self, msg: AckermannDriveStamped):
        """
        Convert AckermannDriveStamped to VESC motor commands
        
        Args:
            msg: AckermannDriveStamped message containing:
                - drive.speed: linear velocity (m/s)
                - drive.steering_angle: steering angle (radians)
        """
        try:
            speed = msg.drive.speed
            steering_angle = msg.drive.steering_angle
            
            # Clamp values
            speed = max(-self.max_speed / 1000, min(self.max_speed / 1000, speed))
            steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
            
            self.current_speed = speed
            self.current_steering = steering_angle
            
            # Convert to VESC commands
            # Speed: 0-100000 for ERPM (electrical RPM)
            # Steering: servo PWM command (typically 1000-2000 microseconds)
            
            vesc_speed = int(speed * 10000)  # Scale to ERPM range
            servo_command = self.steering_to_servo(steering_angle)
            
            self.send_vesc_command(vesc_speed, servo_command)
            
        except Exception as e:
            self.get_logger().error(f'Error in drive callback: {e}')

    def steering_to_servo(self, angle_rad: float) -> int:
        """
        Convert steering angle (radians) to servo PWM microseconds
        
        Assumes servo center at 1500 microseconds
        Max range: 1000-2000 microseconds (±500us from center)
        
        Args:
            angle_rad: Steering angle in radians
            
        Returns:
            Servo command in microseconds (1000-2000)
        """
        center = 1500
        max_deviation = 500
        
        # Normalize angle to [-1, 1]
        normalized = angle_rad / self.max_steering
        normalized = max(-1.0, min(1.0, normalized))
        
        servo_us = int(center + (normalized * max_deviation))
        return servo_us

    def send_vesc_command(self, speed_erpm: int, servo_us: int):
        """
        Send command to VESC via UART using VESC Protocol v3
        
        Args:
            speed_erpm: Motor speed in electrical RPM (can be negative)
            servo_us: Servo command in microseconds
        """
        if not self.serial_port or not self.serial_port.is_open:
            self.get_logger().warn('Serial port not open, cannot send command')
            return
        
        try:
            # VESC Protocol: frame structure
            # [START_FLAG] [LENGTH] [PAYLOAD] [CRC] [STOP_FLAG]
            # We'll use a simplified approach for PWM and servo
            
            # Format: 2 bytes for motor speed + 2 bytes for servo
            payload = struct.pack('>hh', speed_erpm, servo_us)
            
            # Calculate CRC16 (simplified)
            crc = self.crc16(payload)
            
            # Build frame
            frame = bytes([0xAB])  # START_FLAG
            frame += bytes([len(payload)])  # LENGTH
            frame += payload
            frame += struct.pack('>H', crc)  # CRC
            frame += bytes([0xAC])  # STOP_FLAG
            
            self.serial_port.write(frame)
            
        except Exception as e:
            self.get_logger().error(f'Error sending command: {e}')

    def crc16(self, data: bytes) -> int:
        """
        Calculate CRC16 checksum for VESC protocol
        
        Args:
            data: Payload bytes
            
        Returns:
            CRC16 value
        """
        crc = 0
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                crc <<= 1
                if crc & 0x10000:
                    crc ^= 0x1021
                crc &= 0xFFFF
        return crc

    def destroy_node(self):
        """Clean up serial port on shutdown"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    vesc_driver = VESCDriver()
    
    try:
        rclpy.spin(vesc_driver)
    except KeyboardInterrupt:
        pass
    finally:
        vesc_driver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
