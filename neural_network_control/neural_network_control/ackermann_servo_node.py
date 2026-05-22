#!/usr/bin/env python3
"""
ROS 2 node: ackermann_servo_node
---------------------------------
Subscribes to /ackermann_drive (ackermann_msgs/AckermannDriveStamped),
extracts the steering angle, and maps it to a PWM signal sent to a
PCA9685 channel connected to the steering servo.

DC motor speed is extracted and logged but not yet used.

Dependencies:
    pip install adafruit-pca9685 adafruit-blinka
    sudo apt install ros-<distro>-ackermann-msgs
"""

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

import Adafruit_PCA9685


# ---------------------------------------------------------------------------
# Tuning constants — adjust to your hardware
# ---------------------------------------------------------------------------

# I2C address of the PCA9685 board (default 0x40)
PCA9685_I2C_ADDRESS = 0x40

# PCA9685 channel wired to the steering servo
STEERING_SERVO_CHANNEL = 0

# PWM frequency for standard RC servos (50 Hz)
PWM_FREQUENCY_HZ = 50

# Servo pulse-width limits in microseconds
#   Typical values: 1000 µs (full-left) … 2000 µs (full-right)
#   Centre is ~1500 µs — tune these to YOUR servo's datasheet
SERVO_MIN_US = 1000   # pulse width → maximum left steering
SERVO_MAX_US = 2000   # pulse width → maximum right steering
SERVO_CTR_US = 1500   # pulse width → straight ahead

# Corresponding Ackermann steering-angle limits (radians)
#   The node will clamp any incoming angle to this range.
STEERING_ANGLE_MIN_RAD = -0.5   # full left  (≈ -28.6°)
STEERING_ANGLE_MAX_RAD =  0.5   # full right (≈ +28.6°)

# ---------------------------------------------------------------------------
# Helper: convert pulse-width (µs) → PCA9685 12-bit tick count
# ---------------------------------------------------------------------------

def us_to_ticks(pulse_us: float, pwm_freq_hz: int = PWM_FREQUENCY_HZ) -> int:
    """Convert a pulse width in microseconds to a PCA9685 OFF-tick value."""
    period_us = 1_000_000.0 / pwm_freq_hz          # e.g. 20 000 µs @ 50 Hz
    ticks = int(round(pulse_us / period_us * 4096))
    return max(0, min(4095, ticks))


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class AckermannServoNode(Node):

    def __init__(self):
        super().__init__('ackermann_servo_node')

        # ------------------------------------------------------------------
        # Declare ROS parameters (override from launch file or CLI)
        # ------------------------------------------------------------------
        self.declare_parameter('pca9685_address',      PCA9685_I2C_ADDRESS)
        self.declare_parameter('servo_channel',        STEERING_SERVO_CHANNEL)
        self.declare_parameter('pwm_frequency',        PWM_FREQUENCY_HZ)
        self.declare_parameter('servo_min_us',         SERVO_MIN_US)
        self.declare_parameter('servo_max_us',         SERVO_MAX_US)
        self.declare_parameter('servo_ctr_us',         SERVO_CTR_US)
        self.declare_parameter('steering_angle_min',   STEERING_ANGLE_MIN_RAD)
        self.declare_parameter('steering_angle_max',   STEERING_ANGLE_MAX_RAD)
        self.declare_parameter('ackermann_topic',      '/ackermann_drive')

        # Read parameters
        addr        = self.get_parameter('pca9685_address').value
        self.ch     = self.get_parameter('servo_channel').value
        freq        = self.get_parameter('pwm_frequency').value
        self.us_min = self.get_parameter('servo_min_us').value
        self.us_max = self.get_parameter('servo_max_us').value
        self.us_ctr = self.get_parameter('servo_ctr_us').value
        self.ang_min = self.get_parameter('steering_angle_min').value
        self.ang_max = self.get_parameter('steering_angle_max').value
        topic        = self.get_parameter('ackermann_topic').value

        # ------------------------------------------------------------------
        # Initialise PCA9685
        # ------------------------------------------------------------------
        self.get_logger().info(
            f'Connecting to PCA9685 at I2C address 0x{addr:02X} …'
        )
        self.pwm = Adafruit_PCA9685.PCA9685(address=addr)
        self.pwm.set_pwm_freq(freq)
        self.get_logger().info(f'PCA9685 ready — PWM frequency set to {freq} Hz')

        # Centre the servo on startup
        self._set_servo_angle(0.0)
        self.get_logger().info('Steering servo centred.')

        # ------------------------------------------------------------------
        # Subscriber
        # ------------------------------------------------------------------
        self.subscription = self.create_subscription(
            AckermannDriveStamped,
            topic,
            self._ackermann_callback,
            10
        )
        self.get_logger().info(f'Subscribed to {topic}')

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _ackermann_callback(self, msg: AckermannDriveStamped):
        drive = msg.drive

        steering_angle = drive.steering_angle   # radians
        speed          = drive.speed            # m/s (logged only for now)

        self.get_logger().debug(
            f'Received → steering_angle: {steering_angle:.4f} rad | '
            f'speed: {speed:.3f} m/s'
        )

        self._set_servo_angle(steering_angle)

    # ------------------------------------------------------------------
    # Steering → PWM
    # ------------------------------------------------------------------

    def _set_servo_angle(self, angle_rad: float):
        """Map a steering angle (rad) to a PCA9685 PWM pulse."""

        # Clamp to hardware limits
        angle_clamped = max(self.ang_min, min(self.ang_max, angle_rad))

        if angle_clamped != angle_rad:
            self.get_logger().warn(
                f'Steering angle {angle_rad:.4f} rad clamped to '
                f'{angle_clamped:.4f} rad'
            )

        # Linear interpolation: angle → pulse width (µs)
        # angle == 0   → centre pulse
        # angle < 0    → towards SERVO_MIN_US
        # angle > 0    → towards SERVO_MAX_US
        if angle_clamped >= 0.0:
            pulse_us = self.us_ctr + (angle_clamped / self.ang_max) * (
                self.us_max - self.us_ctr
            )
        else:
            pulse_us = self.us_ctr + (angle_clamped / abs(self.ang_min)) * (
                self.us_ctr - self.us_min
            )

        ticks = us_to_ticks(pulse_us)

        # PCA9685: start pulse at tick 0, end at computed tick
        self.pwm.set_pwm(self.ch, 0, ticks)

        self.get_logger().debug(
            f'Servo ch{self.ch} → {pulse_us:.1f} µs ({ticks} ticks)'
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        self.get_logger().info('Shutting down — centring servo.')
        self._set_servo_angle(0.0)
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = AckermannServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
