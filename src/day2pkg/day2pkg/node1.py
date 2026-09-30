import math
import time

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import TwistStamped

TIMER_PERIOD = 0.1   # seconds between published commands
SPEED = 0.2          # forward speed, m/s
TURN_SPEED = 0.5     # turn speed when turning in place, rad/s

# Calibration values
STRAIGHT_CORRECTION = 1.0   #
TURN_CORRECTION = 1.0       # turning in place
ARC_CORRECTION = 1.0        # driving curves


# ---------- Twist helpers ----------

def CreateTwist(x, z):
    """TwistStamped with forward speed x (m/s) and turn speed z (rad/s)."""
    msg = TwistStamped()
    msg.header.stamp = Clock().now().to_msg()
    msg.header.frame_id = 'base_link'
    msg.twist.linear.x = float(x)
    msg.twist.angular.z = float(z)
    return msg


def CreateLinearTwist(x):
    return CreateTwist(x, 0.0)


def CreateAngularTwist(z):
    return CreateTwist(0.0, z)


# ---------- The node ----------

class Node1(Node):
    def __init__(self):
        super().__init__('node1')
        self.publisher_ = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.cmd_x = 0.0
        self.cmd_z = 0.0
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)

    def timer_callback(self):
        self.publisher_.publish(CreateTwist(self.cmd_x, self.cmd_z))

    # ----- low-level helpers -----

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def stop(self):
        self.cmd_x = 0.0
        self.cmd_z = 0.0
        try:
            self.publisher_.publish(CreateLinearTwist(0.0))
        except Exception:
            pass

    def drive_for(self, x, z, duration):
        if not rclpy.ok():
            return
        self.cmd_x = x
        self.cmd_z = z
        self.publisher_.publish(CreateTwist(x, z))
        self.wait(duration)
        self.stop()
        self.wait(0.5)   # let the robot fully stop before the next move

    # ----- Task 5 driving functions -----

    def DriveStraight(self, d):
        """Drive d meters in a straight line (negative d drives backward)."""
        duration = abs(d) / SPEED * STRAIGHT_CORRECTION
        x = SPEED if d >= 0 else -SPEED
        self.get_logger().info(f'DriveStraight({d}): {duration:.2f} s')
        self.drive_for(x, 0.0, duration)

    def DriveArc(self, r, d):
        """Arc of d degrees, radius r meters. Positive d = left, negative = right.
        r = 0 turns in place."""
        angle = math.radians(abs(d))
        direction = 1.0 if d >= 0 else -1.0

        if r == 0:
            duration = angle / TURN_SPEED * TURN_CORRECTION
            self.get_logger().info(f'DriveArc(0, {d}): turn in place, {duration:.2f} s')
            self.drive_for(0.0, direction * TURN_SPEED, duration)
        else:
            turn_speed = SPEED / r
            duration = (r * angle) / SPEED * ARC_CORRECTION
            self.get_logger().info(f'DriveArc({r}, {d}): {duration:.2f} s')
            self.drive_for(SPEED, direction * turn_speed, duration)

    def DriveCircle(self, r):
        """Full circle of radius r meters (curving left)."""
        self.DriveArc(r, 360)

    # ----- Task 6 -----

    def DriveD(self):
        """Drive a 'D' shape and end at the starting position and heading."""
        self.get_logger().info('Driving the D path')
        self.DriveStraight(1.0)     # the straight side of the D
        self.DriveArc(0, -90)       # turn right 90 degrees
        self.DriveArc(0.5, -180)    # half circle curving right
        self.DriveArc(0, -90)       # turn right 90 degrees
        self.get_logger().info('D path finished')


# ---------- Run a test chosen on the command line ----------

def main(args=None):
    rclpy.init(args=args)
    node = Node1()

    node.declare_parameter('test', 'D')
    test = node.get_parameter('test').value
    node.declare_parameter('distance', 1.0)
    distance = node.get_parameter('distance').value

    try:
        node.wait(2.0)   # give the robot time to connect

        if test == 'D':
            node.DriveD()
        elif test == 'straight':       # calibrate STRAIGHT_CORRECTION
            node.DriveStraight(distance)
        # ros2 run day2pkg node1 --ros-args -p test:=straight -p distance:=0.5
        elif test == 'spin':           # calibrate TURN_CORRECTION
            node.DriveArc(0, -360)
        elif test == 'halfcircle':     # calibrate ARC_CORRECTION
            node.DriveArc(0.5, -180)
        elif test == 'circle':
            node.DriveCircle(0.5)
        else:
            node.get_logger().error('Unknown test. Use D, straight, spin, or halfcircle.')

    except (KeyboardInterrupt, ExternalShutdownException):
        node.stop()

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()