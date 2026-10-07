"""Driving Node: the Action Server for the Drive action.

Receives a Request for a distance to move or an angle to turn, publishes the
cmd_vel messages to carry it out, and sends a Response when it has finished.
"""

import math
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

from day3_interfaces.action import Drive

TIMER_PERIOD = 0.1   # seconds between published cmd_vel messages
SPEED = 0.2          # forward speed, m/s
TURN_SPEED = 0.5     # turn speed when turning in place, rad/s

# Calibration values (same meaning as in day2pkg)
STRAIGHT_CORRECTION = 1.0
TURN_CORRECTION = 1.0
ARC_CORRECTION = 1.0


def CreateTwist(x, z):
    """TwistStamped with forward speed x (m/s) and turn speed z (rad/s)."""
    msg = TwistStamped()
    msg.header.stamp = Clock().now().to_msg()
    msg.header.frame_id = 'base_link'
    msg.twist.linear.x = float(x)
    msg.twist.angular.z = float(z)
    return msg


class DrivingNode(Node):
    def __init__(self):
        super().__init__('driving')

        self.publisher_ = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        # A reentrant group so a cancel request can be served while a goal is
        # still executing (needed for Extra Credit B).
        self._group = ReentrantCallbackGroup()
        self._server = ActionServer(
            self,
            Drive,
            'drive',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._group,
        )
        self.get_logger().info("Driving Node ready, listening on action 'drive'")

    # ----- Action callbacks -----

    def goal_callback(self, goal_request):
        """Reject a Request if both parts are non-zero, or if either is negative."""
        d = goal_request.distance
        a = goal_request.angle
        shape = goal_request.shape.strip().lower()
        r = goal_request.radius
        deg = goal_request.arc_degrees

        if r != 0.0 or deg != 0.0:
            if d != 0.0 or a != 0.0 or shape:
                self.get_logger().warn('REJECT: arc given with another command')
                return GoalResponse.REJECT
            if r <= 0.0 or not 0.0 < deg <= 360.0:
                self.get_logger().warn(f'REJECT: bad arc (radius={r}, degrees={deg})')
                return GoalResponse.REJECT
            side = 'left' if goal_request.arc_left else 'right'
            self.get_logger().info(f'ACCEPT: arc {side}, radius={r} m, {deg} deg')
            return GoalResponse.ACCEPT

        if shape:
            if shape not in ('figure8', 'p'):
                self.get_logger().warn(f'REJECT: unknown shape {shape!r}')
                return GoalResponse.REJECT
            if d != 0.0 or a != 0.0:
                self.get_logger().warn('REJECT: shape given with a distance or angle')
                return GoalResponse.REJECT
            self.get_logger().info(f'ACCEPT: shape {shape}')
            return GoalResponse.ACCEPT

        if d != 0.0 and a != 0.0:
            self.get_logger().warn(f'REJECT: both non-zero (distance={d}, angle={a})')
            return GoalResponse.REJECT
        if d < 0.0 or a < 0.0:
            self.get_logger().warn(f'REJECT: negative value (distance={d}, angle={a})')
            return GoalResponse.REJECT

        self.get_logger().info(f'ACCEPT: distance={d} m, angle={a} deg')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Cancel requested')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        shape = goal.shape.strip().lower()

        if shape == 'figure8':
            finished = self.DriveFigure8(goal_handle)
        elif shape == 'p':
            finished = self.DriveP(goal_handle)
        elif goal.arc_degrees != 0.0:
            d = goal.arc_degrees if goal.arc_left else -goal.arc_degrees
            finished = self.DriveArc(goal_handle, goal.radius, d)
        elif goal.distance != 0.0:
            finished = self.DriveStraight(goal_handle, goal.distance)
        elif goal.angle != 0.0:
            finished = self.DriveArc(goal_handle, 0.0, goal.angle)
        else:
            # distance and angle are both 0: nothing to do
            finished = True

        self.stop()

        if not finished:
            goal_handle.canceled()
            self.get_logger().info('Goal canceled')
        else:
            goal_handle.succeed()
            self.get_logger().info('Goal succeeded')

        # The Response is empty for now
        return Drive.Result()

    # ----- Low-level driving helpers -----

    def stop(self):
        """Send a cmd_vel message to stop moving."""
        self.publisher_.publish(CreateTwist(0.0, 0.0))

    def drive_for(self, goal_handle, x, z, duration):
        """Publish cmd_vel at TIMER_PERIOD for duration seconds.

        Returns True if it ran to the end, False if the goal was canceled.
        """
        ticks = max(1, int(round(duration / TIMER_PERIOD)))
        for _ in range(ticks):
            if not rclpy.ok():
                return False
            if goal_handle.is_cancel_requested:
                self.stop()
                return False
            self.publisher_.publish(CreateTwist(x, z))
            time.sleep(TIMER_PERIOD)
        self.stop()
        time.sleep(0.3)   # let the robot settle before the next move
        return True

    def DriveStraight(self, goal_handle, d):
        """Drive d meters in a straight line."""
        duration = abs(d) / SPEED * STRAIGHT_CORRECTION
        self.get_logger().info(f'DriveStraight({d}): {duration:.2f} s')
        return self.drive_for(goal_handle, SPEED, 0.0, duration)

    def DriveArc(self, goal_handle, r, d):
        """Arc of d degrees, radius r meters. Positive d = left. r = 0 turns in place."""
        angle = math.radians(abs(d))
        direction = 1.0 if d >= 0 else -1.0

        if r == 0:
            duration = angle / TURN_SPEED * TURN_CORRECTION
            self.get_logger().info(f'DriveArc(0, {d}): turn in place, {duration:.2f} s')
            return self.drive_for(goal_handle, 0.0, direction * TURN_SPEED, duration)

        turn_speed = SPEED / r
        duration = (r * angle) / SPEED * ARC_CORRECTION
        self.get_logger().info(f'DriveArc({r}, {d}): {duration:.2f} s')
        return self.drive_for(goal_handle, SPEED, direction * turn_speed, duration)

    # ----- Extra Credit A: named patterns -----

    def DriveFigure8(self, goal_handle):
        """A figure-8: a full circle to the left, then a full circle to the right."""
        self.get_logger().info('Driving a figure-8')
        return (self.DriveArc(goal_handle, 0.5, 360)
                and self.DriveArc(goal_handle, 0.5, -360))

    def DriveP(self, goal_handle):
        """A letter P: the long stem, then the loop at the top."""
        self.get_logger().info('Driving a letter P')
        return (self.DriveStraight(goal_handle, 1.0)      # up the stem
                and self.DriveArc(goal_handle, 0, -90)    # face right
                and self.DriveArc(goal_handle, 0.25, -180)  # the loop
                and self.DriveArc(goal_handle, 0, -90)    # face back down the stem
                and self.DriveStraight(goal_handle, 0.5))  # down to the start


def main(args=None):
    rclpy.init(args=args)
    node = DrivingNode()
    executor = MultiThreadedExecutor()

    try:
        rclpy.spin(node, executor=executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.stop()

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
