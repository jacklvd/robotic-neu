"""Do one run and record the odometry before and after it.

Reads /odom, sends a goal to our Driving Node (/drive), waits for the robot
to finish and settle, reads /odom again, and prints both poses, the distance
between them and the heading change. Nothing is written to disk.

Usage:  python3 record_run.py           1 m forward (Task 2)
        python3 record_run.py circle    left circle, 0.5 m radius (Task 3)
        python3 record_run.py dd        1 m with the built-in DriveDistance (Task 4)
"""
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry

from day3_interfaces.action import Drive
from irobot_create_msgs.action import DriveDistance

DISTANCE = 1.0  # metres
CIRCLE_RADIUS = 0.5  # metres
SETTLE_SEC = 1.0  # let the robot come to rest before the second reading


class RecordRun(Node):
    def __init__(self):
        super().__init__('record_run')
        self._odom = None
        self.create_subscription(Odometry, '/odom', self._on_odom,
                                 qos_profile_sensor_data)
        self._client = ActionClient(self, Drive, 'drive')
        self._dd_client = ActionClient(self, DriveDistance, 'drive_distance')

    def _on_odom(self, msg):
        self._odom = msg

    def read_position(self):
        """Wait for a fresh /odom message and return its (x, y, yaw in degrees)."""
        self._odom = None
        while self._odom is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        p = self._odom.pose.pose.position
        q = self._odom.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return p.x, p.y, math.degrees(yaw)

    def drive(self, circle=False, built_in=False):
        client = self._dd_client if built_in else self._client
        if not client.wait_for_server(timeout_sec=5.0):
            sys.exit('No action server found for this drive.')
        goal = DriveDistance.Goal() if built_in else Drive.Goal()
        if built_in:
            goal.distance = DISTANCE
        elif circle:
            goal.arc_left = True
            goal.radius = CIRCLE_RADIUS
            goal.arc_degrees = 360.0
        else:
            goal.distance = DISTANCE
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if not handle.accepted:
            sys.exit('Goal was rejected.')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        if result_future.result().status != GoalStatus.STATUS_SUCCEEDED:
            sys.exit('Drive did not finish (canceled or aborted). Nothing recorded.')


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    circle, built_in = mode == 'circle', mode == 'dd'
    what = f'a {CIRCLE_RADIUS} m circle' if circle else f'{DISTANCE} m'
    if built_in:
        what += ' with DriveDistance'
    rclpy.init()
    node = RecordRun()
    start = node.read_position()
    print(f'start x={start[0]:.4f} y={start[1]:.4f} yaw={start[2]:.2f} deg, driving {what}...')
    node.drive(circle, built_in)
    time.sleep(SETTLE_SEC)
    end = node.read_position()
    node.destroy_node()
    rclpy.shutdown()

    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    turn = (end[2] - start[2] + 180) % 360 - 180
    print(f'end   x={end[0]:.4f} y={end[1]:.4f} yaw={end[2]:.2f} deg')
    print(f'odometry distance = {dist:.4f} m, heading change = {turn:+.2f} deg')


if __name__ == '__main__':
    main()
