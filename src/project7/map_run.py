"""Task 1: occupancy grids on the way out and on the way back.

The robot drives DISTANCE metres straight ahead, updating a grid after each
0.1 m, turns 180 degrees, then drives the same distance back, accumulating a
second grid the same way. Each leg's grid has its own frame: the lidar's
position at the start of that leg, x along the robot's heading then (see
recording.py).

Every scan and odometry pose is saved, so the grids can be rebuilt without
the robot. Each run goes in its own folder, runs/<date-time>/:
  leg_out.json, leg_back.json   every update of each leg
  grid_out.png, grid_back.png   the finished grids, lidar positions in red

Usage:
  python3 map_run.py drive            drive the run and save it
  python3 map_run.py replay [folder]  rebuild the grids of a saved run
                                      (default: the newest one in runs/)

Add --show to also open the grids in a window (press any key to close it).

If no /scan message arrives, start the lidar motor:
  ros2 service call /start_motor std_srvs/srv/Empty
"""
import math
import os
import sys
import time

import cv2

from grid import OccupancyGrid
from recording import Leg, grid_from_leg, load_leg, save_leg

DISTANCE = 3.0  # metres each way
STEP = 0.1  # update the grid after each 0.1 m of travel
DRIVE_SPEED = 0.1  # m/s, slow so each scan is taken from a well-known pose
TURN_SPEED = 0.5  # rad/s
SETTLE_SEC = 0.5  # let the robot come to rest after a drive or turn
# base_link -> rplidar_link on our Turtlebot 4, used if TF2 has no answer:
# the lidar is 4 cm behind the centre and turned 90 degrees to the left.
LIDAR_IN_BASE = (-0.04, 0.0, math.pi / 2)
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
LEGS = ('out', 'back')


def save_image(grid, path, leg_name, folder):
    name = os.path.join(folder, f'grid_{leg_name}.png')
    cv2.imwrite(name, grid.image(path))
    print(f'  saved {name}')
    return name


def show_images(names):
    for name in names:
        cv2.imshow(os.path.basename(name), cv2.imread(name))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drive(show):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from action_msgs.msg import GoalStatus
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformListener
    from irobot_create_msgs.action import DriveDistance, RotateAngle

    def yaw_of_quaternion(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    class MapNode(Node):
        def __init__(self):
            super().__init__('map_run')
            self.scan = None
            self.odom = None
            self.create_subscription(LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
            self.create_subscription(Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.drive_client = ActionClient(self, DriveDistance, 'drive_distance')
            self.rotate_client = ActionClient(self, RotateAngle, 'rotate_angle')

        def _on_scan(self, msg):
            self.scan = msg

        def _on_odom(self, msg):
            self.odom = msg

        def wait_for(self, attr, timeout=10.0):
            setattr(self, attr, None)
            end = time.time() + timeout
            while getattr(self, attr) is None:
                if time.time() > end:
                    sys.exit(f'No /{attr} message in {timeout:.0f} s. '
                             'Is the lidar spinning? (ros2 service call /start_motor std_srvs/srv/Empty)')
                rclpy.spin_once(self, timeout_sec=0.1)
            return getattr(self, attr)

        def odom_pose(self):
            p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
            return (p.x, p.y, yaw_of_quaternion(q))

        def lidar_in_base(self):
            """base_link -> rplidar_link (x, y, yaw) from TF2, with a fallback."""
            end = time.time() + 3.0
            while time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.1)
                try:
                    t = self.tf_buffer.lookup_transform('base_link', 'rplidar_link', Time())
                except Exception:
                    continue
                tr = t.transform.translation
                return (tr.x, tr.y, yaw_of_quaternion(t.transform.rotation))
            print('  (TF2 had no base_link -> rplidar_link, using the measured mount)')
            return LIDAR_IN_BASE

        def start_action(self, client, goal, what):
            if not client.wait_for_server(timeout_sec=5.0):
                sys.exit(f'No /{what} action server.')
            send_future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            handle = send_future.result()
            if not handle.accepted:
                sys.exit(f'{what} goal was rejected.')
            return handle.get_result_async()

        def check_result(self, result_future, what):
            status = result_future.result().status
            if status != GoalStatus.STATUS_SUCCEEDED:
                print(f'  WARNING: {what} ended with status {status} '
                      '(4 = succeeded); was a bumper or cliff sensor triggered?')

    def drive_leg(name, folder):
        node.wait_for('odom')
        start_odom = node.odom_pose()
        leg = Leg(start_odom, base_lidar)
        grid = OccupancyGrid()
        path = []

        def travelled():
            x, y, _ = node.odom_pose()
            return math.hypot(x - start_odom[0], y - start_odom[1])

        def update(label):
            scan = node.wait_for('scan')
            odom = node.odom_pose()
            dist = travelled()
            leg.record(label, dist, odom, scan)
            pose = leg.world_T_lidar(odom)
            used = grid.add_scan(scan, pose)
            path.append((pose[0, 2], pose[1, 2]))
            print(f'  {label}: travelled {dist:.3f} m, lidar at '
                  f'x={pose[0, 2]:.3f} y={pose[1, 2]:.3f}, {used} rays used')

        print(f'Leg "{name}": driving {DISTANCE} m at {DRIVE_SPEED} m/s')
        update('0.0 m')
        goal = DriveDistance.Goal()
        goal.distance = DISTANCE
        goal.max_translation_speed = DRIVE_SPEED
        result_future = node.start_action(node.drive_client, goal, 'drive_distance')
        next_mark = STEP
        while not result_future.done():
            rclpy.spin_once(node, timeout_sec=0.05)
            # update at 0.1, 0.2, ...; the last update is made once stopped
            if travelled() >= next_mark and next_mark < DISTANCE - 1e-6:
                update(f'{next_mark:.1f} m')
                next_mark += STEP
        node.check_result(result_future, 'drive_distance')
        time.sleep(SETTLE_SEC)
        node.wait_for('odom')
        update(f'{DISTANCE:.1f} m (stopped)')
        print(f'  grid: {grid.summary()}')
        save_leg(leg, os.path.join(folder, f'leg_{name}.json'))
        return save_image(grid, path, name, folder)

    def turn_around():
        print('Turning 180 degrees')
        goal = RotateAngle.Goal()
        goal.angle = math.pi
        goal.max_rotation_speed = TURN_SPEED
        result_future = node.start_action(node.rotate_client, goal, 'rotate_angle')
        rclpy.spin_until_future_complete(node, result_future)
        node.check_result(result_future, 'rotate_angle')
        time.sleep(SETTLE_SEC)

    folder = os.path.join(RUNS, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(folder)
    print(f'Saving this run in {folder}')
    rclpy.init()
    node = MapNode()
    base_lidar = node.lidar_in_base()
    images = [drive_leg('out', folder)]
    turn_around()
    images.append(drive_leg('back', folder))
    node.destroy_node()
    rclpy.shutdown()
    if show:
        show_images(images)


def replay(folder, show):
    if folder is None:
        # skip runs that stopped before both legs were saved
        runs = sorted(r for r in (os.listdir(RUNS) if os.path.isdir(RUNS) else [])
                      if all(os.path.exists(os.path.join(RUNS, r, f'leg_{n}.json')) for n in LEGS))
        if not runs:
            sys.exit(f'No complete runs in {RUNS}. Drive one first.')
        folder = os.path.join(RUNS, runs[-1])
    print(f'Replaying {folder}')
    images = []
    for name in LEGS:
        leg = load_leg(os.path.join(folder, f'leg_{name}.json'))
        grid, path = grid_from_leg(leg)
        print(f'Leg "{name}": {len(leg.updates)} updates, '
              f'travelled {leg.updates[-1]["travelled"]:.3f} m')
        print(f'  grid: {grid.summary()}')
        images.append(save_image(grid, path, name, folder))
    if show:
        show_images(images)


def main():
    args = [a for a in sys.argv[1:] if a != '--show']
    show = '--show' in sys.argv
    mode = args[0] if args else None
    if mode == 'drive':
        drive(show)
    elif mode == 'replay':
        replay(args[1] if len(args) > 1 else None, show)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
