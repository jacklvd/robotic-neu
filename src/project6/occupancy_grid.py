"""Occupancy grid from the Turtlebot's lidar (/scan).

The grid is 200 x 200 cells of 5 cm, so it covers 10 m x 10 m. The world
frame {W} has its origin at the lidar's position when the program starts (the
centre cell of the grid) and its x axis along the robot's forward direction. Each cell holds a log-odds value L, starting at 0
(unknown). For each lidar ray, Bresenham ray casting from the sensor to the
hit point subtracts 0.95 from every cell the ray passes through (empty) and
adds 0.95 to the cell it hit (filled).

Usage:
  python3 occupancy_grid.py file scan_sample.txt   Task 1 offline, from a saved scan
  python3 occupancy_grid.py scan                   Task 1, one live scan
  python3 occupancy_grid.py drive                  Task 2, drive 1 m, update every 0.1 m

Add --show to also open each grid in a window (press any key to close it).
Images are always saved as PNG files next to this script.

Save a scan with:  ros2 topic echo --once --full-length /scan > scan_sample.txt
(without --full-length, ros2 cuts the ranges list off after 128 values)
"""
import math
import os
import sys
import time

import cv2
import numpy as np

GRID_SIZE = 200  # cells per side
CELL = 0.05  # metres per cell
ORIGIN = GRID_SIZE // 2  # grid index of the world origin
L_FILLED = math.log10(0.9 / 0.1)  # +0.95, cell the ray hit
L_EMPTY = math.log10(0.1 / 0.9)  # -0.95, cells the ray passed through
STEP = 0.1  # Task 2: update the grid after each 0.1 m of travel
DRIVE_DISTANCE = 1.0
DRIVE_SPEED = 0.1  # m/s, slow so each scan is taken from a well-known pose
# base_link -> rplidar_link on our Turtlebot 4, used if TF2 has no answer:
# the lidar is 4 cm behind the centre and turned 90 degrees to the left.
LIDAR_IN_BASE = (-0.04, 0.0, math.pi / 2)
HERE = os.path.dirname(os.path.abspath(__file__))


def pose_matrix(x, y, yaw):
    """3x3 homogeneous transform for a 2D pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def to_cell(x, y):
    """World point (metres) to grid indices (i along x, j along y)."""
    return ORIGIN + math.floor(x / CELL), ORIGIN + math.floor(y / CELL)


def in_grid(i, j):
    return 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE


def bresenham(i0, j0, i1, j1):
    """All cells from (i0, j0) up to but not including (i1, j1).

    The integer version of stepping one cell at a time along the longer axis
    and adding the fraction dj/di to the other axis, for every direction.
    """
    di, dj = abs(i1 - i0), abs(j1 - j0)
    si = 1 if i1 > i0 else -1
    sj = 1 if j1 > j0 else -1
    cells = []
    i, j = i0, j0
    if di >= dj:
        frac = 0
        for _ in range(di):
            cells.append((i, j))
            i += si
            frac += dj
            if 2 * frac > di:
                j += sj
                frac -= di
    else:
        frac = 0
        for _ in range(dj):
            cells.append((i, j))
            j += sj
            frac += di
            if 2 * frac > dj:
                i += si
                frac -= dj
    return cells


class OccupancyGrid:
    def __init__(self):
        self.L = np.zeros((GRID_SIZE, GRID_SIZE))

    def add_scan(self, scan, world_T_sensor):
        """Ray-cast one LaserScan (a dict or a ROS message) into the grid.

        world_T_sensor is the 3x3 pose of the lidar in the world frame.
        """
        get = scan.get if isinstance(scan, dict) else lambda k: getattr(scan, k)
        angle_min, step = get('angle_min'), get('angle_increment')
        range_min, range_max = get('range_min'), get('range_max')
        x0, y0 = world_T_sensor[0, 2], world_T_sensor[1, 2]
        i0, j0 = to_cell(x0, y0)
        used = 0
        for k, d in enumerate(get('ranges')):
            # values outside [range_min, range_max] (or inf / nan) are discarded
            if not (range_min <= d <= range_max):
                continue
            theta = angle_min + k * step
            hit = world_T_sensor @ np.array([d * math.cos(theta), d * math.sin(theta), 1.0])
            i1, j1 = to_cell(hit[0], hit[1])
            for i, j in bresenham(i0, j0, i1, j1):
                if in_grid(i, j):
                    self.L[i, j] += L_EMPTY
            if in_grid(i1, j1):
                self.L[i1, j1] += L_FILLED
            used += 1
        return used

    def image(self, robot_path=(), scale=3):
        """Greyscale picture: black = filled, white = empty, grey = unknown.

        Up in the picture is the robot's starting forward direction (+x),
        left is +y. Red dots are where the lidar was for each update.
        """
        p = 1.0 - 1.0 / (1.0 + np.power(10.0, self.L))  # log-odds -> probability
        grey = (255 * (1.0 - p)).astype(np.uint8)
        grey = grey[::-1, ::-1]  # row 0 = far ahead, column 0 = far left
        img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        def pixel(x, y):
            i, j = to_cell(x, y)
            return ((GRID_SIZE - 1 - j) * scale + scale // 2,
                    (GRID_SIZE - 1 - i) * scale + scale // 2)

        for x, y in robot_path:
            cv2.circle(img, pixel(x, y), 3, (0, 0, 255), -1)
        return img


def save(img, name, title, show):
    path = os.path.join(HERE, name)
    cv2.imwrite(path, img)
    print(f'  saved {path}')
    if show:
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def summary(grid):
    filled = int((grid.L > 0).sum())
    empty = int((grid.L < 0).sum())
    print(f'  cells: {filled} filled, {empty} empty, '
          f'{GRID_SIZE * GRID_SIZE - filled - empty} unknown')


def from_file(path, show):
    import yaml
    with open(path) as f:
        scan = next(yaml.safe_load_all(f))
    if any(isinstance(d, str) for d in scan['ranges']):
        sys.exit("The ranges list was cut off ('...'). "
                 'Save the scan again with --full-length.')
    grid = OccupancyGrid()
    # the lidar is at the origin, turned 90 degrees from the robot's forward axis
    used = grid.add_scan(scan, pose_matrix(0.0, 0.0, LIDAR_IN_BASE[2]))
    print(f'Scan from {path}: {used} of {len(scan["ranges"])} rays used')
    summary(grid)
    save(grid.image([(0.0, 0.0)]), 'grid_from_file.png', 'Occupancy grid (file)', show)


def live(mode, show):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformListener
    from irobot_create_msgs.action import DriveDistance

    class GridNode(Node):
        def __init__(self):
            super().__init__('occupancy_grid')
            self.scan = None
            self.odom = None
            self.create_subscription(LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
            self.create_subscription(Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.drive_client = ActionClient(self, DriveDistance, 'drive_distance')

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

        def lidar_in_base(self):
            """base_link -> rplidar_link from TF2 (static), with a fallback."""
            end = time.time() + 3.0
            while time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.1)
                try:
                    t = self.tf_buffer.lookup_transform('base_link', 'rplidar_link', Time())
                except Exception:
                    continue
                q = t.transform.rotation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                tr = t.transform.translation
                return pose_matrix(tr.x, tr.y, yaw)
            print('  (TF2 had no base_link -> rplidar_link, using the measured mount)')
            return pose_matrix(*LIDAR_IN_BASE)

    def odom_T_base(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return pose_matrix(p.x, p.y, yaw)

    rclpy.init()
    node = GridNode()
    base_T_lidar = node.lidar_in_base()
    start_odom = node.wait_for('odom')
    # world frame = the lidar's position right now, axes along the robot's heading
    odom_T_world = odom_T_base(start_odom) @ pose_matrix(base_T_lidar[0, 2], base_T_lidar[1, 2], 0.0)
    world_T_odom = np.linalg.inv(odom_T_world)

    def world_T_lidar():
        return world_T_odom @ odom_T_base(node.odom) @ base_T_lidar

    grid = OccupancyGrid()
    path = []

    def update(label):
        scan = node.wait_for('scan')
        pose = world_T_lidar()
        used = grid.add_scan(scan, pose)
        path.append((pose[0, 2], pose[1, 2]))
        print(f'{label}: lidar at x={pose[0, 2]:.3f} y={pose[1, 2]:.3f}, {used} rays used')

    update('Start')
    summary(grid)
    if mode == 'scan':
        save(grid.image(path), 'grid_scan.png', 'Occupancy grid', show)
    else:
        # not shown yet: a window here would hold up the drive until a key press
        save(grid.image(path), 'grid_0.0m.png', 'Occupancy grid at start', False)

        if not node.drive_client.wait_for_server(timeout_sec=5.0):
            sys.exit('No /drive_distance action server.')
        goal = DriveDistance.Goal()
        goal.distance = DRIVE_DISTANCE
        goal.max_translation_speed = DRIVE_SPEED
        send_future = node.drive_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send_future)
        handle = send_future.result()
        if not handle.accepted:
            sys.exit('DriveDistance goal was rejected.')
        result_future = handle.get_result_async()
        print(f'Driving {DRIVE_DISTANCE} m at {DRIVE_SPEED} m/s...')

        start = odom_T_base(start_odom)[:2, 2]
        next_mark = STEP
        while not result_future.done():
            rclpy.spin_once(node, timeout_sec=0.05)
            travelled = float(np.linalg.norm(odom_T_base(node.odom)[:2, 2] - start))
            # update at 0.1, 0.2, ... 0.9 m; the 1.0 m update is made once stopped
            if travelled >= next_mark and next_mark < DRIVE_DISTANCE - 1e-6:
                update(f'{next_mark:.1f} m (travelled {travelled:.3f})')
                if abs(next_mark - 0.5) < 1e-6:
                    summary(grid)
                    save(grid.image(path), 'grid_0.5m.png', 'Occupancy grid at 0.5 m', False)
                next_mark += STEP
        time.sleep(0.5)  # let the robot come to rest
        node.wait_for('odom')
        travelled = float(np.linalg.norm(odom_T_base(node.odom)[:2, 2] - start))
        update(f'1.0 m (travelled {travelled:.3f})')
        summary(grid)
        save(grid.image(path), 'grid_1.0m.png', 'Occupancy grid at 1.0 m', False)
        if show:
            for name in ('grid_0.0m.png', 'grid_0.5m.png', 'grid_1.0m.png'):
                cv2.imshow(name, cv2.imread(os.path.join(HERE, name)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


def main():
    args = [a for a in sys.argv[1:] if a != '--show']
    show = '--show' in sys.argv
    mode = args[0] if args else 'file'
    if mode == 'file':
        from_file(args[1] if len(args) > 1 else os.path.join(HERE, 'scan_sample.txt'), show)
    elif mode in ('scan', 'drive'):
        live(mode, show)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
