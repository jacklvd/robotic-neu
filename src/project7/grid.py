"""Occupancy grid, carried over from Project 6 with a bigger grid.

The grid is 320 x 320 cells of 5 cm, so it covers 16 m x 16 m: room for a
classroom's walls on every side after a 3 m drive from the centre. Each cell
holds a log-odds value L, starting at 0 (unknown). For each lidar ray,
Bresenham ray casting from the sensor to the hit point subtracts 0.95 from
every cell the ray passes through (empty) and adds 0.95 to the cell it hit
(filled).
"""
import math

import cv2
import numpy as np

GRID_SIZE = 320  # cells per side
CELL = 0.05  # metres per cell
ORIGIN = GRID_SIZE // 2  # grid index of the grid frame's origin
L_FILLED = math.log10(0.9 / 0.1)  # +0.95, cell the ray hit
L_EMPTY = math.log10(0.1 / 0.9)  # -0.95, cells the ray passed through


def pose_matrix(x, y, yaw):
    """3x3 homogeneous transform for a 2D pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


def to_cell(x, y):
    """Point in the grid frame (metres) to grid indices (i along x, j along y)."""
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

    def add_scan(self, scan, grid_T_sensor):
        """Ray-cast one LaserScan (a dict or a ROS message) into the grid.

        grid_T_sensor is the 3x3 pose of the lidar in the grid frame.
        """
        get = scan.get if isinstance(scan, dict) else lambda k: getattr(scan, k)
        angle_min, step = get('angle_min'), get('angle_increment')
        range_min, range_max = get('range_min'), get('range_max')
        x0, y0 = grid_T_sensor[0, 2], grid_T_sensor[1, 2]
        i0, j0 = to_cell(x0, y0)
        used = 0
        for k, d in enumerate(get('ranges')):
            # values outside [range_min, range_max] (or inf / nan) are discarded
            if not (range_min <= d <= range_max):
                continue
            theta = angle_min + k * step
            hit = grid_T_sensor @ np.array([d * math.cos(theta), d * math.sin(theta), 1.0])
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

        Up in the picture is the grid frame's +x (the robot's heading at the
        start of the leg), left is +y. Red dots are where the lidar was for
        each update.
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

    def summary(self):
        filled = int((self.L > 0).sum())
        empty = int((self.L < 0).sum())
        return (f'{filled} filled, {empty} empty, '
                f'{GRID_SIZE * GRID_SIZE - filled - empty} unknown')
