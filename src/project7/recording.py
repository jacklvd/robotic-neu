"""One leg of a run: every scan and odometry pose, saved to a JSON file.

Each leg has its own grid frame {G}: the origin is the lidar's position at
the start of the leg and the x axis is the robot's heading then. So the
return leg's grid is in a different frame from the outbound one, and Task 2
finds the transform between them from the landmarks. The odometry pose at
the start of each leg is saved too, to check that transform against.
"""
import json

import numpy as np

from grid import OccupancyGrid, pose_matrix

SCAN_FIELDS = ('angle_min', 'angle_increment', 'range_min', 'range_max')


class Leg:
    def __init__(self, start_odom, lidar_in_base):
        """start_odom: robot pose (x, y, yaw) in /odom at the start of the leg.
        lidar_in_base: lidar pose (x, y, yaw) on the robot (base_link).
        """
        self.start_odom = tuple(start_odom)
        self.lidar_in_base = tuple(lidar_in_base)
        self.updates = []
        x, y, _ = self.lidar_in_base
        odom_T_grid = pose_matrix(*self.start_odom) @ pose_matrix(x, y, 0.0)
        self._grid_T_odom = np.linalg.inv(odom_T_grid)
        self._base_T_lidar = pose_matrix(*self.lidar_in_base)

    def world_T_lidar(self, odom):
        """Lidar pose in this leg's grid frame, for robot pose odom in /odom."""
        return self._grid_T_odom @ pose_matrix(*odom) @ self._base_T_lidar

    def record(self, label, travelled, odom, scan):
        """Keep one update: the robot's /odom pose and the scan taken there."""
        get = scan.get if isinstance(scan, dict) else lambda k: getattr(scan, k)
        self.updates.append({
            'label': label,
            'travelled': float(travelled),
            'odom': tuple(float(v) for v in odom),
            'scan': {**{k: float(get(k)) for k in SCAN_FIELDS},
                     'ranges': [float(d) for d in get('ranges')]},
        })


def save_leg(leg, path):
    # json writes inf / nan ranges as Infinity / NaN and reads them back
    with open(path, 'w') as f:
        json.dump({'start_odom': leg.start_odom,
                   'lidar_in_base': leg.lidar_in_base,
                   'updates': leg.updates}, f)


def load_leg(path):
    with open(path) as f:
        d = json.load(f)
    leg = Leg(d['start_odom'], d['lidar_in_base'])
    for u in d['updates']:
        leg.record(u['label'], u['travelled'], u['odom'], u['scan'])
    return leg


def grid_from_leg(leg):
    """Rebuild a leg's grid from its saved updates, in the order they were made.

    Returns the grid and the lidar's (x, y) at each update.
    """
    grid = OccupancyGrid()
    path = []
    for u in leg.updates:
        pose = leg.world_T_lidar(u['odom'])
        grid.add_scan(u['scan'], pose)
        path.append((pose[0, 2], pose[1, 2]))
    return grid, path
