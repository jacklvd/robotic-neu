"""Tests for the parts of Project 7 that run without the robot.

Run from this folder with:  python3 -m pytest test_project7.py
"""
import math

import numpy as np
import pytest

from grid import CELL, GRID_SIZE, ORIGIN, OccupancyGrid, bresenham, pose_matrix, to_cell
from recording import Leg, grid_from_leg, load_leg, save_leg

LIDAR = (-0.04, 0.0, math.pi / 2)  # lidar mount on our Turtlebot 4


def room_scan(x, y, yaw, half=3.0, n=360):
    """A LaserScan dict seen from a lidar at (x, y, yaw) inside a square room.

    The room's walls are at +-half metres in the world frame. Rays are cast
    analytically, so the true wall cells are known.
    """
    ranges = []
    for k in range(n):
        a = yaw + 2 * math.pi * k / n
        c, s = math.cos(a), math.sin(a)
        ts = []
        if c > 1e-9:
            ts.append((half - x) / c)
        if c < -1e-9:
            ts.append((-half - x) / c)
        if s > 1e-9:
            ts.append((half - y) / s)
        if s < -1e-9:
            ts.append((-half - y) / s)
        ranges.append(min(ts))
    return {'angle_min': 0.0, 'angle_increment': 2 * math.pi / n,
            'range_min': 0.15, 'range_max': 12.0, 'ranges': ranges}


def test_grid_covers_16_m_with_5_cm_cells():
    assert CELL == 0.05
    assert GRID_SIZE * CELL == pytest.approx(16.0)
    assert to_cell(0.0, 0.0) == (ORIGIN, ORIGIN)
    assert to_cell(7.9, -7.9) == (GRID_SIZE - 2, 2)


def test_bresenham_stops_before_the_end_cell():
    assert bresenham(0, 0, 3, 1) == [(0, 0), (1, 0), (2, 1)]
    assert bresenham(5, 5, 5, 2) == [(5, 5), (5, 4), (5, 3)]


def test_scan_marks_walls_filled_and_the_floor_empty():
    grid = OccupancyGrid()
    grid.add_scan(room_scan(0.0, 0.0, 0.0), pose_matrix(0.0, 0.0, 0.0))
    i_wall, j_mid = to_cell(3.0, 0.0)
    assert grid.L[i_wall, j_mid] > 0
    assert grid.L[to_cell(1.5, 0.0)] < 0
    assert grid.L[to_cell(5.0, 0.0)] == 0  # behind the wall: never seen


def test_leg_frame_is_the_lidar_position_at_the_start_of_the_leg():
    # robot starts somewhere arbitrary in the odom frame, facing +y
    leg = Leg(start_odom=(1.0, 2.0, math.pi / 2), lidar_in_base=LIDAR)
    start = leg.world_T_lidar((1.0, 2.0, math.pi / 2))
    assert start[:2, 2] == pytest.approx([0.0, 0.0], abs=1e-9)
    # driving 1 m forward (+y in odom) is +1 m along the leg's x axis
    moved = leg.world_T_lidar((1.0, 3.0, math.pi / 2))
    assert moved[:2, 2] == pytest.approx([1.0, 0.0], abs=1e-9)


def test_leg_after_a_turn_has_its_own_frame():
    # the return leg starts 3 m out, turned around: its x axis points home
    leg = Leg(start_odom=(3.0, 0.0, math.pi), lidar_in_base=LIDAR)
    moved = leg.world_T_lidar((2.0, 0.0, math.pi))
    assert moved[:2, 2] == pytest.approx([1.0, 0.0], abs=1e-9)


def test_saved_leg_loads_back_unchanged(tmp_path):
    leg = Leg(start_odom=(0.5, -0.2, 0.3), lidar_in_base=LIDAR)
    scan = room_scan(0.0, 0.0, 0.0)
    scan['ranges'][7] = math.inf  # out-of-range rays must survive the file
    scan['ranges'][8] = math.nan
    leg.record('Start', 0.0, (0.5, -0.2, 0.3), scan)
    path = tmp_path / 'leg.json'
    save_leg(leg, path)
    back = load_leg(path)
    assert back.start_odom == leg.start_odom
    assert back.lidar_in_base == leg.lidar_in_base
    got = back.updates[0]
    assert got['label'] == 'Start' and got['travelled'] == 0.0
    assert got['odom'] == (0.5, -0.2, 0.3)
    assert got['scan']['ranges'][7] == math.inf
    assert math.isnan(got['scan']['ranges'][8])
    assert got['scan']['ranges'][9] == scan['ranges'][9]


def test_replayed_grid_matches_the_live_one(tmp_path):
    """The grid rebuilt from a saved leg is identical to the one built live."""
    leg = Leg(start_odom=(0.0, 0.0, 0.0), lidar_in_base=LIDAR)
    live = OccupancyGrid()
    for k in range(11):  # 1 m in 0.1 m steps
        odom = (0.1 * k, 0.0, 0.0)
        pose = leg.world_T_lidar(odom)
        scan = room_scan(pose[0, 2], pose[1, 2], math.atan2(pose[1, 0], pose[0, 0]))
        leg.record(f'{0.1 * k:.1f} m', 0.1 * k, odom, scan)
        live.add_scan(scan, pose)
    save_leg(leg, tmp_path / 'leg.json')
    replay, path = grid_from_leg(load_leg(tmp_path / 'leg.json'))
    np.testing.assert_array_equal(replay.L, live.L)
    assert len(path) == 11
    assert path[-1] == pytest.approx((1.0, 0.0), abs=1e-9)


def test_replay_rebuilds_both_grid_images_of_a_saved_run(tmp_path):
    from map_run import LEGS, replay
    for name in LEGS:
        leg = Leg(start_odom=(0.0, 0.0, 0.0), lidar_in_base=LIDAR)
        pose = leg.world_T_lidar((0.0, 0.0, 0.0))
        leg.record('0.0 m', 0.0, (0.0, 0.0, 0.0), room_scan(pose[0, 2], pose[1, 2], math.pi / 2))
        save_leg(leg, tmp_path / f'leg_{name}.json')
    replay(str(tmp_path), show=False)
    assert sorted(p.name for p in tmp_path.glob('grid_*.png')) == ['grid_back.png', 'grid_out.png']
