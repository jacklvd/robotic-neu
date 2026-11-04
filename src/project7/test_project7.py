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


def test_last_label_reports_the_distance_actually_travelled():
    from map_run import stopped_label
    assert stopped_label(0.015) == '0.015 m (stopped)'
    assert stopped_label(3.0034) == '3.003 m (stopped)'


# ---- Task 2: landmarks -------------------------------------------------------

def put_blob(grid, x, y, size=2, L=20.0):
    """Mark a size x size block of filled cells with its corner cell at (x, y)."""
    i, j = to_cell(x, y)
    grid.L[i:i + size, j:j + size] = L


def test_small_strong_isolated_blob_is_a_landmark_at_its_centre():
    from landmarks import find_landmarks
    grid = OccupancyGrid()
    put_blob(grid, 1.0, 0.5)  # cells 1.00-1.10 x 0.50-0.60: centre (1.05, 0.55)
    found = find_landmarks(grid)
    assert len(found) == 1
    assert found[0][:2] == pytest.approx((1.05, 0.55), abs=1e-6)


def test_centre_is_weighted_by_how_filled_each_cell_is():
    from landmarks import find_landmarks
    grid = OccupancyGrid()
    i, j = to_cell(1.0, 0.0)
    grid.L[i, j] = 30.0
    grid.L[i + 1, j] = 10.0  # weaker cell 5 cm further along x
    (x, y, _), = find_landmarks(grid)
    assert x == pytest.approx(1.025 + 0.05 * 10 / 40)
    assert y == pytest.approx(0.025)


def test_walls_weak_blobs_and_blobs_beside_walls_are_not_landmarks():
    from landmarks import find_landmarks
    grid = OccupancyGrid()
    i, j = to_cell(2.0, -1.0)
    grid.L[i, j:j + 40] = 20.0  # a 2 m wall
    put_blob(grid, -1.0, 1.0, L=1.5)  # seen by too few scans
    put_blob(grid, 2.1, 0.0)  # 5 cm from the wall, not touching it
    assert find_landmarks(grid) == []


def test_rigid_fit_recovers_a_known_rotation_and_shift():
    from landmarks import apply, rigid_fit
    q = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.5, -0.5]])
    T = pose_matrix(3.1, -0.2, math.radians(178.0))
    np.testing.assert_allclose(rigid_fit(apply(T, q), q), T, atol=1e-9)


def test_ransac_matches_landmarks_without_being_told_the_transform():
    from landmarks import apply, ransac_align
    rng = np.random.default_rng(1)
    a = rng.uniform(-3, 3, size=(9, 2))
    T = pose_matrix(3.08, 0.05, math.radians(-177.9))  # back grid -> out grid
    # the back grid sees landmarks 0-6 (with 1 cm noise) plus two of its own,
    # listed in a different order
    seen = apply(np.linalg.inv(T), a[:7]) + rng.normal(0, 0.01, size=(7, 2))
    b = np.vstack([seen[::-1], [[5.0, 5.0], [-4.0, 2.0]]])
    T_est, pairs = ransac_align(a, b)
    assert sorted(pairs) == sorted((6 - k, k) for k in range(7))
    assert T_est[:2, 2] == pytest.approx(T[:2, 2], abs=0.02)
    assert math.degrees(math.atan2(T_est[1, 0], T_est[0, 0])) == pytest.approx(-177.9, abs=0.5)


def test_ransac_needs_two_landmarks_in_each_grid():
    from landmarks import ransac_align
    with pytest.raises(ValueError):
        ransac_align(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_odometry_transform_between_the_two_legs():
    from recording import grid1_T_grid2
    out = Leg(start_odom=(1.0, 2.0, math.pi / 2), lidar_in_base=LIDAR)
    back = Leg(start_odom=(2.0, 5.0, 0.0), lidar_in_base=LIDAR)
    # grid origins (lidar positions) in odom: (1, 1.96) and (1.96, 5), so the
    # back origin is 3.04 m along the out grid's x (odom +y), 0.96 m to its right
    np.testing.assert_allclose(grid1_T_grid2(out, back),
                               pose_matrix(3.04, -0.96, -math.pi / 2), atol=1e-9)


def test_back_grid_moved_into_the_out_grid_frame():
    from landmarks import grid_in_frame
    back = OccupancyGrid()
    put_blob(back, 1.0, 0.5, size=1)  # cell centre (1.025, 0.525) in the back frame
    T = pose_matrix(3.0, 0.5, math.pi / 2)
    L = grid_in_frame(back.L, T)
    # (1.025, 0.525) -> (3 - 0.525, 0.5 + 1.025) = (2.475, 1.525) in the out frame
    assert L[to_cell(2.475, 1.525)] == 20.0
    assert (L > 0).sum() == 1


def test_pixel_puts_the_grid_origin_in_the_middle_and_x_up():
    from grid import to_pixel
    mid = (GRID_SIZE - 1 - ORIGIN) * 3 + 1
    assert to_pixel(0.0, 0.0) == (mid, mid)
    assert to_pixel(1.0, 0.0) == (mid, mid - 60)  # 1 m ahead: 20 cells up
    assert to_pixel(0.0, 1.0) == (mid - 60, mid)  # 1 m left


def test_compare_lists_matches_unmatched_and_mean_difference():
    from task2 import compare
    a = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 5.0]])
    b = np.array([[1.0, 0.03], [9.0, 9.0], [2.0, 0.0]])  # already in a's frame
    shift = pose_matrix(0.0, 0.01, 0.0)  # "odometry" is 1 cm off in y
    c = compare(a, b, np.eye(3), [(0, 0), (1, 2)], shift)
    assert [(m['a'], m['b']) for m in c['matched']] == [(0, 0), (1, 2)]
    assert c['only_a'] == [2] and c['only_b'] == [1]
    assert c['matched'][0]['diff'] == pytest.approx(0.03)
    assert c['mean'] == pytest.approx(0.015)
    assert c['mean_odom'] == pytest.approx((0.04 + 0.01) / 2)
    assert c['b_in_a'] == pytest.approx(b)


# ---- Task 3: A* ---------------------------------------------------------------

def open_floor():
    """A grid whose every cell has been seen empty."""
    grid = OccupancyGrid()
    grid.L[:] = -5.0
    return grid


def test_obstacles_grow_by_the_robot_radius_and_unknown_floor_is_blocked():
    from planner import blocked_cells
    grid = open_floor()
    i, j = to_cell(1.0, 0.0)
    grid.L[i, j] = 20.0
    grid.L[to_cell(-2.0, -2.0)] = 0.0  # never seen
    blocked = blocked_cells(grid, radius=0.20)  # 4 cells
    assert blocked[i + 4, j] and blocked[i, j - 4]
    assert not blocked[i + 5, j] and not blocked[i + 3, j + 3]  # 4.2 cells away
    assert blocked[to_cell(-2.0, -2.0)]
    assert not blocked[to_cell(0.0, 0.0)]


def test_astar_goes_straight_on_open_floor():
    from planner import astar
    blocked = np.zeros((20, 20), bool)
    assert astar(blocked, (2, 5), (12, 5)) == [(i, 5) for i in range(2, 13)]


def test_astar_takes_diagonal_steps_at_root_two_cost():
    from planner import astar, path_length
    blocked = np.zeros((20, 20), bool)
    path = astar(blocked, (0, 0), (3, 3))
    assert path == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert path_length(path) == pytest.approx(3 * math.sqrt(2))


def test_astar_bends_round_a_wall_by_the_shortest_way():
    from planner import astar, path_length
    blocked = np.zeros((30, 30), bool)
    blocked[10, :20] = True  # wall from the edge across the straight line
    path = astar(blocked, (5, 10), (15, 10))
    assert not any(blocked[c] for c in path)
    assert path[0] == (5, 10) and path[-1] == (15, 10)
    # shortest: 4 diagonal + 6 straight steps to (9, 20), straight through the
    # gap to (11, 20) (no cutting the wall's corner), and the same back
    assert path_length(path) == pytest.approx(14 + 8 * math.sqrt(2))


def test_astar_does_not_squeeze_diagonally_between_two_blocked_cells():
    from planner import astar
    blocked = np.ones((5, 5), bool)
    blocked[1, 1] = blocked[2, 2] = False  # only a corner-to-corner gap
    assert astar(blocked, (1, 1), (2, 2)) is None


def test_plan_rejects_bad_goals():
    from planner import plan
    grid = open_floor()
    with pytest.raises(ValueError, match='at least 2'):
        plan(grid, (0.0, 0.0), (1.5, 0.0))
    grid.L[to_cell(3.0, 0.0)] = 20.0
    with pytest.raises(ValueError, match='goal'):
        plan(grid, (0.0, 0.0), (3.0, 0.0))


def test_plan_returns_a_path_in_metres_that_avoids_an_obstacle():
    from planner import blocked_cells, max_deviation, plan
    grid = open_floor()
    i, j = to_cell(1.5, 0.0)
    grid.L[i - 2:i + 3, j - 6:j + 7] = 20.0  # a 0.25 m x 0.65 m box in the way
    path = plan(grid, (0.0, 0.0), (3.0, 0.0))
    assert path[0] == pytest.approx((0.025, 0.025)) and path[-1] == pytest.approx((3.025, 0.025))
    blocked = blocked_cells(grid)
    assert not any(blocked[to_cell(x, y)] for x, y in path)
    assert max_deviation(path, path[0], path[-1]) > 0.5  # round the box: 0.325 + 0.20 m


def dijkstra_length(blocked, start, goal):
    """Shortest path length with the same moves as A*, by plain Dijkstra."""
    import heapq
    best = {start: 0.0}
    frontier = [(0.0, start)]
    while frontier:
        d, (i, j) = heapq.heappop(frontier)
        if (i, j) == goal:
            return d
        if d > best[(i, j)]:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (di == dj == 0 or not (0 <= ni < blocked.shape[0] and 0 <= nj < blocked.shape[1])
                        or blocked[ni, nj] or (di and dj and (blocked[i + di, j] or blocked[i, j + dj]))):
                    continue
                nd = d + math.hypot(di, dj)
                if nd < best.get((ni, nj), math.inf):
                    best[(ni, nj)] = nd
                    heapq.heappush(frontier, (nd, (ni, nj)))
    return None


def test_astar_finds_the_shortest_path_through_clutter():
    from planner import astar, path_length
    rng = np.random.default_rng(0)
    for _ in range(20):
        blocked = rng.random((30, 30)) < 0.3
        blocked[0, 0] = blocked[29, 29] = False
        path = astar(blocked, (0, 0), (29, 29))
        best = dijkstra_length(blocked, (0, 0), (29, 29))
        if best is None:
            assert path is None
        else:
            assert path_length(path) == pytest.approx(best)
