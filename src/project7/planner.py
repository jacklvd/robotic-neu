"""Task 3: A* path planning on an occupancy grid.

The robot is treated as a point by growing every filled cell by the robot's
radius (plus a margin): if the centre stays out of the grown cells, the whole
robot stays clear. Cells never seen as empty are blocked too, so the path
only crosses floor the lidar has seen.

A* moves to any of the 8 neighbouring cells, a diagonal step costing sqrt(2)
times a straight one, and never cuts the corner of a blocked cell. The
heuristic is the straight-line distance to the goal, which never
overestimates, so the path found is the shortest.
"""
import heapq
import math

import numpy as np

from grid import CELL, ORIGIN, to_cell
from landmarks import OCCUPIED_L

ROBOT_RADIUS = 0.17  # metres: a Turtlebot 4 is about 34 cm across
MARGIN = 0.03  # metres of extra clearance
MIN_GOAL_DISTANCE = 2.0  # metres from the start


def blocked_cells(grid, radius=ROBOT_RADIUS + MARGIN):
    """Bool array: True where the robot's centre may not go."""
    r = radius / CELL
    n = math.floor(r + 1e-9)
    di, dj = np.mgrid[-n:n + 1, -n:n + 1]
    disk = di * di + dj * dj <= r * r + 1e-9
    filled = grid.L > OCCUPIED_L
    grown = np.zeros_like(filled)
    for i, j in zip(*np.nonzero(disk)):
        # shift the filled mask by (i - n, j - n) and OR it in
        si, sj = i - n, j - n
        src = filled[max(-si, 0):filled.shape[0] - max(si, 0), max(-sj, 0):filled.shape[1] - max(sj, 0)]
        grown[max(si, 0):filled.shape[0] - max(-si, 0), max(sj, 0):filled.shape[1] - max(-sj, 0)] |= src
    return grown | ~(grid.L < 0)


STEPS = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]


def astar(blocked, start, goal):
    """Shortest list of cells from start to goal (both included), or None."""
    def h(c):
        return math.hypot(c[0] - goal[0], c[1] - goal[1])

    rows, cols = blocked.shape
    g = {start: 0.0}
    came_from = {}
    done = set()
    frontier = [(h(start), 0.0, start)]
    while frontier:
        _, cost, cell = heapq.heappop(frontier)
        if cell in done:
            continue
        if cell == goal:
            path = [cell]
            while cell in came_from:
                cell = came_from[cell]
                path.append(cell)
            return path[::-1]
        done.add(cell)
        i, j = cell
        for di, dj, step in STEPS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < rows and 0 <= nj < cols) or blocked[ni, nj]:
                continue
            if di and dj and (blocked[i + di, j] or blocked[i, j + dj]):
                continue  # would cut a blocked cell's corner
            new = cost + step
            if new < g.get((ni, nj), math.inf):
                g[(ni, nj)] = new
                came_from[(ni, nj)] = cell
                heapq.heappush(frontier, (new + h((ni, nj)), new, (ni, nj)))
    return None


def path_length(cells):
    """Length of a cell path, in cells."""
    return sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(cells, cells[1:]))


def cell_centre(i, j):
    return ((i + 0.5 - ORIGIN) * CELL, (j + 0.5 - ORIGIN) * CELL)


def max_deviation(path, a, b):
    """Furthest any point of path is from the straight line a -> b, in metres."""
    (ax, ay), (bx, by) = a, b
    d = math.hypot(bx - ax, by - ay)
    return max(abs((bx - ax) * (ay - y) - (ax - x) * (by - ay)) / d for x, y in path)


def plan(grid, start, goal, radius=ROBOT_RADIUS + MARGIN):
    """Shortest safe path from start to goal (metres, grid frame), as a list
    of cell centres in metres. Raises ValueError if there is none."""
    if math.dist(start, goal) < MIN_GOAL_DISTANCE:
        raise ValueError(f'the goal must be at least {MIN_GOAL_DISTANCE:.0f} m from the start')
    blocked = blocked_cells(grid, radius)
    s, t = to_cell(*start), to_cell(*goal)
    if blocked[s]:
        raise ValueError('the start is too close to an obstacle (or was never seen)')
    if blocked[t]:
        raise ValueError('the goal is too close to an obstacle (or was never seen as empty)')
    cells = astar(blocked, s, t)
    if cells is None:
        raise ValueError('no path to the goal keeps the robot clear of obstacles')
    return [cell_centre(i, j) for i, j in cells]
