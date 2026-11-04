"""Task 3: an A* path from the robot's start to a goal, on the outbound grid.

The start is where the robot's centre was at the start of the outbound leg.
The goal defaults to GOAL, a point behind the box ahead and to the left in
our classroom run, so the straight line to it is blocked and the path has to
bend round. Obstacles are grown by the robot's radius plus a margin (see
planner.py).

Saves in the run folder:
  task3_path.png   the grid with the grown obstacles, the path, and the
                   blocked straight line
  task3_path.txt   the path's points (x, y in metres, outbound grid frame)

Usage:  python3 task3.py [run folder] [--goal X Y]
        (default: the newest run in runs/, goal GOAL)
Add --show to also open the image in a window (press any key to close it).
"""
import os
import sys

import cv2

from grid import bresenham, to_cell, to_pixel
from landmarks import OCCUPIED_L
from map_run import latest_run, show_images
from planner import MARGIN, ROBOT_RADIUS, blocked_cells, max_deviation, plan
from recording import grid_from_leg, load_leg
from task2 import legend, mask_image

GOAL = (3.0, 1.75)  # metres, outbound grid frame
GROWN = (190, 190, 255)  # BGR: light red, too close to an obstacle for the robot's centre
PATH = (220, 60, 0)  # blue
STRAIGHT = (0, 140, 255)  # orange
START = (0, 150, 0)  # green
END = (0, 0, 230)  # red


def dashed_line(img, a, b, colour, dash=8):
    (ax, ay), (bx, by) = a, b
    n = max(1, int(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 // dash))
    for k in range(0, n, 2):
        p = (round(ax + (bx - ax) * k / n), round(ay + (by - ay) * k / n))
        q = (round(ax + (bx - ax) * (k + 1) / n), round(ay + (by - ay) * (k + 1) / n))
        cv2.line(img, p, q, colour, 2, cv2.LINE_AA)


def main():
    args = sys.argv[1:]
    show = '--show' in args
    goal = GOAL
    if '--goal' in args:
        k = args.index('--goal')
        goal = (float(args[k + 1]), float(args[k + 2]))
        del args[k:k + 3]
    args = [a for a in args if a != '--show']
    folder = args[0] if args else latest_run()

    leg = load_leg(os.path.join(folder, 'leg_out.json'))
    grid, _ = grid_from_leg(leg)
    # the grid's origin is the lidar; the robot's centre is offset from it
    start = (0.0 - leg.lidar_in_base[0], 0.0 - leg.lidar_in_base[1])  # 0.0 - avoids printing -0.000
    try:
        path = plan(grid, start, goal)
    except ValueError as e:
        sys.exit(f'Cannot plan: {e}.')

    blocked = blocked_cells(grid)
    crossed = sum(blocked[c] for c in bresenham(*to_cell(*start), *to_cell(*goal)))
    length = sum(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 for (x0, y0), (x1, y1) in zip(path, path[1:]))
    straight = ((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2) ** 0.5
    report = '\n'.join([
        f'Task 3: {folder}',
        f'  start ({start[0]:.3f}, {start[1]:.3f}) m, goal ({goal[0]:.3f}, {goal[1]:.3f}) m '
        f'(outbound grid frame), {straight:.2f} m apart',
        f'  obstacles grown by {100 * (ROBOT_RADIUS + MARGIN):.0f} cm '
        f'(robot radius {100 * ROBOT_RADIUS:.0f} cm + {100 * MARGIN:.0f} cm margin)',
        f'  straight line: crosses {crossed} cells too close to an obstacle',
        f'  A* path: {len(path)} cells, {length:.2f} m long, '
        f'bends up to {100 * max_deviation(path, start, goal):.0f} cm from the straight line',
    ])
    print(report)
    with open(os.path.join(folder, 'task3_path.txt'), 'w') as f:
        f.write('# ' + report.replace('\n', '\n# ') + '\n# x (m)  y (m)\n')
        f.writelines(f'{x:.3f} {y:.3f}\n' for x, y in path)

    img = grid.image()
    grown = mask_image(blocked & (grid.L < 0) & ~(grid.L > OCCUPIED_L))
    img[grown] = GROWN
    dashed_line(img, to_pixel(*start), to_pixel(*goal), STRAIGHT)
    for a, b in zip(path, path[1:]):
        cv2.line(img, to_pixel(*a), to_pixel(*b), PATH, 3, cv2.LINE_AA)
    cv2.circle(img, to_pixel(*start), 7, START, -1)
    cv2.circle(img, to_pixel(*goal), 7, END, -1)
    legend(img, [(START, 'start'), (END, f'goal ({goal[0]:.2f}, {goal[1]:.2f}) m'),
                 (PATH, f'A* path, {length:.2f} m'), (STRAIGHT, 'straight line (blocked)'),
                 (GROWN, 'too close for the robot\'s centre')])
    name = os.path.join(folder, 'task3_path.png')
    cv2.imwrite(name, img)
    print(f'  saved {name}')
    if show:
        show_images([name])


if __name__ == '__main__':
    main()
