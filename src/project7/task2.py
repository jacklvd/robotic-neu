"""Task 2: landmarks in the outbound and return grids of a run, compared.

Finds the landmarks in each grid (see landmarks.py), matches them with
RANSAC to get the transform from the return grid's frame to the outbound
grid's, and reports:
  - every return-grid landmark in the outbound grid's frame
  - the landmarks found in one grid but not the other
  - for those found in both, the mean difference in their position
  - the landmark transform next to odometry's, as a check

Saves in the run folder:
  task2_report.txt                          the report (also printed)
  landmarks_out.png, landmarks_back.png     each grid, landmarks circled
  overlay.png                               the return grid moved onto the outbound one

Usage:  python3 task2.py [run folder]   (default: the newest run in runs/)
Add --show to also open the images in a window (press any key to close it).
"""
import math
import os
import sys

import cv2
import numpy as np

from grid import yaw_of, to_pixel
from landmarks import (ISOLATION, LANDMARK_L, MAX_SIZE, OCCUPIED_L, apply,
                       find_landmarks, grid_in_frame, ransac_align)
from map_run import latest_run, show_images
from recording import grid1_T_grid2, grid_from_leg, load_leg

MATCHED = (0, 150, 0)  # BGR: green, found in both grids
ONLY_OUT = (0, 140, 255)  # orange, only in the outbound grid
ONLY_BACK = (200, 80, 0)  # blue, only in the return grid
BACK_FILLED = (0, 0, 230)  # red, return grid's filled cells in the overlay
BOTH_FILLED = (140, 0, 140)  # purple, filled in both grids


def compare(a, b, T, pairs, T_odom):
    """Everything the report needs, from landmarks a (outbound frame), b
    (return frame), the landmark transform T (b -> a), the matched pairs and
    odometry's transform T_odom (b -> a)."""
    b_in_a, b_odom = apply(T, b), apply(T_odom, b)
    matched = [{'a': ia, 'b': ib,
                'diff': float(np.linalg.norm(a[ia] - b_in_a[ib])),
                'diff_odom': float(np.linalg.norm(a[ia] - b_odom[ib]))}
               for ia, ib in sorted(pairs)]
    in_a = {m['a'] for m in matched}
    in_b = {m['b'] for m in matched}
    return {
        'b_in_a': b_in_a,
        'matched': matched,
        'only_a': [i for i in range(len(a)) if i not in in_a],
        'only_b': [i for i in range(len(b)) if i not in in_b],
        'mean': float(np.mean([m['diff'] for m in matched])) if matched else math.nan,
        'mean_odom': float(np.mean([m['diff_odom'] for m in matched])) if matched else math.nan,
    }


def labels(c):
    """Matched landmarks share a number; the rest are O1, O2... and B1, B2..."""
    la, lb = {}, {}
    for n, m in enumerate(c['matched'], 1):
        la[m['a']] = lb[m['b']] = str(n)
    for n, i in enumerate(c['only_a'], 1):
        la[i] = f'O{n}'
    for n, i in enumerate(c['only_b'], 1):
        lb[i] = f'B{n}'
    return la, lb


def mark(img, xy, text, colour):
    px = to_pixel(*xy)
    cv2.circle(img, px, 8, colour, 2)
    at = (px[0] + 9, px[1] - 7)
    # white outline first, so the label reads on top of the map
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def legend(img, rows):
    """rows: (colour, text). Drawn top left on a white box."""
    h = 20 * len(rows) + 10
    w = 12 + max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for _, t in rows) + 24
    cv2.rectangle(img, (5, 5), (5 + w, 5 + h), (255, 255, 255), -1)
    cv2.rectangle(img, (5, 5), (5 + w, 5 + h), (0, 0, 0), 1)
    for k, (colour, text) in enumerate(rows):
        y = 25 + 20 * k
        cv2.circle(img, (20, y - 4), 6, colour, -1)
        cv2.putText(img, text, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def mask_image(mask, scale=3):
    """A cell mask laid out like OccupancyGrid.image() (x up, y left)."""
    m = mask[::-1, ::-1].astype(np.uint8)
    return cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0


def fmt_T(T):
    return f'{T[0, 2]:7.3f} {T[1, 2]:7.3f} {math.degrees(yaw_of(T)):9.2f}'


def main():
    args = [a for a in sys.argv[1:] if a != '--show']
    show = '--show' in sys.argv
    folder = args[0] if args else latest_run()
    out, back = (load_leg(os.path.join(folder, f'leg_{n}.json')) for n in ('out', 'back'))
    g1, path1 = grid_from_leg(out)
    g2, path2 = grid_from_leg(back)
    a = np.array([lm[:2] for lm in find_landmarks(g1)]).reshape(-1, 2)
    b = np.array([lm[:2] for lm in find_landmarks(g2)]).reshape(-1, 2)
    T, pairs = ransac_align(a, b)
    T_odom = grid1_T_grid2(out, back)
    c = compare(a, b, T, pairs, T_odom)
    la, lb = labels(c)

    r = [f'Task 2: {folder}', '',
         f'Landmarks: {len(a)} in the outbound grid, {len(b)} in the return grid',
         f'(filled clusters no bigger than {100 * MAX_SIZE:.0f} cm, strongest cell L >= {LANDMARK_L:.0f},',
         f' no other filled cell within {100 * ISOLATION:.0f} cm)', '',
         'Return grid frame -> outbound grid frame',
         '                x (m)   y (m)  rotation (deg)',
         f'  landmarks  {fmt_T(T)}',
         f'  odometry   {fmt_T(T_odom)}',
         f'  difference {T[0, 2] - T_odom[0, 2]:7.3f} {T[1, 2] - T_odom[1, 2]:7.3f} '
         f'{math.degrees(yaw_of(np.linalg.inv(T_odom) @ T)):9.2f}', '',
         'Outbound grid landmarks (outbound frame)',
         '  label   x (m)   y (m)']
    r += [f'  {la[i]:>5} {x:7.3f} {y:7.3f}' for i, (x, y) in enumerate(a)]
    r += ['', 'Return grid landmarks: own frame -> outbound frame (landmark transform)',
          '  label   x (m)   y (m)  ->   x (m)   y (m)']
    r += [f'  {lb[i]:>5} {x:7.3f} {y:7.3f}  -> {c["b_in_a"][i][0]:7.3f} {c["b_in_a"][i][1]:7.3f}'
          for i, (x, y) in enumerate(b)]
    r += ['', f'Found in both grids: {len(c["matched"])}',
          '  label   outbound x, y   return x, y (moved)   difference (cm)']
    for m in c['matched']:
        (xa, ya), (xb, yb) = a[m['a']], c['b_in_a'][m['b']]
        r.append(f'  {la[m["a"]]:>5}  {xa:6.3f} {ya:6.3f}    {xb:6.3f} {yb:6.3f}        {100 * m["diff"]:5.1f}')
    r += [f'  mean difference: {100 * c["mean"]:.1f} cm with the landmark transform '
          f'({100 * c["mean_odom"]:.1f} cm with odometry\'s)', '',
          f'Only in the outbound grid: {len(c["only_a"])}']
    r += [f'  {la[i]:>5}  ({a[i][0]:.3f}, {a[i][1]:.3f})' for i in c['only_a']]
    r += ['', f'Only in the return grid: {len(c["only_b"])} (positions moved into the outbound frame)']
    r += [f'  {lb[i]:>5}  ({c["b_in_a"][i][0]:.3f}, {c["b_in_a"][i][1]:.3f})' for i in c['only_b']]
    report = '\n'.join(r)
    print(report)
    with open(os.path.join(folder, 'task2_report.txt'), 'w') as f:
        f.write(report + '\n')

    images = []

    def save(img, name):
        path = os.path.join(folder, name)
        cv2.imwrite(path, img)
        print(f'  saved {path}')
        images.append(path)

    img = g1.image(path1)
    for i, xy in enumerate(a):
        mark(img, xy, la[i], ONLY_OUT if i in c['only_a'] else MATCHED)
    legend(img, [(MATCHED, 'landmark in both grids'), (ONLY_OUT, 'only in this (outbound) grid'),
                 ((0, 0, 255), 'lidar position at each update')])
    save(img, 'landmarks_out.png')

    img = g2.image(path2)
    for i, xy in enumerate(b):
        mark(img, xy, lb[i], ONLY_BACK if i in c['only_b'] else MATCHED)
    legend(img, [(MATCHED, 'landmark in both grids'), (ONLY_BACK, 'only in this (return) grid'),
                 ((0, 0, 255), 'lidar position at each update')])
    save(img, 'landmarks_back.png')

    img = g1.image()
    filled1 = mask_image(g1.L > OCCUPIED_L)
    filled2 = mask_image(grid_in_frame(g2.L, T) > OCCUPIED_L)
    img[filled2 & ~filled1] = BACK_FILLED
    img[filled2 & filled1] = BOTH_FILLED
    for i, xy in enumerate(a):
        mark(img, xy, la[i], ONLY_OUT if i in c['only_a'] else MATCHED)
    for i in c['only_b']:
        mark(img, c['b_in_a'][i], lb[i], ONLY_BACK)
    legend(img, [((0, 0, 0), 'filled in the outbound grid'), (BACK_FILLED, 'filled in the return grid, moved'),
                 (BOTH_FILLED, 'filled in both'), (MATCHED, 'landmark in both grids'),
                 (ONLY_OUT, 'only in the outbound grid'), (ONLY_BACK, 'only in the return grid')])
    save(img, 'overlay.png')
    if show:
        show_images(images)


if __name__ == '__main__':
    main()
