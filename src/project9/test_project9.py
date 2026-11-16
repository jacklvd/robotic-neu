"""Tests for the parts of Project 9 that run without the robot.

Run from this folder with:  python3 -m pytest test_project9.py
"""
import math
from types import SimpleNamespace

import numpy as np

from cones import (COLORS, approach_distance, find_cone, image_to_bgr, nearest_return,
                   pixel_bearing, to_odom)

LIDAR = (-0.04, 0.0, math.pi / 2)


def test_find_cone_picks_the_right_colour_and_column():
    """A coloured square on grey is found at its column, other colours are not."""
    img = np.full((240, 320, 3), 128, np.uint8)
    img[100:140, 240:280] = (0, 128, 255)  # orange in BGR, centred at x = 260
    cx, area = find_cone(img, 'orange')
    assert abs(cx - 260 / 320) < 0.01 and area == 40 * 40
    assert all(find_cone(img, c) is None for c in COLORS if c != 'orange')


def test_pixel_bearing_signs():
    """Left of centre is a positive (left) angle, centre is 0."""
    assert pixel_bearing(0.5) == 0
    assert pixel_bearing(0.25) > 0 > pixel_bearing(0.75)


def test_nearest_return_finds_a_cone_in_front_of_a_wall():
    """A cone 1 m ahead (base frame) is picked over a wall 3 m away, through the lidar's 90 degree mount."""
    n = 360
    ranges = [3.0] * n
    # straight ahead in base_link is -90 degrees in the lidar frame
    k = round((-math.pi / 2 - (-math.pi)) / (2 * math.pi / n))
    ranges[k] = 1.04  # 1 m from the centre = 1.04 m from the lidar, 4 cm behind
    scan = SimpleNamespace(ranges=ranges, angle_min=-math.pi, angle_increment=2 * math.pi / n,
                           range_min=0.15, range_max=12.0)
    x, y = nearest_return(scan, 0.0, lidar=LIDAR)
    assert abs(x - 1.0) < 1e-6 and abs(y) < 1e-6
    # nothing but wall to the left
    assert math.hypot(*nearest_return(scan, math.pi / 2, lidar=LIDAR)) > 2.9


def test_approach_and_to_odom():
    """The drive leaves STOP_GAP; base points rotate into odom correctly."""
    import cones
    d = approach_distance((1.0, 0.0))
    assert abs(d + cones.ROBOT_RADIUS + cones.CONE_BASE_OVER_LIDAR + cones.STOP_GAP - 1.0) < 1e-9
    x, y = to_odom((1.0, 2.0, math.pi / 2), (1.0, 0.0))
    assert abs(x - 1.0) < 1e-9 and abs(y - 3.0) < 1e-9


def test_image_to_bgr_handles_rgb_and_row_padding():
    """rgb8 is flipped to BGR and padding at the end of each row is dropped."""
    w, h, step = 2, 2, 8  # 6 bytes of pixels + 2 bytes of padding per row
    rows = [bytes([255, 0, 0, 0, 255, 0, 9, 9])] * h  # red, green in RGB
    msg = SimpleNamespace(width=w, height=h, step=step, encoding='rgb8', data=b''.join(rows))
    img = image_to_bgr(msg)
    assert img.shape == (2, 2, 3)
    assert tuple(img[0, 0]) == (0, 0, 255) and tuple(img[0, 1]) == (0, 255, 0)
