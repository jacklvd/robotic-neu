"""Task 2: landmarks in an occupancy grid, and matching them between grids.

A landmark is a small, strongly filled, isolated cluster of cells: in an
empty classroom, mostly the legs of the tables and chairs. Its position is
the centre of the cluster, weighted by each cell's log-odds, so it is found
to better than one cell.

The two lists are matched with RANSAC: two landmarks from each grid fix a
rotation and shift, and the one that lines up the most landmarks wins. With
a few dozen landmarks every pair can be tried, so instead of drawing random
samples the search tries them all (and gives the same answer every time).
Odometry is not used, so the result can be checked against it.
"""
import itertools
import math

import cv2
import numpy as np

from grid import CELL, ORIGIN

OCCUPIED_L = 2.0  # a cell counts as filled above this log-odds
LANDMARK_L = 10.0  # a landmark's strongest cell must reach this (about 11 net hits)
MAX_SIZE = 0.15  # metres: no bigger than a table leg, smeared by odometry
ISOLATION = 0.15  # metres: no other filled cells this close (so not part of a wall)
MATCH_TOL = 0.10  # metres: landmarks closer than this after the transform match


def find_landmarks(grid):
    """List of (x, y, strength) in the grid frame, strongest first."""
    occ = (grid.L > OCCUPIED_L).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    pad = math.ceil(ISOLATION / CELL)
    found = []
    for k in range(1, n):
        # stats are in image terms: x = column = j, y = row = i
        j0, i0, w, h = (stats[k, s] for s in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                              cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        if max(w, h) * CELL > MAX_SIZE + 1e-9:
            continue
        window = (slice(max(i0 - pad, 0), i0 + h + pad), slice(max(j0 - pad, 0), j0 + w + pad))
        mine = labels[window] == k
        L = grid.L[window]
        if L[mine].max() < LANDMARK_L:
            continue
        # any other filled cell within ISOLATION of the blob?
        near = cv2.dilate(mine.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
        if (occ[window].astype(bool) & near & ~mine).any():
            continue
        ii, jj = np.nonzero(mine)
        wts = L[ii, jj]
        x = ((ii + window[0].start + 0.5 - ORIGIN) * CELL * wts).sum() / wts.sum()
        y = ((jj + window[1].start + 0.5 - ORIGIN) * CELL * wts).sum() / wts.sum()
        found.append((float(x), float(y), float(wts.max())))
    return sorted(found, key=lambda f: -f[2])


def apply(T, pts):
    """Apply a 3x3 transform to an N x 2 array of points."""
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:2, :2].T + T[:2, 2]


def rigid_fit(p, q):
    """Least-squares rotation + shift T with T @ q ~ p, for matched N x 2 points."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    H = (q - qc).T @ (p - pc)
    theta = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = pc - R @ qc
    return T


def match(a, b, tol=MATCH_TOL):
    """One-to-one pairs (ia, ib) of points closer than tol, closest first."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    pairs, used_a, used_b = [], set(), set()
    for ia, ib in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape)):
        if d[ia, ib] >= tol:
            break
        if ia not in used_a and ib not in used_b:
            pairs.append((int(ia), int(ib)))
            used_a.add(ia)
            used_b.add(ib)
    return pairs


def ransac_align(a, b, tol=MATCH_TOL):
    """Transform T (3x3) taking grid b's landmarks onto grid a's, and the pairs.

    a, b: N x 2 landmark positions in each grid's own frame. Returns T with
    T @ b[ib] ~ a[ia] for every (ia, ib) in pairs.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError('need at least two landmarks in each grid')
    da = np.linalg.norm(a[:, None] - a[None, :], axis=2)
    db = np.linalg.norm(b[:, None] - b[None, :], axis=2)
    best_key, best_T = None, None
    for i, j in itertools.combinations(range(len(a)), 2):
        for k, l in itertools.permutations(range(len(b)), 2):
            # a rigid transform keeps distances, so only try pairs that agree
            if abs(da[i, j] - db[k, l]) > tol:
                continue
            T = rigid_fit(a[[i, j]], b[[k, l]])
            pairs = match(a, apply(T, b), tol)
            err = sum(np.linalg.norm(a[ia] - apply(T, b[[ib]])[0]) for ia, ib in pairs)
            key = (len(pairs), -err)
            if best_key is None or key > best_key:
                best_key, best_T = key, T
    # refine on every matched pair until the matches stop changing
    T, pairs = best_T, match(a, apply(best_T, b), tol)
    for _ in range(10):
        T = rigid_fit(a[[ia for ia, _ in pairs]], b[[ib for _, ib in pairs]])
        new = match(a, apply(T, b), tol)
        if sorted(new) == sorted(pairs):
            break
        pairs = new
    return T, pairs


def grid_in_frame(L, a_T_b):
    """Grid b's log-odds array, resampled into grid a's cells."""
    # cell index (i, j) -> metres, and (row, col) <-> (x, y) swap for OpenCV
    S = np.array([[CELL, 0, (0.5 - ORIGIN) * CELL], [0, CELL, (0.5 - ORIGIN) * CELL], [0, 0, 1]])
    P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    M = P @ np.linalg.inv(S) @ np.linalg.inv(a_T_b) @ S @ P  # a's (col, row) -> b's
    return cv2.warpAffine(L, M[:2], (L.shape[1], L.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
