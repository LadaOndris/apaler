import math
from typing import Tuple

import numpy as np


# Author: f5r5e5d
# Date: answered Jan 6, 2018 at 23:43
# https://stackoverflow.com/a/48133025/3961841
def plane_intersect(plane1: np.ndarray, plane2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    plane1_norm, plane2_norm = np.array(plane1[:3]), np.array(plane2[:3])

    aXb_vec = np.cross(plane1_norm, plane2_norm)

    A = np.array([plane1_norm, plane2_norm, aXb_vec])
    d = np.array([-plane1[3], -plane2[3], 0.]).reshape(3, 1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]


def line_plane_intersection(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


def get_distance(pos1: np.ndarray, pos2: np.ndarray):
    return math.sqrt(math.pow(pos1[0] - pos2[0], 2) +
                     math.pow(pos1[1] - pos2[1], 2) +
                     math.pow(pos1[2] - pos2[2], 2))
