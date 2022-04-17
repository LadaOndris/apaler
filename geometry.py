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


def degs_to_rads(degs: float) -> float:
    return degs / 180 * np.pi


def rads_to_degs(rads: float) -> float:
    return rads / np.pi * 180


def rot_along_z(vec, degs):
    return np.dot(rot_along_z_matrix(degs), vec)


def rot_along_z_matrix(degs):
    rads = degs / 180 * np.pi
    mat = np.array([[np.cos(rads), np.sin(rads), 0, 0],
                    [-np.sin(rads), np.cos(rads), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return mat


def rot_along_x(vec, degs):
    return np.dot(rot_along_x_matrix(degs), vec)


def rot_along_x_matrix(degs):
    rads = degs / 180 * np.pi
    mat = np.array([[1, 0, 0, 0],
                    [0, np.cos(rads), -np.sin(rads), 0],
                    [0, np.sin(rads), np.cos(rads), 0],
                    [0, 0, 0, 1]])
    return mat


def rot_along_y(vec, degs):
    return np.dot(rot_along_y_matrix(degs), vec)


def rot_along_y_matrix(degs):
    rads = degs / 180 * np.pi
    mat = np.array([[np.cos(rads), 0, -np.sin(rads), 0],
                    [0, 1, 0, 0],
                    [np.sin(rads), 0, np.cos(rads), 0],
                    [0, 0, 0, 1]])
    return mat


"""
The following functions originate from 
https://arccoder.medium.com/process-the-output-of-cv2-houghlines-f43c7546deae
"""


def line_end_points_on_image(rho: float, theta: float, image_shape: tuple):
    """
    Returns end points of the line on the end of the image
    Args:
        rho: input line rho
        theta: input line theta
        image_shape: shape of the image

    Returns:
        list: [(x1, y1), (x2, y2)]
    """
    m, b = polar2cartesian(rho, theta, True)

    end_pts = []

    if not np.isclose(m, 0.0):
        x = int(0)
        y = int(solve4y(x, m, b))
        if is_point_within_image(x, y, image_shape):
            end_pts.append((x, y))

        x = int(image_shape[1] - 1)
        y = int(solve4y(x, m, b))
        if is_point_within_image(x, y, image_shape):
            end_pts.append((x, y))

    if m is not np.nan:
        y = int(0)
        x = int(solve4x(y, m, b))
        if is_point_within_image(x, y, image_shape):
            end_pts.append((x, y))

        y = int(image_shape[0] - 1)
        x = int(solve4x(y, m, b))
        if is_point_within_image(x, y, image_shape):
            end_pts.append((x, y))

    return end_pts


def is_point_within_image(x: int, y: int, image_shape: tuple):
    """
    Returns true is x and y are on the image
    """
    return 0 <= y < image_shape[0] and 0 <= x < image_shape[1]


def polar2cartesian(rho: float, theta_rad: float, rotate90: bool = False):
    """
    Converts line equation from polar to cartesian coordinates

    Args:
        rho: input line rho
        theta_rad: input line theta
        rotate90: output line perpendicular to the input line

    Returns:
        m: slope of the line
           For horizontal line: m = 0
           For vertical line: m = np.nan
        b: intercept when x=0
    """
    x = np.cos(theta_rad) * rho
    y = np.sin(theta_rad) * rho
    m = np.nan
    if not np.isclose(x, 0.0):
        m = y / x
    if rotate90:
        if m is np.nan:
            m = 0.0
        elif np.isclose(m, 0.0):
            m = np.nan
        else:
            m = -1.0 / m
    b = 0.0
    if m is not np.nan:
        b = y - m * x

    return m, b


def solve4x(y: float, m: float, b: float):
    """
    From y = m * x + b
         x = (y - b) / m
    """
    if np.isclose(m, 0.0):
        return 0.0
    if m is np.nan:
        return b
    return (y - b) / m


def solve4y(x: float, m: float, b: float):
    """
    y = m * x + b
    """
    if m is np.nan:
        return b

    return m * x + b


def rotation_matrix_to_align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)

    v = np.cross(a_norm, b_norm)
    c = np.dot(a_norm, b_norm)
    v_mat = ssc(v)

    rotation_matrix = np.eye(3, dtype=np.float64) + v_mat + v_mat.dot(v_mat) / (1 + c)
    return rotation_matrix


def ssc(vector: np.ndarray):
    v1, v2, v3 = vector
    v_mat = np.array([[0, -v3, v2],
                     [v3, 0, -v1],
                     [-v2, v1, 0]])
    return v_mat


if __name__ == "__main__":
    A = np.array([100, 214.15, 100])
    B = np.array([0, 214.15, 0])
    R = rotation_matrix_to_align_vectors(B, A)
    pass
