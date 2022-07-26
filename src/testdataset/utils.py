import math

import numpy as np


def positive_line_angle(point1, point2):
    angle_rads = line_angle(point1, point2)
    positive_angle_rads = angle_to_positive(angle_rads)
    return positive_angle_rads


def line_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    if dx == 0:
        return np.inf
    angle_rads = math.atan(dy / dx)
    return angle_rads


def angle_to_positive(angle_rads):
    """
    Converts negative angles to positive.
    Such as the angle of -45° is the same as 135°.
    The difference is only in the direction of the line, which,
    in this use case, is irrelevant.
    """
    if angle_rads < 0:
        return np.pi + angle_rads
    else:
        return angle_rads