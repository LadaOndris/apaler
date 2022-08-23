from unittest import TestCase

import numpy as np

from src.localization.geometry import line_plane_intersection


class Test(TestCase):
    def test_line_plane_intersection(self):
        planeNormal = np.array([0, 0, 1])
        planePoint = np.array([0, 0, 5])  # Any point on the plane

        rayDirection = np.array([0, -1, -1])
        rayPoint = np.array([0, 0, 10])  # Any point along the ray

        point = line_plane_intersection(planeNormal, planePoint, rayDirection, rayPoint)

        self.assertTrue(np.array_equal(point, np.array([0, -5, 5])))
