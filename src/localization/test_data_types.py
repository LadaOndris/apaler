from unittest import TestCase

import numpy as np

from data_types import Camera, CameraOrientation, CandidatePlane, ImageSize, Line, Position


class TestCandidatePlane(TestCase):
    def test_get_coeffs(self):
        a = np.array([1, 2, 3])
        b = np.array([-1, 4, 3])
        c = np.array([2, 1, 4])

        plane = CandidatePlane(Line(Position.from_array(a), Position.from_array(b)),
                               Line(Position.from_array(a), Position.from_array(c)))

        coeffs = plane.get_coeffs()

        self.assertTrue(np.array_equal(coeffs, np.array([2, 2, 0, -6])))


class TestCamera(TestCase):
    def test_half_ifovs(self):
        pixel_size = 3.45 * 1e-6
        focal_length = 20 * 1e-3
        elevation = 0
        cam = Camera(Position(0, 0, 500),
                     ImageSize(6480, 4860),
                     CameraOrientation(0, elevation),
                     focal_length, pixel_size)

        half_width = int(cam.resolution.width / 2)
        half_ifovs = cam.half_ifovs(num_pixels=half_width)
        half_fov = np.sum(half_ifovs)

        # The sum of angles should give half of FOV.
        expected_half_fov = cam.fov_azim / 2
        self.assertTrue(np.isclose(half_fov, expected_half_fov))

        # Check the angles are decreasing
        for i in range(1, len(half_ifovs)):
            self.assertTrue(half_ifovs[i - 1] > half_ifovs[i])
