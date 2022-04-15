from unittest import TestCase
import numpy as np
from data_types import CandidatePlane, Line, Position


class TestCandidatePlane(TestCase):
    def test_get_coeffs(self):
        a = np.array([1, 2, 3])
        b = np.array([-1, 4, 3])
        c = np.array([2, 1, 4])

        plane = CandidatePlane(Line(Position.from_array(a), Position.from_array(b)),
                               Line(Position.from_array(a), Position.from_array(c)))

        coeffs = plane.get_coeffs()

        self.assertTrue(np.array_equal(coeffs, np.array([2, 2, 0, -6])))