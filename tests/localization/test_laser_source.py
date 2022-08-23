from unittest import TestCase

import numpy as np

from src.localization.data_types import Line, Position
from src.localization.laser_source import LaserSourceDeterminator


class TestLaserSourceDeterminator(TestCase):
    def test_project_onto_surface(self):
        determinator = LaserSourceDeterminator(None)

        line = Line(Position(0, 0, 450), Position(0, 100, 550))
        position = determinator.project_onto_surface(line)

        self.assertTrue(np.array_equal(position.to_array(), np.array([0, 50, 500])))

