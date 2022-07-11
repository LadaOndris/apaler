import math
import random
from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class LaserLineSetting:

    def __init__(self, source: Tuple, target: Tuple, width: int, pixel_dissipation_factor: float):
        self.source = source
        self.target = target
        self.width = width
        self.pixel_dissipation_factor = pixel_dissipation_factor

    def to_underscore_string(self) -> str:
        ...


class RealisticLaserGenerator:
    """
    Creates synthetic laser lines in an image according to specified options.

    Options are:
    - The background image
    - Laser line intensity
    - Laser length as a pixel dissipation factor
    - Laser line width
    - Randomness range
    """

    def __init__(self, ):
        self.random_interval = np.array([0, 1], dtype=float)
        self.base_intensity = None
        self.laser_width = None
        # The intensity decreases by the distance from source * dissipation factor.
        # Factor of 0.01 means that no intensity is added after 100 pixels.
        self.pixel_dissipation_factor = None

    def draw_laser(self, image: np.ndarray, line_settings: LaserLineSetting, base_intensity: int):
        """
        The Bresenham's algorithm for drawing line with thinkness.
        The base of the algorithm is adopted from members.chello.at/~easyfilter/bresenham.html.

        It is modified to simulate a laser dissipating in air.
        """
        self.base_intensity = base_intensity
        self.pixel_dissipation_factor = line_settings.pixel_dissipation_factor
        self.laser_width = line_settings.width
        wd = self.laser_width
        source = line_settings.source
        target = line_settings.target
        x0 = source[0]
        x1 = target[0]
        y0 = source[1]
        y1 = target[1]

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        if dx + dy == 0:
            ed = 1
        else:
            ed = math.sqrt(dx * dx + dy * dy)

        wd = int((wd + 1) // 2)

        while True:
            image[y0, x0] = self._get_new_pixel_value(image[y0, x0],
                                                      dist_from_middle=dist_from_line2((x0, y0), source, target),
                                                      dist_from_source=dist(source[0], source[1], x0, y0))
            e2 = err
            x2 = x0
            if 2 * e2 >= -dx:
                e2 += dy
                y2 = y0
                while e2 < ed * wd and (y1 != y2 or dx > dy):
                    y2 += sy
                    image[y2, x0] = self._get_new_pixel_value(image[y2, x0],
                                                              dist_from_middle=dist_from_line2((x0, y2), source,
                                                                                               target),
                                                              dist_from_source=dist(source[0], source[1], x0, y2))
                    e2 += dx

                if x0 == x1:
                    break
                e2 = err
                err -= dy
                x0 += sx
            if 2 * e2 <= dy:
                e2 = dx - e2
                while e2 < ed * wd and (x1 != x2 or dx < dy):
                    x2 += sx
                    image[y0, x2] = self._get_new_pixel_value(image[y0, x2],
                                                              dist_from_middle=dist_from_line2((x2, y0), source,
                                                                                               target),
                                                              dist_from_source=dist(source[0], source[1], x2, y0))
                    e2 += dy

                if y0 == y1:
                    break
                err += dx
                y0 += sy
        return image

    def _get_new_pixel_value(self, current_pixel_value, dist_from_middle, dist_from_source):
        """
        Adds an extra intensity to the current pixel, simulating a single pixel of a laser line.
        """
        extra_intensity = self._get_extra_intensity(dist_from_middle, dist_from_source)
        return self._add_value_with_saturation(current_pixel_value, extra_intensity)

    def _add_value_with_saturation(self, base, value, saturates_at=255):
        """
        Returns the sum of two values or the saturation value if the exceeds the saturation value.
        """
        return min(base + value, saturates_at)

    def _get_extra_intensity(self, dist_from_middle, dist_from_source):
        """
        Generates extra intensity, simulating laser strike in the image.

        :param dist_from_middle: The distance from the laser's center line in pixels.
        :param dist_from_source: The distance from the laser origin in pixels.
        :return: Value in the [0, 255] range.
        """
        gauss_intensity = gaussian(dist_from_middle, mu=0, sigma=self.laser_width / 3.0)  # Range [0, 1]
        randomness = random.uniform(self.random_interval[0], self.random_interval[1])
        linear_factor = 1 - min(1, self.pixel_dissipation_factor * dist_from_source)

        extra_intensity = randomness * linear_factor * gauss_intensity * self.base_intensity
        return int(extra_intensity)


def dist(x0, y0, x1, y1):
    """
    Euclidean distance for two points in a 2D space.
    """
    dx = x1 - x0
    dy = y1 - y0
    return math.sqrt(dx * dx + dy * dy)


def gaussian(x, mu, sigma):
    """
    1D gaussian distribution function
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


def dist_from_line(point, line_angle_rads, line_point):
    """
    Distance of a point (x, y) from a line defined by a point (Px, Py) and an angle.
    :return: A number representing the distance.
    """
    cos_factor = math.cos(line_angle_rads) * (line_point[1] - point[1])
    sin_factor = math.sin(line_angle_rads) * (line_point[0] - point[0])
    distance = abs(cos_factor - sin_factor)
    return distance


def dist_from_line2(point, line_point1, line_point2):
    """
    Distance of a point (x, y) from a line defined by two points---(Px1, Py1) and (Px2, Py2).
    :return: A number representing the distance.
    """
    a = (line_point2[0] - line_point1[0]) * (line_point1[1] - point[1])
    b = (line_point1[0] - point[0]) * (line_point2[1] - line_point1[1])
    numerator = abs(a - b)

    c = math.pow(line_point2[0] - line_point1[0], 2)
    d = math.pow(line_point2[1] - line_point1[1], 2)
    denominator = math.sqrt(c + d)

    return numerator / denominator


if __name__ == "__main__":
    image = np.zeros([400, 800], dtype=np.uint8)
    setting = LaserLineSetting(source=(100, 200), target=(200, 0), width=5,
                               pixel_dissipation_factor=0.005)
    generator = RealisticLaserGenerator()
    painting = generator.draw_laser(image, setting, base_intensity=64)

    cv.imwrite('line.png', painting)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(painting, cmap='gray')
    fig.tight_layout()
    fig.show()
