"""
Given an image
generates fit with laser lines
of varying intensity and


- Various background images
- Image size?
- Laser options: intensity, length, width, randomness
-
"""
import math

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class RealisticLaserGenerator:

    def __init__(self, ):
        pass

    def generate_image(self, background: np.ndarray, dissipation_factor: float,
                       width: int, intensity: float) -> np.ndarray:
        height = background.shape[0]
        width = background.shape[1]

        line_params = self.generate_line_params(dissipation_factor, width, intensity)
        generated_image = self.draw_line(background, line_params)
        return generated_image

    def generate_line_params(self, dissipation_factor: float, width: int, intensity: float):
        params = {}
        # Determine source location pixel - probably random (no known heuristic)

        # Determine laser direction - no horizontal lines (not interested)
        # Angles [-80; 80] suffice

        # Determine length of the laser using dissipation factor and randomness

        return params

    def draw_line(self, image: np.ndarray, line_params):
        """
        The Bresenham's algorithm for drawing line with thinkness.
        The base of the algorithm is adopted from members.chello.at/~easyfilter/bresenham.html.

        It is modified to simulate a laser dissipating in air.
        """
        x0 = line_params['x0']
        x1 = line_params['x1']
        y0 = line_params['y0']
        y1 = line_params['y1']
        wd = line_params['wd']

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy
        e2 = None
        x2 = None
        y2 = None

        if dx + dy == 0:
            ed = 1
        else:
            ed = math.sqrt(dx * dx + dy * dy)

        wd = int((wd + 1) // 2)

        while True:
            image[y0, x0] = 255
            e2 = err
            x2 = x0
            if 2 * e2 >= -dx:
                e2 += dy
                y2 = y0
                while e2 < ed * wd and (y1 != y2 or dx > dy):
                    y2 += sy
                    image[y2, x0] = 255
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
                    image[y0, x2] = 255
                    e2 += dy

                if y0 == y1:
                    break
                err += dx
                y0 += sy
        return image


if __name__ == "__main__":
    image = np.zeros([400, 800], dtype=np.uint8)
    params = {'x0': 100, 'y0': 200, 'x1': 200, 'y1': 0, 'wd': 2}
    generator = RealisticLaserGenerator()
    painting = generator.draw_line(image, params)

    cv.imwrite('line.png', painting)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(painting, cmap='gray')
    fig.tight_layout()
    fig.show()
