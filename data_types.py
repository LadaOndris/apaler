from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


class Position:

    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, array: np.ndarray):
        return Position(array[0], array[1], array[2])


class Line:

    def __init__(self, pos1: Position, pos2: Position):
        self.pos1 = pos1
        self.pos2 = pos2


class ImageSize:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height


class CameraOrientation:

    def __init__(self, azimuth: float, elevation: float):
        self.azimuth = azimuth
        self.elevation = elevation


class Camera:

    def __init__(self, position: Position, resolution: ImageSize,
                 orientation: CameraOrientation, ifov: float, focal_lenght: float,
                 pixel_size: float):
        self.position = position
        self.resolution = resolution
        self.orientation = orientation
        self.ifov = ifov
        self.focal_length = focal_lenght
        self.pixel_size = pixel_size
        self.pixel_azimuths, self.pixel_elevations = self.compute_per_pixel_orientation()

    def compute_per_pixel_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        ifov_width = self.resolution.width * self.ifov
        ifov_height = self.resolution.height * self.ifov
        pixel_azimuths = np.arange(0, ifov_width, self.ifov)
        pixel_elevations = np.arange(0, ifov_height, self.ifov)
        pixel_elevations = np.flip(pixel_elevations)

        pixel_azimuths += self.orientation.azimuth - ifov_width / 2
        pixel_elevations += self.orientation.elevation - ifov_height / 2
        return pixel_azimuths, pixel_elevations

    def display_image(self, image: np.ndarray):
        num_ticks = 8
        xticks_indices = np.linspace(0, self.resolution.width - 1, num_ticks).astype(int)
        xticks = self.pixel_azimuths[xticks_indices]
        yticks_indices = np.linspace(0, self.resolution.height - 1, num_ticks).astype(int)
        yticks = self.pixel_elevations[yticks_indices]

        formatted_xticks = [F"{tick:.1f}°" for tick in xticks]
        formatted_yticks = [F"{tick:.1f}°" for tick in yticks]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.imshow(image)
        ax.set_xticklabels(formatted_xticks)
        ax.set_yticklabels(formatted_yticks)
        ax.set_xlabel('Azimuth')
        ax.set_ylabel('Elevation')
        fig.tight_layout()
        fig.show()
