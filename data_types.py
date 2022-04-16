import math
from typing import Tuple
from geometry import get_distance

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

    def __str__(self):
        return f'Position({self.x}, {self.y}, {self.z})'


class Line:

    def __init__(self, pos1: Position, pos2: Position):
        self.pos1 = pos1
        self.pos2 = pos2

    def as_vector(self) -> np.ndarray:
        return self.pos2.to_array() - self.pos1.to_array()

    def __str__(self):
        return f'Line({self.pos1}, {self.pos2})'


class ImageSize:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def to_tuple(self):
        return self.height, self.width


class CameraOrientation:

    def __init__(self, azimuth: float, elevation: float):
        self.azimuth = azimuth
        self.elevation = elevation


class Camera:

    def __init__(self, position: Position, resolution: ImageSize,
                 orientation: CameraOrientation, focal_length: float,
                 pixel_size: float):
        self.position = position
        self.resolution = resolution
        self.orientation = orientation
        self.focal_length = focal_length
        self.pixel_size = pixel_size

        self.fov_azim = 2 * math.atan(pixel_size * resolution.width / (2 * focal_length)) / np.pi * 180
        self.fov_elev = 2 * math.atan(pixel_size * resolution.height / (2 * focal_length)) / np.pi * 180

        self.pixel_azimuths, self.pixel_elevations = self.compute_per_pixel_orientation()

    def half_ifovs(self, num_pixels: int):
        # Cumulative sum of pixel sizes
        pixel_sizes = np.full(shape=[num_pixels], fill_value=self.pixel_size)
        cum_pixel_sizes = np.cumsum(pixel_sizes)
        # Compute angles for cumulative sum of pixel sizes
        cum_ifov = np.arctan(cum_pixel_sizes / self.focal_length)
        # Undo cumsum
        ifov_difs_rads = cum_ifov[1:] - cum_ifov[:-1].copy()
        # Prepend the first element that was forgotten in the previous step
        ifov_rads = np.concatenate([[cum_ifov[0]], ifov_difs_rads], axis=-1)
        # Convert to degrees
        ifov_degs = ifov_rads / np.pi * 180
        return ifov_degs

    def compute_per_pixel_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        # ifov_width = self.resolution.width * self.ifov_azim
        # ifov_height = self.resolution.height * self.ifov_elev
        # pixel_azimuths = np.arange(0, ifov_width, self.ifov_azim) + self.ifov_azim
        # pixel_elevations = np.arange(0, ifov_height, self.ifov_elev) + self.ifov_elev
        # pixel_elevations = np.flip(pixel_elevations)

        half_ifovs_azim = self.half_ifovs(int(self.resolution.width / 2))
        half_ifovs_elev = self.half_ifovs(int(self.resolution.height / 2))

        half_ifovs_azim_cum = np.cumsum(half_ifovs_azim)
        half_ifovs_elev_cum = np.cumsum(half_ifovs_elev)

        # Flip the array symmetrically
        reversed_half_ifovs_azim = np.flipud(half_ifovs_azim_cum)
        reversed_half_ifovs_elev = np.flipud(half_ifovs_elev_cum)

        # There is no 0 in the cetre of the image -> add it manually
        # by removing the first element and later adding 0.
        reversed_half_ifovs_azim_corrected = reversed_half_ifovs_azim[1:]
        reversed_half_ifovs_elev_corrected = reversed_half_ifovs_elev[1:]

        # Concatenate symmetrically and add the 0.
        ifovs_azim = np.concatenate([-reversed_half_ifovs_azim_corrected, [0], half_ifovs_azim_cum])
        ifovs_elev = np.concatenate([-reversed_half_ifovs_elev_corrected, [0], half_ifovs_elev_cum])
        ifovs_elev_flipped = np.flip(ifovs_elev)

        pixel_azimuths = ifovs_azim + self.orientation.azimuth
        pixel_elevations = ifovs_elev_flipped + self.orientation.elevation
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
        ax.set_xticks(xticks_indices)
        ax.set_yticks(yticks_indices)
        ax.set_xticklabels(formatted_xticks)
        ax.set_yticklabels(formatted_yticks)
        ax.set_xlabel('Azimuth')
        ax.set_ylabel('Elevation')
        fig.tight_layout()
        fig.show()


class CandidatePlane:
    """
    CandidateLine is a representation of a 2D line in the image.
    It contains information about the azimuth, elevation and the cameras position.
    Together they form a plane.
    """

    def __init__(self, line1: Line, line2: Line):
        self.line1 = line1
        self.line2 = line2
        self.normal = np.cross(line1.as_vector(), line2.as_vector())

    def get_coeffs(self):
        c = -np.dot(self.normal, self.line1.pos1.to_array())
        coeffs = np.concatenate([self.normal, [c]], axis=-1)
        return coeffs

    def get_normal(self):
        return self.normal


class ImageCameraPair:

    def __init__(self, image, camera: Camera):
        self.image = image
        self.camera = camera
