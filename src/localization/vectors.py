from typing import Tuple

import numpy as np

from src.localization.data_generation import transform_camera_to_world_matrix
from src.localization.data_types import Camera, CandidatePlane, Line, Position


def plane_from_image_points(camera: Camera,
                            point1: Tuple[int, int], point2: Tuple[int, int]) -> CandidatePlane:
    line1 = line_from_camera_to_pixel(camera, point1)
    line2 = line_from_camera_to_pixel(camera, point2)

    plane = CandidatePlane(line1, line2)

    # Uncomment for debugging information
    # coeffs = plane.get_coeffs()
    # print(F"{coeffs[0]}x+({coeffs[1]})y+({coeffs[2]})z+({coeffs[3]})=0")
    return plane


def line_from_camera_to_pixel(camera: Camera, pixel: Tuple[int, int]) -> Line:
    """
    Creates vector in world coordinates in the direction specified by the pixel.

    :param camera: Camera
    :param pixel: (x, y)
    """
    base_position = camera.position

    # Firstly, determine direction inside the camera's coordinate system
    direction_in_cameras_coords = vector_in_direction_of_pixel(camera, pixel)

    # Secondly, transform the vector from Camera to World
    camera_to_world = transform_camera_to_world_matrix(base_position.to_array(),
                                                       camera.orientation.azimuth,
                                                       camera.orientation.elevation)

    oriented_direction = np.dot(camera_to_world, direction_in_cameras_coords)

    line = Line(base_position, Position.from_array(oriented_direction))

    # Uncomment for debugging information
    # direction = oriented_direction[:3] - base_position.to_array()
    # print(F"Vector(({base_position.x},{base_position.y},{base_position.z}), "
    #       F"({base_position.x}+10000*{direction[0]},{base_position.y}+10000*{direction[1]},{base_position.z}+10000*{direction[2]}))")

    return line


def vector_in_direction_of_pixel(camera: Camera, pixel: Tuple[int, int]):
    """
    Uses pixel's location to determine physical point on a projection plane
    in the camera's coordinate system.

    :param camera: Camera
    :param pixel: (x, y)
    :return: np.ndarray([x, y, z, 1])
    """
    x = (pixel[0] - camera.resolution.width / 2) * camera.pixel_size
    z = (camera.resolution.height / 2 - pixel[1]) * camera.pixel_size

    direction_in_cameras_coords = np.array([x, camera.focal_length, z])
    direction_in_cameras_coords /= np.linalg.norm(direction_in_cameras_coords)
    direction_in_cameras_coords = np.concatenate([direction_in_cameras_coords, [1]])

    return direction_in_cameras_coords
