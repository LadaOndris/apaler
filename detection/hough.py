from typing import List, Tuple

import cv2
import numpy as np

from data_generation import get_cameras
from data_types import Camera, CandidatePlane, ImageCameraPair, Line, Position
from detection.base import LineDetector


class HoughLineDetector(LineDetector):

    def __init__(self):
        super().__init__()

    def get_candidates(self, imageCameraPair: ImageCameraPair) -> List[CandidatePlane]:
        # Detect lines in the image
        #
        image = imageCameraPair.image
        # cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
        cv2.namedWindow('lines', cv2.WINDOW_NORMAL)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=7)
        # cv2.imshow('edges', edges)

        candidate_planes = []
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        for rho, theta in lines[0]:
            end_points = line_end_points_on_image(rho, theta, imageCameraPair.camera.resolution.to_tuple())
            # a = np.cos(theta)
            # b = np.sin(theta)
            # x0 = a * rho
            # y0 = b * rho
            # x1 = int(x0 + 1000 * (-b))
            # y1 = int(y0 + 1000 * (a))
            # x2 = int(x0 - 1000 * (-b))
            # y2 = int(y0 - 1000 * (a))
            (x1, y1), (x2, y2) = end_points
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            plane = self.get_plane_from_image_points(imageCameraPair.camera, end_points[0], end_points[1])
            candidate_planes.append(plane)

        cv2.imshow('lines', image)
        cv2.waitKey(0)
        return candidate_planes

    def get_plane_from_image_points(self, camera: Camera,
                                    point1: Tuple[int, int], point2: Tuple[int, int]) -> CandidatePlane:
        line1 = self.line_from_camera_to_pixel(camera, point1)
        line2 = self.line_from_camera_to_pixel(camera, point2)

        plane = CandidatePlane(line1, line2)
        return plane

    def line_from_camera_to_pixel(self, camera: Camera, pixel: Tuple[int, int]) -> Line:
        """
        :param camera: Camera
        :param pixel: (x, y)
        """
        # Get azimuth and elevation of the pixel
        azim = camera.pixel_azimuths[pixel[0]]
        elev = camera.pixel_elevations[pixel[1]]
        # Construct line from the camera in the direction of the pixel
        line = self.line_in_the_direction(camera.position, azim, elev)
        return line

    def line_in_the_direction(self, base_position: Position, azimuth: float, elevation: float) -> Line:
        # If the cameras was looking at the position with azimuth of 0
        # and elevation 0, then it is looking exactly along the Y axis.
        # The azimuth is measured from the Y axis.
        # direction = base_position.to_array() + np.array([0, 1, 0])
        direction = np.array([0, 1, 0])

        # Angles to radians
        azim_rads = azimuth / 180 * np.pi
        elev_rads = elevation / 180 * np.pi

        rotation_matrix_azim = np.array([[np.cos(azim_rads), np.sin(azim_rads), 0],
                                         [-np.sin(azim_rads), np.cos(azim_rads), 0],
                                         [0, 0, 1]])
        rotation_matrix_elev = np.array([[1, 0, 0],
                                         [0, np.cos(elev_rads), -np.sin(elev_rads)],
                                         [0, np.sin(elev_rads), np.cos(elev_rads)]])
        rotation_matrix = np.dot(rotation_matrix_elev, rotation_matrix_azim)
        oriented_direction = np.dot(rotation_matrix, direction)

        line = Line(base_position, Position.from_array(base_position.to_array() + oriented_direction))
        return line

"""
The following functions originate from 
https://arccoder.medium.com/process-the-output-of-cv2-houghlines-f43c7546deae
"""
def line_end_points_on_image(rho: float, theta: float, image_shape: tuple):
    """
    Returns end points of the line on the end of the image
    Args:
        rho: input line rho
        theta: input line theta
        image_shape: shape of the image

    Returns:
        list: [(x1, y1), (x2, y2)]
    """
    m, b = polar2cartesian(rho, theta, True)

    end_pts = []

    if not np.isclose(m, 0.0):
        x = int(0)
        y = int(solve4y(x, m, b))
        if is_point_within_image(x, y, image_shape):
            end_pts.append((x, y))
            x = int(image_shape[1] - 1)
            y = int(solve4y(x, m, b))
            if is_point_within_image(x, y, image_shape):
                end_pts.append((x, y))

    if m is not np.nan:
        y = int(0)
        x = int(solve4x(y, m, b))
        if is_point_within_image(x, y, image_shape):
            end_pts.append((x, y))
            y = int(image_shape[0] - 1)
            x = int(solve4x(y, m, b))
            if is_point_within_image(x, y, image_shape):
                end_pts.append((x, y))

    return end_pts


def polar2cartesian(rho: float, theta_rad: float, rotate90: bool = False):
    """
    Converts line equation from polar to cartesian coordinates

    Args:
        rho: input line rho
        theta_rad: input line theta
        rotate90: output line perpendicular to the input line

    Returns:
        m: slope of the line
           For horizontal line: m = 0
           For vertical line: m = np.nan
        b: intercept when x=0
    """
    x = np.cos(theta_rad) * rho
    y = np.sin(theta_rad) * rho
    m = np.nan
    if not np.isclose(x, 0.0):
        m = y / x
    if rotate90:
        if m is np.nan:
            m = 0.0
        elif np.isclose(m, 0.0):
            m = np.nan
        else:
            m = -1.0 / m
    b = 0.0
    if m is not np.nan:
        b = y - m * x

    return m, b


def solve4x(y: float, m: float, b: float):
    """
    From y = m * x + b
         x = (y - b) / m
    """
    if np.isclose(m, 0.0):
        return 0.0
    if m is np.nan:
        return b
    return (y - b) / m


def solve4y(x: float, m: float, b: float):
    """
    y = m * x + b
    """
    if m is np.nan:
        return b

    return m * x + b


def is_point_within_image(x: int, y: int, image_shape: tuple):
    """
    Returns true is x and y are on the image
    """
    return 0 <= y < image_shape[0] and 0 <= x < image_shape[1]


if __name__ == "__main__":
    cam1, cam2 = get_cameras()
    detector = HoughLineDetector()
    image = cv2.imread('./data/line_0_0.png')
    kernel = np.ones((15, 15), np.uint8)
    image = cv2.erode(image, kernel)
    detector.get_candidates(ImageCameraPair(image, cam1))
