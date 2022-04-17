from typing import List, Tuple

import cv2
import numpy as np
import quaternion

from data_generation import get_cameras, transform_camera_to_world, project_point
from data_types import Camera, CandidatePlane, ImageCameraPair, Line, Position
from detection.base import LineDetector
from geometry import line_end_points_on_image, rot_along_x, rot_along_x_matrix, rot_along_z, rot_along_z_matrix


class HoughLineDetector(LineDetector):

    def __init__(self):
        super().__init__()

    def get_candidates(self, imageCameraPair: ImageCameraPair) -> List[CandidatePlane]:
        # Detect lines in the image
        #
        image = imageCameraPair.image
        # cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('lines', cv2.WINDOW_NORMAL)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=7)
        # cv2.imshow('edges', edges)

        candidate_planes = []
        lines = cv2.HoughLines(edges, 1, np.pi / (16 * 180), 100)
        for rho, theta in lines[0]:
            x0 = np.cos(theta) * rho
            y0 = np.sin(theta) * rho

            if np.isclose(y0, 0.0):
                # Handle the case with a vertical line:
                x1, y1 = int(x0), 0
                x2, y2 = int(x0), imageCameraPair.camera.resolution.height - 1
                end_points = [(x1, y1), (x2, y2)]
            else:
                # The following function doesn't work for vertical lines
                end_points = line_end_points_on_image(rho, theta, imageCameraPair.camera.resolution.to_tuple())
                (x1, y1), (x2, y2) = end_points

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            plane = self.get_plane_from_image_points(imageCameraPair.camera, end_points[0], end_points[1])
            candidate_planes.append(plane)

        # cv2.imshow('lines', image)
        # cv2.waitKey(0)
        return candidate_planes

    def get_plane_from_image_points(self, camera: Camera,
                                    point1: Tuple[int, int], point2: Tuple[int, int]) -> CandidatePlane:
        line1 = self.line_from_camera_to_pixel(camera, point1)
        line2 = self.line_from_camera_to_pixel(camera, point2)

        plane = CandidatePlane(line1, line2)
        coeffs = plane.get_coeffs()
        print(F"{coeffs[0]}x+({coeffs[1]})y+({coeffs[2]})z+({coeffs[3]})=0")
        return plane

    def line_from_camera_to_pixel(self, camera: Camera, pixel: Tuple[int, int]) -> Line:
        """
        :param camera: Camera
        :param pixel: (x, y)
        """
        # Get azimuth and elevation of the pixel
        azim = camera.pixel_azimuths[pixel[0]]
        elev = camera.pixel_elevations[pixel[1]]
        print(F"Azim: {azim:.1f}, Elev: {elev:.1f}")
        # Construct line from the camera in the direction of the pixel
        line = self.line_in_the_direction(camera, azim, elev, pixel)
        return line

    def rotate_inside_camera_system(self, vec, azim_degs, elev_degs):
        direction_in_cameras_coords = vec
        direction_in_cameras_coords = rot_along_x(direction_in_cameras_coords, degs=elev_degs)
        direction_in_cameras_coords = rot_along_z(direction_in_cameras_coords, degs=azim_degs)
        return direction_in_cameras_coords

    def line_in_the_direction(self, camera: Camera, azimuth: float, elevation: float,
                              pixel: Tuple[int, int]) -> Line:
        base_position = camera.position
        camera_azimuth = azimuth - camera.orientation.azimuth
        camera_elevation = elevation - camera.orientation.elevation

        # The camera is looking in the direction of its own "Y" axis
        camera_direction = np.array([0, 1, 0, 1])

        # Firstly, rotate inside the camera's coordinate system
        # direction_in_cameras_coords = self.rotate_inside_camera_system(camera_direction,
        #                                                                camera_azimuth,
        #                                                                camera_elevation)
        x = (pixel[0] - camera.resolution.width / 2) * camera.pixel_size
        z = ((camera.resolution.height - pixel[1]) - camera.resolution.height / 2) * camera.pixel_size
        direction_in_cameras_coords = np.array([x, camera.focal_length, z])
        direction_in_cameras_coords = direction_in_cameras_coords / np.linalg.norm(direction_in_cameras_coords)
        direction_in_cameras_coords = np.concatenate([direction_in_cameras_coords, [1]])

        # Secondly, transform the vector from Camera to World
        camera_to_world = transform_camera_to_world(base_position.to_array(),
                                                    camera.orientation.azimuth,
                                                    camera.orientation.elevation)

        oriented_direction = np.dot(camera_to_world, direction_in_cameras_coords)

        line = Line(base_position, Position.from_array(oriented_direction))
        direction = oriented_direction[:3] - base_position.to_array()
        print(F"Vector(({base_position.x},{base_position.y},{base_position.z}), "
              F"({base_position.x}+10000*{direction[0]},{base_position.y}+10000*{direction[1]},{base_position.z}+10000*{direction[2]}))")
        return line


if __name__ == "__main__":
    cam1, cam2 = get_cameras()
    detector = HoughLineDetector()
    image = cv2.imread('./data/line_0_0.png')
    kernel = np.ones((15, 15), np.uint8)
    image = cv2.erode(image, kernel)
    detector.get_candidates(ImageCameraPair(image, cam1))
