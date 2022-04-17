from typing import List

import cv2
import numpy as np

from data_generation import get_cameras
from data_types import CandidatePlane, ImageCameraPair
from detection.base import LineDetector
from geometry import line_end_points_on_image
from vectors import plane_from_image_points


class HoughLineDetector(LineDetector):

    def __init__(self):
        super().__init__()

    def get_candidates(self, image_camera_pair: ImageCameraPair) -> List[CandidatePlane]:
        # Detect lines in the image
        image = image_camera_pair.image
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
                x2, y2 = int(x0), image_camera_pair.camera.resolution.height - 1
                end_points = [(x1, y1), (x2, y2)]
            else:
                # The following function doesn't work for vertical lines
                end_points = line_end_points_on_image(rho, theta, image_camera_pair.camera.resolution.to_tuple())
                (x1, y1), (x2, y2) = end_points

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            plane = plane_from_image_points(image_camera_pair.camera, end_points[0], end_points[1])
            candidate_planes.append(plane)

        # cv2.imshow('lines', image)
        # cv2.waitKey(0)
        return candidate_planes


if __name__ == "__main__":
    cam1, cam2 = get_cameras()
    detector = HoughLineDetector()
    image = cv2.imread('./data/line_0_0.png')
    kernel = np.ones((15, 15), np.uint8)
    image = cv2.erode(image, kernel)
    detector.get_candidates(ImageCameraPair(image, cam1))
