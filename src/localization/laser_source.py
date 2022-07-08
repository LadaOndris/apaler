from typing import List

import cv2
import numpy as np

from data_generation import get_cameras
from data_types import CandidatePlane, ImageCameraPair, Line, Position
from src.detection.base import LineDetector
from src.detection.hough import HoughLineDetector
from geometry import line_plane_intersection, plane_intersect


class LaserSourceDeterminator:

    def __init__(self, detector: LineDetector):
        self.detector = detector

    def find_position(self, image_camera_pairs: List[ImageCameraPair]) -> Position:
        candidates_per_image: List[List[CandidatePlane]] = []
        for image_camera_pair in image_camera_pairs:
            candidates = self.detector.get_candidates(image_camera_pair)
            candidates_per_image.append(candidates)

        matches = self.find_matches(candidates_per_image)
        for match in matches:
            position = self.project_onto_surface(match)
            return position

    def find_matches(self, candidate_planes: List[List[CandidatePlane]]) -> List[Line]:
        """
        Iterates over all permutations of candidates and find the best
        match.
        """
        candidates_size = [len(camera_candidates) for camera_candidates in candidate_planes]
        candidates_indices_axes = np.indices(candidates_size)
        candidates_indices = np.concatenate(candidates_indices_axes, axis=-1)

        matches: List[Line] = []
        for permutation_indices in candidates_indices:
            # Retrieve a single permutation of candidates
            candidates = []
            for camera_index, index_within_camera in enumerate(permutation_indices):
                # index_within_camera = index[camera_index]
                camera_candidates = candidate_planes[camera_index]
                candidate = camera_candidates[index_within_camera]
                candidates.append(candidate)
            # Try to find match using this permutation
            match = self.find_match(candidates)
            matches.append(match)

        return matches

    def find_match(self, candidates: List[CandidatePlane]) -> Line:
        """
        Given a candidate from each image, it intersects all candidates,
        retrieving a possible laser line in 3D.
        """
        # Only two cameras are supported at the moment.
        # The intersection of more than 2 planes does not result in
        # a line analytically.
        if len(candidates) != 2:
            raise AssertionError("Exactly two cameras are supported only.")

        plane1_coeffs = candidates[0].get_coeffs()
        plane2_coeffs = candidates[1].get_coeffs()

        point1, point2 = plane_intersect(plane1_coeffs, plane2_coeffs)
        matched_line = Line(Position.from_array(point1), Position.from_array(point2))
        return matched_line

    def project_onto_surface(self, match: Line) -> Position:
        # For simplicity, assume the surface at z = 500 m.
        planeNormal = np.array([0, 0, 1])
        planePoint = np.array([0, 0, 500])  # Any point on the plane

        lineDirection = match.as_vector()
        linePoint = match.pos1.to_array()  # Any point on the line

        point = line_plane_intersection(planeNormal, planePoint, lineDirection, linePoint)

        position = Position.from_array(point)
        return position


def load_images():
    kernel = np.ones((15, 15), np.uint8)

    image1 = cv2.imread('../../dataset/synthetic/line_0_0.png')
    image1 = cv2.erode(image1, kernel)

    image2 = cv2.imread('../../dataset/synthetic/line_0_1.png')
    image2 = cv2.erode(image2, kernel)
    return image1, image2


if __name__ == "__main__":
    cam1, cam2 = get_cameras()
    img1, img2 = load_images()
    pairs = [ImageCameraPair(img1, cam1), ImageCameraPair(img2, cam2)]
    determinator = LaserSourceDeterminator(HoughLineDetector())
    position = determinator.find_position(pairs)
    print("Laser source: ", position)
