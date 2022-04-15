from typing import List

import numpy as np

from data_types import Camera, CandidatePlane, Line, Position
from detection.base import LineDetector
from geometry import line_plane_intersection, plane_intersect


class ImageCameraPair:

    def __init__(self, image, camera: Camera):
        self.image = image
        self.camera = camera


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
            print(position)
            return position

    def find_matches(self, candidates: List[List[CandidatePlane]]) -> List[Line]:
        """
        Iterates over all permutations of candidates and find the best
        match.
        """
        candidates_size = [len(camera_candidates) for camera_candidates in candidates]
        candidates_indices_axes = np.indices(candidates_size)
        candidates_indices = np.stack(candidates_indices_axes, axis=-1)

        matches: List[Line] = []
        for permutation_indices in candidates_indices:
            candidates = []
            for camera_index, index in enumerate(permutation_indices):
                camera_candidates = candidates[camera_index]
                candidate = camera_candidates[index]
                candidates.append(candidate)
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
