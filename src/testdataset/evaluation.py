# Enumerate all images from the test dataset
# and starts an abstract method, which runs the detection
# algorithm and compares the given results with the expected results.
import math
from typing import Callable, Dict, Iterable, Tuple

from src.testdataset.generation.laser import dist_from_line2
from src.testdataset.loading import SyntheticLaserDatasetRecord
from src.testdataset.utils import positive_line_angle


class Evaluator:
    """
    Evaluates a laser detection algorithm.
    """

    def __init__(self):
        self.num_bins = 5

    def _reset_counts(self) -> None:
        self.failed_count = 0
        self.success_count = 0
        self.stats = {'intensity': {},
                      'width': {},
                      'length': {},
                      'image': {}}

    def update_stats(self, record: SyntheticLaserDatasetRecord, successful: bool) -> None:
        intensity = record.get_intensity()
        width = record.setting.width
        length_bin = self._length_to_bin(record.setting.get_length(), record.setting.image_size)
        image_number = record.setting.image_index
        val = 1 - successful  # Count only unsuccessful
        self._insert_or_update(self.stats['intensity'], intensity, val)
        self._insert_or_update(self.stats['width'], width, val)
        self._insert_or_update(self.stats['length'], length_bin, val)
        self._insert_or_update(self.stats['image'], image_number, val)

    def _length_to_bin(self, length: int, image_size: Tuple, ) -> int:
        max_length = int(math.sqrt(image_size[0] ** 2 + image_size[1] ** 2))
        bin_size = int(max_length / self.num_bins)
        bin_from_length = int(length // bin_size) + 1
        bin_max_boundary = min(bin_from_length * bin_size, max_length)
        return bin_max_boundary

    def _insert_or_update(self, dict: Dict, key, val: int) -> None:
        if key not in dict:
            dict[key] = val
        else:
            dict[key] += val

    def print_stats(self) -> None:
        print("Failed detections:")
        for key, subdict in self.stats.items():
            print(key)
            sorted_subdict = dict(sorted(subdict.items()))

            for subkey in sorted_subdict.keys():
                print(f'\t{subkey}', end='')
            print()
            for subkey, val in sorted_subdict.items():
                print(f"\t{val}", end='')
            print()
        print("====================================\n")

    def _check_candidate_line(self, candidate: Tuple, record: SyntheticLaserDatasetRecord,
                              angle_epsilon=0.0349, source_distance_epsilon=10, verbose=True) -> bool:
        point1, point2 = candidate

        # Checks the angle of the line
        positive_angle_rads = positive_line_angle(point1, point2)
        expected_positive_angle_rads = positive_line_angle(record.setting.source, record.setting.target)
        angle_diff = abs(positive_angle_rads - expected_positive_angle_rads)
        angle_checks = angle_diff < angle_epsilon
        # if verbose:
        #     if not angle_checks:
        #         print(f"\tAngle difference is not within limits (allowed = {angle_epsilon}, actual = {angle_diff}).")

        # Checks that the true source lies on the line
        actual_dist = dist_from_line2(record.setting.source, point1, point2)
        max_allowed_dist = source_distance_epsilon
        source_dist_checks = actual_dist < max_allowed_dist
        # if verbose:
        #     if not source_dist_checks:
        #         print(f"\tSource distance is not within limits (allowed = {max_allowed_dist}, actual = {actual_dist}).")

        is_successful = angle_checks and source_dist_checks
        return is_successful

    def evaluate(self, dataset: Iterable[SyntheticLaserDatasetRecord],
                 detect_strategy: Callable[[str], Tuple],
                 angle_epsilon=0.0349, source_distance_epsilon=10, verbose=True) -> None:
        """
        Evaluates a detection algorithm on the synthetic laser dataset by
        calling detection algorithm, which returns line represented by two points
        in the original image and checks that the angle of the line is similar
        to the expected angle. It also checks that the true laser source lies on this
        line.
        :param dataset: Input dataset for the evaluation
        :param detect_strategy: The algorithm for laser detection as a function
        :param angle_epsilon: Maximal allowed difference in angles (default 0.0349 rads ~ 2°)
        :param source_distance_epsilon: Maximal number of pixels, which are still considered correct
        :return:
        """
        self._reset_counts()

        for record in dataset:
            # Detection algorithm returns several line candidates,
            # each represented by two points in the original image
            candidates = detect_strategy(record.file_path)

            if verbose:
                print(f"Evaluating {record.file_path}... ", end='')

            checked_candidates = [
                self._check_candidate_line(candidate, record, angle_epsilon, source_distance_epsilon, verbose)
                for candidate in candidates]
            is_successful = any(checked_candidates)

            if verbose:
                message = ' succeeded' if is_successful else ' failed'
                print(message)

            if is_successful:
                self.success_count += 1
            else:
                self.failed_count += 1
            self.update_stats(record, is_successful)

        if verbose:
            print("\n====================================")
            print("Finished evaluation.")
            print(f"Succeeded: \t{self.success_count}")
            print(f"Failed: \t{self.failed_count}")
            print("====================================\n")
