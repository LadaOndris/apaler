# Enumerate all images from the test dataset
# and starts an abstract method, which runs the detection
# algorithm and compares the given results with the expected results.

from typing import Callable, Iterable, Tuple

from src.testdataset.generation.laser import dist_from_line2
from src.testdataset.loading import SyntheticLaserDataset, SyntheticLaserDatasetRecord
from src.testdataset.utils import angle_to_positive, positive_line_angle


class Evaluator:
    """
    Evaluates a laser detection algorithm.
    """

    def _reset_counts(self):
        self.failed_count = 0
        self.success_count = 0

    def evaluate(self, dataset: Iterable[SyntheticLaserDatasetRecord],
                 detect_strategy: Callable[[str], Tuple],
                 angle_epsilon=0.0349, source_distance_epsilon=10):
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
            # Detection algorithm returns a line represented by two points in the original image
            point1, point2 = detect_strategy(record.file_path)
            print(f"Evaluating {record.file_path}...")

            # Checks the angle of the line
            positive_angle_rads = positive_line_angle(point1, point2)
            expected_positive_angle_rads = positive_line_angle(record.setting.source, record.setting.target)
            angle_diff = abs(positive_angle_rads - expected_positive_angle_rads)
            angle_checks = angle_diff < angle_epsilon
            if not angle_checks:
                print(f"\tAngle difference is not within limits (allowed = {angle_epsilon}, actual = {angle_diff}).")

            # Checks that the true source lies on the line
            actual_dist = dist_from_line2(record.setting.source, point1, point2)
            max_allowed_dist = source_distance_epsilon
            source_dist_checks = actual_dist < max_allowed_dist
            if not source_dist_checks:
                print(f"\tSource distance is not within limits (allowed = {max_allowed_dist}, actual = {actual_dist}).")

            if angle_checks and source_dist_checks:
                self.success_count += 1
            else:
                self.failed_count += 1
        print("\n====================================")
        print("Finished evaluation.")
        print(f"Succeeded: \t{self.success_count}")
        print(f"Failed: \t{self.failed_count}")
        print("====================================\n")


def get_detection_algorithm(executable_path: str, pixelCountFilePath: str) -> Callable[[str], Tuple]:
    def detect_using_rotation(file_path: str) -> Tuple:
        import subprocess
        params = f'{executable_path } --image {file_path} --filterSize 30 --slopeThreshold 0.1 --minPixelsThreshold 200 ' \
                 f'--pixelCountFile {pixelCountFilePath}'
        proc = subprocess.Popen(params, shell=True, stdout=subprocess.PIPE)
        proc.wait()
        line = proc.stdout.readline().decode('utf-8')
        string_numbers = line.split(',')
        int_numbers = [int(number) for number in string_numbers]
        return int_numbers[:2], int_numbers[2:]

    return detect_using_rotation


if __name__ == "__main__":
    detection_algorithm = get_detection_algorithm('../rotlinedet-gpu/cmake-build-debug/src/rotlinedet_run',
                                                  '../rotlinedet-gpu/src/scripts/columnPixelCounts.dat')
    evaluator = Evaluator()
    dataset = SyntheticLaserDataset('./dataset/testdataset')
    evaluator.evaluate(dataset, detection_algorithm)
