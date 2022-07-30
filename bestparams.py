"""
This scripts runs evaluation for different parameters of the rotlinedet detection algorithm.
Selects the parameters with the highest success rate on the testdataset.
"""

import argparse
import os

import numpy as np

import src.testdataset.utils as utils
from src.testdataset.algorithms.rotlinedet import get_detection_algorithm
from src.testdataset.evaluation import Evaluator
from src.testdataset.loading import SyntheticLaserDataset


class BestParamsSeeker:

    def __init__(self, exe_path: str, pixel_counts_path: str):
        self.exe_path = exe_path
        self.pixel_counts_path = pixel_counts_path

        self.evaluator = Evaluator()
        self.dataset = SyntheticLaserDataset('./dataset/testdataset')

    def find_best_params(self):
        self._generate_params()
        self._run_evaluation_for_each_config()
        self._select_and_print_best_params()

    def _generate_params(self):
        filter_sizes = np.array([10, 16, 20, 26, 30, 36, 40, 46, 50], dtype=float)
        slope_threshold = np.array([0.025, 0.05, 0.075, 0.1, 0.15, 0.2], dtype=float)
        min_pixels_threshold = np.array([100, 200, 300, 400], dtype=float)
        self.params = utils.cartesian([filter_sizes, slope_threshold, min_pixels_threshold])

    def _run_evaluation_for_each_config(self):
        self._reset_stats()

        for param_idx in range(self.params.shape[0]):
            param = self.params[param_idx]
            self._evaluate_for_config(param)
            self._save_results_for_later()
            print(f"{param_idx}: {self.evaluator.success_count}")

    def _reset_stats(self):
        self.success_counts = []
        self.stats = []

    def _evaluate_for_config(self, param):
        detection_algorithm = get_detection_algorithm(self.exe_path, self.pixel_counts_path,
                                                      filterSize=param[0], slopeThreshold=param[1],
                                                      minPixelsThreshold=param[2])
        self.evaluator.evaluate(self.dataset, detection_algorithm, verbose=False)

    def _save_results_for_later(self):
        self.success_counts.append(self.evaluator.success_count)
        self.stats.append(self.evaluator.stats)

    def _select_and_print_best_params(self):
        success_counts = np.array(self.success_counts)
        best_param_id = np.argmax(success_counts)
        best_param = self.params[best_param_id]

        print("=================================================")
        print(f"Best params: --filterSize {best_param[0]} "
              f"--slopeThreshold {best_param[1]} "
              f"--minPixelsThreshold {best_param[2]}")
        self.evaluator.stats = self.stats[best_param_id]
        self.evaluator.print_stats()
        print("=================================================")


def parse_args():
    parser = argparse.ArgumentParser(description='Find best parameters for rotlinedet detection algorithm')
    parser.add_argument('--exe-path', action='store', type=str,
                        default='../rotlinedet-gpu/cmake-build-release/src/rotlinedet_run')
    parser.add_argument('--pixel-counts-path', action='store', type=str,
                        default='../rotlinedet-gpu/src/scripts/columnPixelCounts.dat')
    args = parser.parse_args()

    if not os.path.isfile(args.exe_path):
        raise ValueError(f"The file {args.exe_path} doesn't exist.")

    if not os.path.isfile(args.pixel_counts_path):
        raise ValueError(f"The file {args.pixel_counts_path} doesn't exist.")
    return args


if __name__ == "__main__":
    args = parse_args()
    param_seeker = BestParamsSeeker(args.exe_path, args.pixel_counts_path)
    param_seeker.find_best_params()
