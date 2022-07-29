import argparse
import os

from src.testdataset.algorithms.rotlinedet import get_detection_algorithm
from src.testdataset.evaluation import Evaluator
from src.testdataset.loading import SyntheticLaserDataset


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


args = parse_args()
detection_algorithm = get_detection_algorithm(args.exe_path, args.pixel_counts_path,
                                              filterSize=30, slopeThreshold=0.1, minPixelsThreshold=200)
evaluator = Evaluator()
dataset = SyntheticLaserDataset('./dataset/testdataset')
evaluator.evaluate(dataset, detection_algorithm)
evaluator.print_stats()
