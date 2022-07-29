"""
This scripts runs evaluation for different parameters of the rotlinedet detection algorithm.
Selects the parameters with the highest success rate on the testdataset.
"""
import numpy as np
import src.testdataset.utils as utils

from src.testdataset.algorithms.rotlinedet import get_detection_algorithm
from src.testdataset.evaluation import Evaluator
from src.testdataset.loading import SyntheticLaserDataset

# Generate parameters
filterSizes = np.array([10, 16, 20, 26, 30, 36, 40, 46, 50], dtype=float)
slopeThreshold = np.array([0.025, 0.05, 0.075, 0.1, 0.15, 0.2], dtype=float)
minPixelsThreshold = np.array([100, 200, 300, 400], dtype=float)
params = utils.cartesian([filterSizes, slopeThreshold, minPixelsThreshold])

# Run evaluation for each config
evaluator = Evaluator()
dataset = SyntheticLaserDataset('./dataset/testdataset')

success_counts = []
stats = []

for param_idx in range(params.shape[0]):
    param = params[param_idx]

    detection_algorithm = get_detection_algorithm('../rotlinedet-gpu/cmake-build-release/src/rotlinedet_run',
                                                  '../rotlinedet-gpu/src/scripts/columnPixelCounts.dat',
                                                  filterSize=param[0], slopeThreshold=param[1], minPixelsThreshold=param[2])

    evaluator.evaluate(dataset, detection_algorithm, verbose=False)
    # Save results in a container
    success_counts.append(evaluator.success_count)
    stats.append(evaluator.stats)

    print(f"{param_idx}: {evaluator.success_count}")

# Select and print the best params and results
success_counts = np.array(success_counts)
best_param_id = np.argmax(success_counts)

print("=================================================")
print(f"Best params: --filterSize {filterSizes[best_param_id]} "
      f"--slopeThreshold {slopeThreshold[best_param_id]} "
      f"--minPixelsThreshold {minPixelsThreshold[best_param_id]}")
evaluator.stats = stats[best_param_id]
evaluator.print_stats()
print("=================================================")
