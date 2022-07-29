from src.testdataset.algorithms.rotlinedet import get_detection_algorithm
from src.testdataset.evaluation import Evaluator
from src.testdataset.loading import SyntheticLaserDataset

detection_algorithm = get_detection_algorithm('../rotlinedet-gpu/cmake-build-release/src/rotlinedet_run',
                                              '../rotlinedet-gpu/src/scripts/columnPixelCounts.dat',
                                              filterSize=30, slopeThreshold=0.1, minPixelsThreshold=200)
evaluator = Evaluator()
dataset = SyntheticLaserDataset('./dataset/testdataset')
evaluator.evaluate(dataset, detection_algorithm)
evaluator.print_stats()
