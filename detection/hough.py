from typing import List

from data_types import CandidatePlane
from detection.base import LineDetector

class HoughLineDetector(LineDetector):

    def __init__(self):
        super().__init__()

    def get_candidates(self, image) -> List[CandidatePlane]:
        # Detect lines in the image
        #

        pass
