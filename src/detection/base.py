from abc import ABC, abstractmethod
from typing import List

from src.localization.data_types import CandidatePlane


class LineDetector(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_candidates(self, image) -> List[CandidatePlane]:
        pass
