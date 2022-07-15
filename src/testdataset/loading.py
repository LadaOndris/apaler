import glob
import os
from pathlib import Path

from src.testdataset.generation.laser import LaserLineSetting


class SyntheticLaserDatasetRecord:

    def __init__(self, file_path: str, setting: LaserLineSetting):
        self.file_path = file_path
        self.setting = setting


class SyntheticLaserDataset:
    """
    Iterable wrapper over the Synthetic Laser Dataset.
    Reads all images in the corresponding directory and
    creates a SyntheticLaserDatasetRecord for each file.
    """

    def __init__(self, dataset_dir: str):
        self.directory = dataset_dir

    def __iter__(self):
        image_patterns = os.path.join(self.directory, 'i*/*')
        self.file_paths = glob.glob(image_patterns)
        self.file_path_index = 0
        return self

    def __next__(self):
        if self.file_path_index < len(self.file_paths):
            file_path = self.file_paths[self.file_path_index]
            self.file_path_index += 1

            filename = Path(file_path).stem
            line_setting = LaserLineSetting.from_underscore_string(filename)

            if line_setting is None:
                return self.__next__()
            else:
                record = SyntheticLaserDatasetRecord(file_path, line_setting)
                return record
        else:
            raise StopIteration()