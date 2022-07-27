import glob
import os
from typing import List, Tuple

import cv2 as cv
import numpy as np

from laser import LaserLineSetting, RealisticLaserGenerator


class SyntheticLaserDatasetGenerator:

    def __init__(self, dataset_directory: str):
        self.directory = dataset_directory
        self.laser_generator = RealisticLaserGenerator()

        self.num_settings = 10
        self.img_width = 1920
        self.img_height = 1080
        self.allowed_angle_range = [-160, -20]
        self.allowed_laser_length = [100, 2204]

    def generate_dataset(self, background_images: List[str], target_intensities: List[int]) -> None:
        """
        Creates a new dataset from the given images and intensities of the laser.
        :param background_images: Images into which lasers are drawn.
        :param target_intensities: A value in the [0, 255] range for 8-bit pixels.
        """
        os.makedirs(self.directory, exist_ok=True)
        # Generate source image pixels and angles
        # (will be applied to each background image and intesitysetting)
        line_settings = self._generate_laser_line_settings()

        # Iterate over all intesities.
        for intensity in target_intensities:
            subdir = self._prepare_subdir_for_intensity(intensity)
            for img_index, image_path in enumerate(background_images):
                image = cv.imread(image_path)
                if self._check_image_shape(image):
                    self._generate_and_save_for_all_settings(image, line_settings, intensity, subdir, img_index)
                else:
                    print(f"Image '{image_path}' is of different size than expected "
                          f"(expected: {self._get_expected_shape()}, actual: {image.shape}).")

    def _get_expected_shape(self) -> Tuple[int, int, int]:
        return (self.img_height, self.img_width, 3)

    def _check_image_shape(self, image: np.ndarray) -> bool:
        return self._get_expected_shape() == image.shape

    def _prepare_subdir_for_intensity(self, intensity: int) -> str:
        # Create a separate directory for each intensity setting
        subdir = os.path.join(self.directory, f'i{intensity}')
        os.makedirs(subdir, exist_ok=True)
        return subdir

    def _generate_and_save_for_all_settings(self, image: np.ndarray,
                                            line_settings: List[LaserLineSetting],
                                            intensity: int, save_dir: str, index: int):
        for setting in line_settings:
            painting = image.copy()
            image_with_laser = self.laser_generator.draw_laser(painting, setting, intensity)
            save_path = os.path.join(save_dir, f'{setting.to_underscore_string()}_{index}.png')
            cv.imwrite(save_path, image_with_laser)

    def _generate_laser_line_settings(self) -> List[LaserLineSetting]:
        """
        Generates laser line settings for the dataset that is being created.

        Determines position of the laser, its angle, length, and width.
        """
        # Determine source location pixel - probably random (no known heuristic)
        sources_x = np.random.randint(100, self.img_width - 100, self.num_settings)
        sources_y = np.random.randint(int(self.img_height / 2), self.img_height, self.num_settings)
        sources = np.stack([sources_x, sources_y], axis=-1)

        # Determine laser direction - no horizontal lines (not interested)
        random_angles_degs = np.random.randint(self.allowed_angle_range[0], self.allowed_angle_range[1],
                                               self.num_settings)
        random_angles_rads = random_angles_degs / 180 * np.pi

        targets = self._angles_to_targets(sources, random_angles_rads)

        # Determine length of the laser using dissipation factor and randomness
        random_laser_lengths = np.random.randint(self.allowed_laser_length[0], self.allowed_laser_length[1],
                                                 self.num_settings)
        pixel_dissipation_factors = 1 / random_laser_lengths

        # Laser width - predefined
        max_laser_width = 16
        laser_widths = np.linspace(1, max_laser_width, self.num_settings).astype(int)

        settings = []
        for i in range(self.num_settings):
            setting = LaserLineSetting(sources[i], targets[i],
                                       laser_widths[i], pixel_dissipation_factors[i],
                                       (self.img_width, self.img_height))
            settings.append(setting)
        return settings

    def _angles_to_targets(self, sources: np.ndarray, angles_rads: np.ndarray) -> np.ndarray:
        # Determine the target point
        targets = self._angles_to_target_points(sources, angles_rads)
        targets = self._clip_targets(targets)
        return targets

    def _angles_to_target_points(self, sources: np.ndarray, angles_rads: np.ndarray) -> np.ndarray:
        length = 1000  # just an arbitrary factor
        x = length * np.cos(angles_rads)
        y = length * np.sin(angles_rads)
        x = x.astype(int)
        y = y.astype(int)

        targets_x = sources[:, 0] + x
        targets_y = sources[:, 1] + y
        return np.stack([targets_x, targets_y], axis=-1)

    def _clip_targets(self, targets: np.ndarray) -> np.ndarray:
        """
        Clips the target points to lie in the image.
        Subtracts 16 pixels from the bottom of the image, so that
        the laser origin does not lie on the image boundary.
        :param targets: An ndarray of [x,y] points.
        """
        targets[:, 0] = np.clip(targets[:, 0], 0, self.img_width - 16).astype(int)
        targets[:, 1] = np.clip(targets[:, 1], 0, self.img_height - 16).astype(int)
        return targets


if __name__ == "__main__":
    background_images = glob.glob('./dataset/testdataset/images/*')
    dataset_dir = './dataset/testdataset'
    target_intensities = (2 ** np.arange(3, 8)).tolist()

    generator = SyntheticLaserDatasetGenerator(dataset_dir)
    generator.generate_dataset(background_images, target_intensities)
