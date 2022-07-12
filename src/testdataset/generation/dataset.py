import glob
import os
from typing import List

import cv2 as cv
import numpy as np

from laser import LaserLineSetting, RealisticLaserGenerator


class SyntheticLaserDatasetGenerator:

    def __init__(self, dataset_directory: str):
        self.directory = dataset_directory
        self.laser_generator = RealisticLaserGenerator()

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
            for image_path in background_images:
                image = cv.imread(image_path)
                self._generate_and_save_for_all_settings(image, line_settings, intensity, subdir)

    def _prepare_subdir_for_intensity(self, intensity: int) -> str:
        # Create a separate directory for each intensity setting
        subdir = os.path.join(self.directory, f'i{intensity}')
        os.makedirs(subdir, exist_ok=True)
        return subdir

    def _generate_and_save_for_all_settings(self, image: np.ndarray,
                                            line_settings: List[LaserLineSetting],
                                            intensity: int, save_dir: str):
        for setting in line_settings:
            painting = image.copy()
            image_with_laser = self.laser_generator.draw_laser(painting, setting, intensity)
            save_path = os.path.join(save_dir, f'{setting.to_underscore_string()}.png')
            cv.imwrite(save_path, image_with_laser)

    def _generate_laser_line_settings(self) -> List[LaserLineSetting]:
        """
        Generates laser line settings for the dataset that is being created.

        Determines position of the laser, its angle, length, and width.
        """
        num_settings = 5
        img_width = 4096
        img_height = 3000
        allowed_angle_range = [-160, -20]
        allowed_laser_length = [100, 5077]

        # Determine source location pixel - probably random (no known heuristic)
        sources_x = np.random.randint(100, img_width - 100, num_settings)
        sources_y = np.random.randint(int(img_height / 2), img_height, num_settings)
        sources = np.stack([sources_x, sources_y], axis=-1)

        # Determine laser direction - no horizontal lines (not interested)
        random_angles_degs = np.random.randint(allowed_angle_range[0], allowed_angle_range[1], num_settings)
        random_angles_rads = random_angles_degs / 180 * np.pi

        # Determine length of the laser using dissipation factor and randomness
        random_laser_lengths = np.random.randint(allowed_laser_length[0], allowed_laser_length[1], num_settings)
        pixel_dissipation_factors = 1 / random_laser_lengths

        # Laser width - predefined
        max_laser_width = 16
        laser_widths = np.linspace(1, max_laser_width, num_settings).astype(int)

        settings = []
        for i in range(num_settings):
            setting = LaserLineSetting(sources[i], random_angles_rads[i],
                                       laser_widths[i], pixel_dissipation_factors[i], (img_width, img_height))
            settings.append(setting)
        return settings


if __name__ == "__main__":
    background_images = glob.glob('./dataset/testdataset/images/*')
    dataset_dir = './dataset/testdataset'
    target_intensities = (2 ** np.arange(3, 8)).tolist()

    generator = SyntheticLaserDatasetGenerator(dataset_dir)
    generator.generate_dataset(background_images, target_intensities)
