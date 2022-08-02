import re
import subprocess
from typing import Callable, List, Tuple

import cv2


def get_detection_algorithm(executable_path: str, pixelCountFilePath: str,
                            filterSize: int, slopeThreshold: float, minPixelsThreshold: int,
                            candidates: int) -> Callable[[str], List[Tuple]]:
    def detect_using_rotation(file_path: str) -> List[Tuple]:
        image_bytes = cv2.imread(file_path).tobytes()

        params = f'{executable_path} --filterSize {filterSize} --slopeThreshold {slopeThreshold}' \
                 f' --minPixelsThreshold {minPixelsThreshold} ' \
                 f'--pixelCountFile {pixelCountFilePath} ' \
                 f'--candidates {candidates}'
        proc = subprocess.Popen(params, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        output = proc.communicate(input=image_bytes)[0]
        lines = output.decode('utf-8').split('\n')
        lines = filter(lambda x: not re.match(r'^\s*$', x), lines) # Remove empty lines

        selected_candidates = []
        for line in lines:
            string_numbers = line.split(',')
            int_numbers = [int(number) for number in string_numbers]
            selected_candidates.append((int_numbers[:2], int_numbers[2:]))
        return selected_candidates

    return detect_using_rotation
