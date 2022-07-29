from typing import Callable, Tuple

import cv2
import subprocess


def get_detection_algorithm(executable_path: str, pixelCountFilePath: str,
                            filterSize: int, slopeThreshold: float, minPixelsThreshold: int) -> Callable[[str], Tuple]:
    def detect_using_rotation(file_path: str) -> Tuple:
        image_bytes = cv2.imread(file_path).tobytes()

        params = f'{executable_path} --filterSize {filterSize} --slopeThreshold {slopeThreshold}' \
                 f' --minPixelsThreshold {minPixelsThreshold} ' \
                 f'--pixelCountFile {pixelCountFilePath}'
        proc = subprocess.Popen(params, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        output = proc.communicate(input=image_bytes)[0]
        #proc.wait()
        #output = proc.stdout.readline()
        line = output.decode('utf-8')
        string_numbers = line.split(',')
        int_numbers = [int(number) for number in string_numbers]
        return int_numbers[:2], int_numbers[2:]

    return detect_using_rotation