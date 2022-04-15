import math

import cv2
import numpy as np

from data_types import Camera, CameraOrientation, ImageSize, Line, Position
from rasterization import rasterize_line_3d


def get_cameras():
    # Camera: Prosilica GT 6400
    # https://www.alliedvision.com/en/camera-selector/detail/prosilica-gt/6400/
    # Pixel size	3.45 µm × 3.45 µm
    # Resolution: 6480 (H) × 4860 (V)
    # Lens: Kowa LM50-IR-F
    # Focal length: 50 mm?
    pixel_size = 3.45 * 1e-6
    focal_length = 10 * 1e-3
    cam1 = Camera(Position(2000, 0, 500),
                  ImageSize(6480, 4860),
                  CameraOrientation(90 - 75.07, 1.5),
                  focal_length, pixel_size)

    cam2 = Camera(Position(0, 1500, 500),
                  ImageSize(6480, 4860),
                  CameraOrientation(90 - 65.61, 1.5),
                  focal_length, pixel_size)
    # print(F"Distance betwen cameras: {get_distance(cam1.position, cam2.position)}")
    return cam1, cam2


def get_distance(pos1: Position, pos2: Position):
    return math.sqrt(math.pow(pos1.x - pos2.x, 2) +
                     math.pow(pos1.y - pos2.y, 2) +
                     math.pow(pos1.z - pos2.z, 2))


def test_camera_image(cam):
    image = np.zeros((cam.resolution.height, cam.resolution.width, 3))
    cam.display_image(image)


def project_line(cam: Camera, line: Line):
    image = np.zeros((cam.resolution.height, cam.resolution.width))
    # Translate camera's and  positions  into the origin
    # X and Z are width and height, Y is distance to the line
    translation = -cam.position.to_array()
    p1_trans = line.pos1.to_array() + translation
    p2_trans = line.pos2.to_array() + translation

    # Angles to radians
    azim_rads = cam.orientation.azimuth / 180 * np.pi
    elev_rads = cam.orientation.elevation / 180 * np.pi

    # Rotate line into the camera's view (Y axis)
    # counterclockwise
    rotation_matrix_azim = np.array([[np.cos(azim_rads), -np.sin(azim_rads), 0],
                                     [np.sin(azim_rads), np.cos(azim_rads), 0],
                                     [0, 0, 1]])
    # clockwise
    rotation_matrix_elev = np.array([[1, 0, 0],
                                     [0, np.cos(elev_rads), np.sin(elev_rads)],
                                     [0, -np.sin(elev_rads), np.cos(elev_rads)]])
    rotation_matrix = np.dot(rotation_matrix_azim, rotation_matrix_elev)
    p1 = np.dot(rotation_matrix, p1_trans)
    p2 = np.dot(rotation_matrix, p2_trans)

    # Rasterize line from p1 to p2
    points = rasterize_line_3d(Position.from_array(p1),
                               Position.from_array(p2))

    # for each point, calc projection (u, v)
    for point in points:
        # Project point onto projection plane.
        # Assumes that Y axis is the principle axis,
        # which is perpendicular to the projection plane.
        d_z = cam.focal_length / point[1]
        u = int(point[0] * d_z / cam.pixel_size)
        v = int(point[2] * d_z / cam.pixel_size)
        # (0, 0) is the center of the image
        # so move projected pixels onto a plane beginning with indices at 0
        u += int(cam.resolution.width / 2)
        v += int(cam.resolution.height / 2)
        # inverse vertical axis (0 at the bottom)
        v = cam.resolution.height - v
        if 0 < u < cam.resolution.width and 0 < v < cam.resolution.height:
            image[v, u] = 255
        # else:
        #     print(u, v, point)
    return image


def project_and_display(cam: Camera, line: Line, show: bool, save: str = None):
    mask = project_line(cam, line)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel)

    image = np.full((cam.resolution.height, cam.resolution.width, 3), fill_value=0)
    image[mask > 0] = [0, 255, 0]
    if show:
        cam.display_image(image)
    if save:
        cv2.imwrite(save, image)


if __name__ == "__main__":
    cam1, cam2 = get_cameras()
    # test_camera_image(cam1)
    # line = Line(Position(2500, 6600, 500), Position(1000, 9000, 10000))
    # line = Line(Position(2400, 4600, 500), Position(1000, 7000, 10000))
    # line = Line(Position(4000, 11000, 500), Position(4000, 7000, 10000))
    line = Line(Position(2000, 3000, 500), Position(1000, 4000, 10000))
    # line = Line(Position(3000, 1000, 500), Position(1000, 1000, 500))
    project_and_display(cam1, line, show=True, save='./data/line_0_0.png')
    project_and_display(cam2, line, show=True, save='./data/line_0_1.png')
