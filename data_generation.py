import cv2
import numpy as np

from data_types import Camera, CameraOrientation, ImageSize, Line, Position
from geometry import is_point_within_image, rot_along_x_matrix, rot_along_z_matrix, translation_matrix, get_distance
from rasterization import rasterize_line_3d


def get_cameras():
    # Camera: Prosilica GT 6400
    # https://www.alliedvision.com/en/camera-selector/detail/prosilica-gt/6400/
    # Pixel size	3.45 µm × 3.45 µm
    # Resolution: 6480 (H) × 4860 (V)
    # Lens: Kowa LM50-IR-F
    # Focal length: 50 mm?
    pixel_size = 3.45 * 1e-6
    focal_length = 50 * 1e-3
    elevation = 8
    cam1 = Camera(Position(2000, 0, 500),
                  ImageSize(6480, 4860),
                  CameraOrientation(90 - 75.07, elevation),
                  focal_length, pixel_size)

    cam2 = Camera(Position(0, 1500, 500),
                  ImageSize(6480, 4860),
                  CameraOrientation(90 - 65.61, elevation),
                  focal_length, pixel_size)
    # print(F"Distance betwen cameras: {get_distance(cam1.position, cam2.position)}")
    return cam1, cam2


# def get_cameras():
#     # Camera: Prosilica GT 6400
#     # https://www.alliedvision.com/en/camera-selector/detail/prosilica-gt/6400/
#     # Pixel size	3.45 µm × 3.45 µm
#     # Resolution: 6480 (H) × 4860 (V)
#     # Lens: Kowa LM50-IR-F
#     # Focal length: 50 mm?
#     pixel_size = 3.45 * 1e-6
#     focal_length = 20 * 1e-3
#     elevation = 30
#     cam1 = Camera(Position(-1500, 0, 500),
#                   ImageSize(6480, 4860),
#                   CameraOrientation(0, elevation),
#                   focal_length, pixel_size)
#
#     cam2 = Camera(Position(1500, 0, 500),
#                   ImageSize(6480, 4860),
#                   CameraOrientation(0, elevation),
#                   focal_length, pixel_size)
#     # print(F"Distance betwen cameras: {get_distance(cam1.position, cam2.position)}")
#     return cam1, cam2


def transform_camera_to_world_matrix(translation_vec: np.ndarray, azim_degs: float, elev_degs: float):
    trans_matrix = translation_matrix(translation_vec)

    # Rotate line the camera's principal axis into the world coordinates
    rotation_matrix_azim = rot_along_z_matrix(azim_degs)
    rotation_matrix_elev = rot_along_x_matrix(elev_degs)

    # The order of matrix multiplication matters!
    transformation_matrix = trans_matrix @ rotation_matrix_azim @ rotation_matrix_elev
    return transformation_matrix


def transform_world_to_camera(point1, point2, cam: Camera):
    camera_to_world = transform_camera_to_world_matrix(cam.position.to_array(),
                                                       cam.orientation.azimuth,
                                                       cam.orientation.elevation)
    world_to_camera = np.linalg.inv(camera_to_world)

    p1_orig = np.concatenate([point1, [1]], axis=-1)
    p2_orig = np.concatenate([point2, [1]], axis=-1)

    p1_transformed = np.dot(world_to_camera, p1_orig)
    p2_transformed = np.dot(world_to_camera, p2_orig)

    return p1_transformed, p2_transformed


def project_point(point, cam: Camera):
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
    return u, v


def project_line(cam: Camera, line: Line):
    image = np.zeros((cam.resolution.height, cam.resolution.width))

    p1, p2 = transform_world_to_camera(line.pos1.to_array(), line.pos2.to_array(), cam)

    # Rasterize line from p1 to p2
    points = rasterize_line_3d(Position.from_array(p1),
                               Position.from_array(p2))

    # for each point, calc projection (u, v)
    for point in points:
        # Project point onto projection plane.
        u, v = project_point(point, cam)
        # Check image boundaries
        if is_point_within_image(u, v, cam.resolution.to_tuple()):
            image[v, u] = 255
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
    line = Line(Position(2800, 6600, 500), Position(2400, 8400, 10000))  # Line at an angle
    # line = Line(Position(2800, 6600, 500), Position(2800, 6600, 10000))  # Vertical line
    # line = Line(Position(1600, 3200, 500), Position(1600, 3200, 10000))  # Vertical line a bit closer
    # line = Line(Position(0, 3000, -2500), Position(3000, 3000, -2500)) # Horizontal line
    # line = Line(Position(0, 3000, 500), Position(0, 3000, 10500))  # Vertical line on Y axis
    # line = Line(Position(2400, 4600, 500), Position(1000, 7000, 10000))
    # line = Line(Position(4000, 11000, 500), Position(4000, 7000, 10000))
    # line = Line(Position(2000, 3000, 500), Position(1000, 4000, 10000))
    # line = Line(Position(3000, 6000, 500), Position(1000, 6000, 500))

    cams_distance = get_distance(cam1.position.to_array(), cam2.position.to_array())
    line_cam1_distance = get_distance(cam1.position.to_array(), line.pos1.to_array())
    line_cam2_distance = get_distance(cam2.position.to_array(), line.pos1.to_array())
    print(F"Distance between cameras: {cams_distance}")
    print(F"Distance the laser's origin and cam1: {line_cam1_distance}")
    print(F"Distance the laser's origin and cam2: {line_cam2_distance}")

    project_and_display(cam1, line, show=True, save='./data/line_0_0.png')
    project_and_display(cam2, line, show=True, save='./data/line_0_1.png')
