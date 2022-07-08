import cv2
import numpy as np

from src.localization.geometry import line_end_points_on_image


class BoundingBox:

    def __init__(self, x_begin, x_end, y_begin, y_end):
        self.x_begin = x_begin
        self.x_end = x_end
        self.y_begin = y_begin
        self.y_end = y_end

    def get_width(self):
        return self.x_end - self.x_begin

    def get_height(self):
        return self.y_end - self.y_begin


def display_frame(frame_number: int):
    cap = cv2.VideoCapture("dataset/S1580002.MP4")

    # get total number of frames
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # check for valid frame number
    if frame_number >= 0 and frame_number <= total_frames:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        cv2.imshow("video", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


class VideoWrapper:

    def __init__(self, file_path):
        self.file_path = file_path

    def get_frames(self, first_frame_idx: int, num_frames: int, box: BoundingBox) -> np.ndarray:
        cap = cv2.VideoCapture(self.file_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = box.get_width()
        height = box.get_height()

        # check for valid frame number
        if first_frame_idx >= 0 and first_frame_idx + num_frames < total_frames:
            # set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)

        frames = np.empty((num_frames, height, width, 1))
        for i in range(num_frames):
            ret, frame = cap.read()
            frames[i] = frame[box.y_begin:box.y_end, box.x_begin:box.x_end, 1:2]

        cap.release()
        return frames


def display_image(image: np.ndarray):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    while cv2.waitKey(0) & 0xFF != 27:
        pass
    cv2.destroyWindow('image')


def image_summing(frames: np.ndarray):
    frames_sum = np.sum(frames, axis=0)
    frames_mean = frames_sum / np.shape(frames)[0]
    return frames_mean


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def zuran():
    box_left = BoundingBox(0, 1000, 700, 1200)
    box_center = BoundingBox(960, 2880, 700, 1200)
    box_center_focused = BoundingBox(1500, 2400, 700, 1200)
    box_right = BoundingBox(2800, 3800, 700, 1200)
    video = VideoWrapper("dataset/S1140003.MP4")

    summed_frames = []
    step = 200
    for i in range(0, 5000, step):
        frames = video.get_frames(i, step, box_center_focused)
        frames_mean = image_summing(frames)
        summed_frames.append(frames_mean)

        frame = frames_mean.astype(np.uint8)
        # equalized = cv2.equalizeHist(frame)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        equalized = clahe.apply(frame)

        display_image(equalized)

    result = image_summing(np.stack(summed_frames, axis=0))

    # num_frames = 200
    # frames = video.get_frames(4988, num_frames, box)
    # result = image_summing(frames / num_frames)

    # display_image(result / 255.)

    frame = result.astype(np.uint8)
    # equalized = cv2.equalizeHist(frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    equalized = clahe.apply(frame)

    display_image(equalized)
    #
    # sobelx = cv2.Sobel(result, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(result, cv2.CV_64F, 0, 1, ksize=3)
    # sobelx[sobelx < 0] = 0
    # sobely[sobely < 0] = 0
    # gradient = sobelx + sobely
    #
    # # small_changes_highlighted = 50 / sobelx
    # # small_changes_highlighted[np.isinf(small_changes_highlighted)] = 0
    # for sigma in range(1, 11):
    #     print("Sigma: ", sigma)
    #     highlights = gaussian(sobelx, sigma, 0.25)
    #     highlights = highlights / np.max(highlights) * 255
    #     display_image(highlights.astype(result.dtype))
    #
    # new_color_channel = np.zeros_like(result)
    # image = np.concatenate([new_color_channel, result, new_color_channel], axis=-1)
    # display_image(image)


def fit():
    box = BoundingBox(0, 1920, 0, 1080)
    video = VideoWrapper("dataset/DSC_2747.MOV")

    summed_frames = []
    step = 10
    for i in range(0, 50, step):
        frames = video.get_frames(i, step, box)
        # display_image(frames[10] / 255.)
        result = image_summing(frames)

        frame = (result).astype(np.uint8)
        equalized = cv2.equalizeHist(frame)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
        # cl1 = clahe.apply(frame)
        display_image(equalized)
        summed_frames.append(equalized)

    result = image_summing(np.stack(summed_frames, axis=0))
    result = (result / len(summed_frames)).astype(np.uint8)
    equalized = cv2.equalizeHist(result)[..., np.newaxis]

    display_image(equalized)


def fit_image():
    img = cv2.imread('dataset/fit/DSC_2749.JPG')
    green = img[..., 1]
    green[green > 20] = 0
    green_gauss = gaussian(green, 16, 10)
    green_gauss /= np.max(green_gauss)

    # display_image(green)

    morphed_green = cv2.morphologyEx(green_gauss, cv2.MORPH_OPEN, (5, 5))

    # display_image(morphed_green)

    morphed_uint8 = (morphed_green * 255).astype(np.uint8)
    # ret, binary = cv2.threshold(morphed_green, 0.5, 1, cv2.THRESH_BINARY)
    # binary = binary.astype(np.uint8)
    blurred = cv2.GaussianBlur(morphed_uint8, ksize=(9, 9), sigmaX=9)
    # display_image(blurred)

    edges = cv2.Canny(blurred, 100, 150, apertureSize=3)
    display_image(edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    for rho, theta in lines[0]:
        end_points = line_end_points_on_image(rho, theta, img.shape[:2])
        (x1, y1), (x2, y2) = end_points

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    display_image(img)

    # Gradients are used by Canny.
    # ksize = 43
    # blurred_green_gauss = cv2.GaussianBlur(morphed_green, ksize=(ksize, ksize), sigmaX=19)
    #
    # green_gauss_int = (blurred_green_gauss * 255).astype(green.dtype)
    # ksize = 15
    # sobelx = cv2.Sobel(green_gauss_int, cv2.CV_32F, 1, 0, ksize=ksize)#, scale=2**(2+1-ksize*2))
    # sobely = cv2.Sobel(green_gauss_int, cv2.CV_32F, 0, 1, ksize=ksize)#, scale=2**(2+1-ksize*2))
    #
    # sobelx = np.abs(sobelx)
    # sobely = np.abs(sobely)
    # gradient = sobelx + sobely
    # gradient = gradient / np.max(gradient)

    # display_image(gradient / np.max(gradient))


zuran()
