import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QSpinBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from numba import njit, prange
import cv2
import sys


# from joblib import Parallel, delayed

class ThreadSobel(QThread):
    changePixmap = pyqtSignal(QImage)
    updateParams = pyqtSignal(int, int)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True
        self.edges = False
        self.blur = False
        self.p_tile = False
        self.Iter = False
        self.k_mean = False
        self.adaptive = False

    def run(self):
        print('start')

        cap = cv2.VideoCapture(self.source)
        self.running = True

        while self.running:
            ret, frame = cap.read()

            if ret:
                if self.blur:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Конвертируем в оттенки серого
                    frame_blur = cv2.blur(frame_gray, (7, 7))  # Применяем размытие к оттенкам серого
                    frame[:, :, 0] = frame_blur  # Заменяeм красный канал размытым изображением
                    frame[:, :, 1] = frame_blur  # Заменяем зеленый канал размытым изображением
                    frame[:, :, 2] = frame_blur  # Заменяем синий канал размытым изображением
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.edges:
                    frame_gray = rgb_to_gray(frame)
                    edges = sobel_filter(frame_gray)
                    frame = gray_to_rgb(edges)

                if self.p_tile:
                    frame_gray = rgb_to_gray(frame)
                    edges = LoG_detection(frame_gray)
                    frame = gray_to_rgb(edges)

                if self.Iter:
                    frame_gray = rgb_to_gray(frame)
                    edges = DoG_detection(frame_gray)
                    frame = gray_to_rgb(edges)

                if self.k_mean:
                    frame_gray = rgb_to_gray(frame)
                    edges = k_means_segmentation(frame_gray)
                    frame = gray_to_rgb(edges)

                if self.adaptive:
                    frame_gray = rgb_to_gray(frame)
                    edges = adaptive_segmentation(frame_gray, self.k_size, self.T)
                    frame = gray_to_rgb(edges)

                h, w, ch = frame.shape
                bytes_per_line = ch * w

                image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                image = image.scaled(640, 480, Qt.KeepAspectRatio)

                self.changePixmap.emit(image)

        cap.release()

        print('stop')

    def stop(self):
        self.running = False


@njit(parallel=True)
def rgb_to_gray(image):
    height, width, _ = image.shape
    gray_image = np.empty((height, width), dtype=np.uint8)
    for y in prange(height):
        for x in prange(width):
            gray_value = 0.2989 * image[y, x, 0] + 0.5870 * image[y, x, 1] + 0.1140 * image[y, x, 2]
            gray_image[y, x] = int(gray_value + 0.5)
    return gray_image


@njit(parallel=True)
def gray_to_rgb(edges):
    height, width = edges.shape
    rgb_image = np.empty((height, width, 3), dtype=np.uint8)
    for y in prange(height):
        for x in prange(width):
            gray_value = edges[y, x]
            rgb_image[y, x, 0] = gray_value
            rgb_image[y, x, 1] = gray_value
            rgb_image[y, x, 2] = gray_value
    return rgb_image


def LoG_detection(image):
    _, segmented_image = cv2.threshold(image, segment_image_with_global_threshold(image), 255, cv2.THRESH_BINARY)
    return segmented_image


def segment_image_with_global_threshold(image):
    p = 0.5
    hist, _ = np.histogram(image.ravel(), bins=255, range=[0, 255])
    cumulative_hist = np.cumsum(hist)
    total_pixels = image.shape[0] * image.shape[1]
    threshold_value = np.argmax(cumulative_hist >= p * total_pixels)
    return threshold_value


def sobel_filter(image):
    edges = cv2.Canny(image, 100, 200)
    return edges.astype(np.uint8)


def DoG_detection(image):
    max_iterations = 100
    threshold = 100
    for _ in range(max_iterations):
        m1, m2 = calculate_means(image, threshold)
        new_threshold = (m1 + m2) / 2
        if abs(new_threshold - threshold) == 0:
            break
        threshold = new_threshold
    segmented_image = np.where(image < threshold, 0, 255)
    return segmented_image


def calculate_means(image, T):
    below_T = image[image < T]
    if len(below_T) == 0:
        below_T = np.array([1])
    above_T = image[image > T]
    if len(above_T) == 0:
        above_T = np.array([255])
    m1 = np.mean(below_T)
    m2 = np.mean(above_T)
    return m1, m2


def k_means_segmentation(image):
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    Z = small_image.reshape((-1, 1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(small_image.shape)
    segmented_image = cv2.resize(res2, (image.shape[1], image.shape[0]))
    return segmented_image


@njit(parallel=True)
def adaptive_segmentation(image, k_size, T):
    k_size = k_size * 2 + 1
    smoothed_image = medianBlur(image, k_size)
    thresholded_image = np.zeros_like(image)
    height, width = image.shape[:2]
    for y in prange(height):
        for x in prange(width):
            neighborhood = smoothed_image[max(0, y - k_size):min(height, y + k_size + 1),
                           max(0, x - k_size):min(width, x + k_size + 1)]
            threshold = np.mean(neighborhood)
            if image[y, x] - threshold > T:
                thresholded_image[y, x] = 255
    return thresholded_image


@njit(parallel=True)
def medianBlur(image, k_size):
    height, width = image.shape[:2]
    smoothed_image = np.zeros_like(image)
    half_kernel = k_size // 2
    for y in prange(half_kernel, height - half_kernel):
        for x in prange(half_kernel, width - half_kernel):
            neighborhood = image[y - half_kernel:y + half_kernel + 1,
                           x - half_kernel:x + half_kernel + 1]
            smoothed_image[y, x] = np.median(neighborhood)
    return smoothed_image


class Widget(QMainWindow):

    def __init__(self):
        super().__init__()

        self.thread = ThreadSobel(0)
        self.thread.changePixmap.connect(self.setImage)
        self.thread.updateParams.connect(self.updateParams)

        layout = QVBoxLayout()

        self.label_video = QLabel()
        layout.addWidget(self.label_video)

        self.btn1 = QPushButton("PLAY")
        self.btn1.clicked.connect(self.playVideo)
        layout.addWidget(self.btn1)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.clicked.connect(self.stopVideo)
        layout.addWidget(self.btn_stop)

        self.btn_edges = QPushButton("NORMAL <-> EDGES")
        self.btn_edges.clicked.connect(self.toggleEdges)
        layout.addWidget(self.btn_edges)

        self.btn_LoG = QPushButton("NORMAL <-> P_TILE")
        self.btn_LoG.clicked.connect(self.LoGVideo)
        layout.addWidget(self.btn_LoG)

        self.btn_DoG = QPushButton("NORMAL <-> ITERATIONS")
        self.btn_DoG.clicked.connect(self.DoGVideo)
        layout.addWidget(self.btn_DoG)

        self.btn_K_middle = QPushButton("NORMAL <-> K_MIDDLE")
        self.btn_K_middle.clicked.connect(self.K_middleVideo)
        layout.addWidget(self.btn_K_middle)

        self.btn_adapt = QPushButton("NORMAL <-> ADAPTIVE")
        self.btn_adapt.clicked.connect(self.adaptVideo)
        layout.addWidget(self.btn_adapt)

        self.btn_blur = QPushButton("NORMAL <-> BLURRED")
        self.btn_blur.clicked.connect(self.blurVideo)
        layout.addWidget(self.btn_blur)

        self.label1 = QLabel("Параметр К:")
        layout.addWidget(self.label1)
        self.spinBox1 = QSpinBox()
        self.spinBox1.setMinimum(1)
        self.spinBox1.setMaximum(99)
        layout.addWidget(self.spinBox1)

        self.label2 = QLabel("Параметр Т:")
        layout.addWidget(self.label2)
        self.spinBox2 = QSpinBox()
        self.spinBox2.setMinimum(0)
        self.spinBox2.setMaximum(255)
        layout.addWidget(self.spinBox2)

        self.widget = QWidget()
        self.widget.setLayout(layout)

        self.setCentralWidget(self.widget)

    def playVideo(self):
        self.thread.start()

    def stopVideo(self):
        self.thread.running = False

    def blurVideo(self):
        self.thread.blur = not self.thread.blur

    def toggleEdges(self):
        self.thread.edges = not self.thread.edges
        self.thread.p_tile = False
        self.thread.Iter = False
        self.thread.k_mean = False
        self.thread.adaptive = False


    def LoGVideo(self):
        self.thread.p_tile = not self.thread.p_tile
        self.thread.edges = False
        self.thread.Iter = False
        self.thread.k_mean = False
        self.thread.adaptive = False

    def DoGVideo(self):
        self.thread.Iter = not self.thread.Iter
        self.thread.edges = False
        self.thread.p_tile = False
        self.thread.k_mean = False
        self.thread.adaptive = False


    def K_middleVideo(self):
        self.thread.k_mean = not self.thread.k_mean
        self.thread.edges = False
        self.thread.p_tile = False
        self.thread.Iter = False
        self.thread.adaptive = False

    def adaptVideo(self):
        self.thread.adaptive = not self.thread.adaptive
        self.thread.edges = False
        self.thread.p_tile = False
        self.thread.k_mean = False
        self.thread.Iter = False
        self.thread.k_size = self.spinBox1.value()
        self.thread.T = self.spinBox2.value()

    def setImage(self, image):
        self.label_video.setPixmap(QPixmap.fromImage(image))

    def updateParams(self, k_size, T):
        self.spinBox1.setValue(k_size)
        self.spinBox2.setValue(T)


if __name__ == '__main__':
    import cv2

    app = QApplication([])
    mw = Widget()
    mw.show()
    app.exec()



