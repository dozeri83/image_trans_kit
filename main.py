import sys
import os
import cv2
import numpy as np
import json

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, \
    QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class ImageConverterThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, source_dir, target_dir, gamma, t):
        super().__init__()
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.gamma = gamma
        self.t = t
        self.is_running = True

    def run(self):
        image_files = [f for f in os.listdir(self.source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        total_files = len(image_files)

        for i, filename in enumerate(image_files):
            if not self.is_running:
                break

            source_path = os.path.join(self.source_dir, filename)
            target_path = os.path.join(self.target_dir, filename)

            image = cv2.imread(source_path)
            processed_image = self.process_image(image, self.gamma, self.t)
            cv2.imwrite(target_path, processed_image)

            self.progress.emit(int((i + 1) / total_files * 100))

        params = {"gamma": self.gamma, "inverse":self.t}
        params_path = os.path.join(self.target_dir,  "trans_params.json")
        with open(params_path, "w") as f:
            json.dump(params,f, indent=4)

        self.finished.emit()

    def process_image(self, image, gamma, t):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v_float = v.astype(np.float32) / 255.0
        v_gamma = np.power(v_float, gamma) * 255.0
        v_gamma = v_gamma.astype(np.uint8)

        v_new = t * (255 - v_gamma) + (1 - t) * v_gamma
        v_new = v_new.astype(np.uint8)

        hsv_new = cv2.merge([h, s, v_new])
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    def stop(self):
        self.is_running = False


class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Processor')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        layout.addStretch(1)

        gamma_layout = QHBoxLayout()
        gamma_label = QLabel("Gamma:")
        gamma_layout.addWidget(gamma_label)

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(0, 300)
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.update_image)
        gamma_layout.addWidget(self.gamma_slider)

        self.gamma_value_label = QLabel("γ = 1.00")
        gamma_layout.addWidget(self.gamma_value_label)

        layout.addLayout(gamma_layout)

        inverse_layout = QHBoxLayout()
        inverse_label = QLabel("Inverse:")
        inverse_layout.addWidget(inverse_label)

        self.inverse_slider = QSlider(Qt.Horizontal)
        self.inverse_slider.setRange(0, 100)
        self.inverse_slider.valueChanged.connect(self.update_image)
        inverse_layout.addWidget(self.inverse_slider)

        self.t_value_label = QLabel("t = 0.00")
        inverse_layout.addWidget(self.t_value_label)

        layout.addLayout(inverse_layout)

        button_layout = QHBoxLayout()
        load_button = QPushButton('Load Image')
        load_button.clicked.connect(self.load_image)
        button_layout.addWidget(load_button)

        save_button = QPushButton('Save Image')
        save_button.clicked.connect(self.save_image)
        button_layout.addWidget(save_button)

        layout.addLayout(button_layout)

        convert_layout = QHBoxLayout()
        self.convert_button = QPushButton('Convert Images in Dir')
        self.convert_button.clicked.connect(self.convert_images_in_dir)
        convert_layout.addWidget(self.convert_button)

        self.stop_button = QPushButton('Stop Conversion')
        self.stop_button.clicked.connect(self.stop_conversion)
        self.stop_button.setEnabled(False)
        convert_layout.addWidget(self.stop_button)

        layout.addLayout(convert_layout)

        self.progress_label = QLabel()
        layout.addWidget(self.progress_label)

        central_widget.setLayout(layout)

        self.image = None
        self.processed_image = None
        self.scaled_size = None
        self.converter_thread = None
        self.image_path = None

    def load_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if image_path:
            self.image = cv2.imread(image_path, -1)
            self.scale_image()
            self.update_image()
            self.image_path = image_path

    def scale_image(self):
        if self.image is not None:
            screen = QApplication.primaryScreen().size()
            target_width = screen.width() // 4
            target_height = screen.height() // 4

            h, w = self.image.shape[:2]
            aspect_ratio = w / h

            if aspect_ratio > 1:
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(new_height * aspect_ratio)

            self.scaled_size = (new_width, new_height)
            self.image = cv2.resize(self.image, self.scaled_size, interpolation=cv2.INTER_AREA)

    def update_image(self):
        if self.image is not None:
            gamma = self.gamma_slider.value() / 100.0
            self.gamma_value_label.setText(f"γ = {gamma:.2f}")

            t = self.inverse_slider.value() / 100.0
            self.t_value_label.setText(f"t = {t:.2f}")

            self.processed_image = self.process_image(self.image, gamma, t)
            height, width, channel = self.processed_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.processed_image.data, width, height, bytes_per_line,
                             QImage.Format_RGB888).rgbSwapped()
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def process_image(self, image, gamma, t):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v_float = v.astype(np.float32) / 255.0
        v_gamma = np.power(v_float, gamma) * 255.0
        v_gamma = v_gamma.astype(np.uint8)

        v_new = t * (255 - v_gamma) + (1 - t) * v_gamma
        v_new = v_new.astype(np.uint8)

        hsv_new = cv2.merge([h, s, v_new])
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    def save_image(self):
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
            if file_path:
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'
                cv2.imwrite(file_path, self.processed_image)

    def convert_images_in_dir(self):

        if (not self.image_path):
            QMessageBox.warning(self, "Warning",
                                "image was not loaded")
            return

        source_dir = QFileDialog.getExistingDirectory(self, "Select or Create Source Directory")
        if not source_dir:
            return

        target_dir = source_dir

        gamma = self.gamma_slider.value() / 100.0
        t = self.inverse_slider.value() / 100.0

        image_dir_name = os.path.dirname(self.image_path)

        self.converter_thread = ImageConverterThread(image_dir_name, target_dir, gamma, t)
        self.converter_thread.progress.connect(self.update_progress)
        self.converter_thread.finished.connect(self.conversion_finished)
        self.converter_thread.start()

        self.convert_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_conversion(self):
        if self.converter_thread and self.converter_thread.isRunning():
            self.converter_thread.stop()
            self.converter_thread.wait()
            self.conversion_finished()

    def update_progress(self, value):
        self.progress_label.setText(f"Conversion progress: {value}%")

    def conversion_finished(self):
        self.progress_label.setText("Conversion finished")
        self.convert_button.setEnabled(True)
        self.stop_button.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())
