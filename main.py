import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSlider,
                             QSpacerItem, QHBoxLayout, QSizePolicy,
                             QLineEdit, QPushButton, QFileDialog)
from PyQt5.QtGui import QPixmap, QFont
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from PIL import Image
from io import BytesIO
import mozjpeg_lossless_optimization
import numpy as np


def dssim(image1, image2):
    return (1 - ssim(image1, image2)) / 2

def psnr_avg(image1, image2):
    # Compute mean squared error between images
    mse = mean_squared_error(image1, image2)

    # Compute PSNR-AVG score from mean squared error
    max_value = np.iinfo(image1.dtype).max
    score = 10 * np.log10((max_value ** 2) / mse)

    return score

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window title and size
        self.setWindowTitle('SSIM-SCORE-GUI')
        self.setFixedSize(1080, 1080)

        # Set central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QGridLayout()
        central_widget.setLayout(layout)

        # Create label for displaying drop instructions
        self.drop_label = QLabel('Drop An Image')
        self.drop_label.setAlignment(QtCore.Qt.AlignCenter)
        font = QFont()
        font.setPointSize(24)
        self.drop_label.setFont(font)
        layout.addWidget(self.drop_label, 1, 0, 1, 2)

        # Create label for displaying image
        self.image_label = QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.image_label, 1, 0, 2, 2)

        # Create label for displaying zoomed-in view of image
        self.zoomed_image_label = QLabel()
        self.zoomed_image_label.setFixedSize(400, 400)
        layout.addWidget(self.zoomed_image_label, 1, 0, 2, 2, QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)

        # Create labels and input fields for SSIM, DSSIM, and PSNR values
        metrics = ['SSIM', 'DSSIM', 'PSNR_AVG']
        for i, metric in enumerate(metrics):
            # Create a horizontal layout for the label and input field
            row_layout = QHBoxLayout()

            row_layout.addSpacing(280)
            # Create label for displaying metric score
            label = QLabel()
            font = QFont()
            font.setPointSize(24)
            label.setFont(font)
            row_layout.addWidget(label)
            setattr(self, f'{metric.lower()}_label', label)

            # Add a small spacer item between the label and input field
            row_layout.addSpacing(10)

            # Add a fixed-size spacer item to the left of the DSSIM input field
            if metric == 'DSSIM':
                row_layout.addSpacing(105)

            # Create input field for user to enter desired metric value
            input_field = QLineEdit()
            row_layout.addWidget(input_field, alignment=QtCore.Qt.AlignRight)
            input_field.setMaximumWidth(100)
            input_field.hide()
            setattr(self, f'{metric.lower()}_input', input_field)

            # Add a stretchable spacer item to the right of the input field
            row_layout.addStretch()

            # Add the horizontal layout to the main layout
            layout.addLayout(row_layout, 3 + i, 0, 1, 2)

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Add a slider to adjust JPEG quality
        self.quality_slider = QSlider(QtCore.Qt.Horizontal)
        self.quality_slider.setMinimum(0)
        self.quality_slider.setMaximum(100)
        self.quality_slider.setValue(90)
        self.quality_slider.setTickPosition(QSlider.TicksBelow)
        self.quality_slider.setTickInterval(1)

        # Create a nested layout for the slider
        slider_layout = QHBoxLayout()
        slider_layout.addItem(QSpacerItem(100, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))
        slider_layout.addWidget(self.quality_slider)
        slider_layout.addItem(QSpacerItem(100, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        # Add the nested layout to the main layout
        layout.addLayout(slider_layout, 6, 0, 1, 2)

        # Connect slider valueChanged signal to update_image slot
        self.quality_slider.valueChanged.connect(self.update_image)
        self.quality_slider.hide()

        # Create a horizontal layout for the quality and file size labels
        label_layout = QHBoxLayout()
        label_layout.setSpacing(100)

        # Add a spacer item to the left of the quality label
        label_layout.addStretch()

        # Create label for displaying current quality value
        self.quality_label = QLabel(f'Quality: {self.quality_slider.value()}')
        label_layout.addWidget(self.quality_label)
        self.quality_label.hide()

        # Create label for displaying estimated final file size
        self.file_size_label = QLabel()
        label_layout.addWidget(self.file_size_label)
        self.file_size_label.hide()

        # Add the horizontal layout to the main layout
        layout.addLayout(label_layout, 7, 0, 1, 2)

        # Add a spacer item to the left of the quality label
        label_layout.addStretch()

        self.setStyleSheet("""
            QLabel#Title {
                font-size: 48px;
                font-weight: bold;
            }
        """)
        self.drop_label.setObjectName("Title")
        layout.setVerticalSpacing(10)
        # layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 8)
        layout.setRowStretch(2, 1)
        layout.setRowStretch(3, 1)
        layout.setRowStretch(4, 1)
        layout.setRowStretch(5, 1)

        self.quality_slider.valueChanged.connect(self.update_zoomed_image)

        # Add an attribute to store the position of the last mouse click within the original image
        self.last_click_pos = None

        # Create "Update Quality" button
        self.update_quality_button = QtWidgets.QPushButton('Optimize')
        self.update_quality_button.setMinimumSize(200, 40)
        self.update_quality_button.setMaximumSize(200, 40)
        layout.addWidget(self.update_quality_button, 8, 0, 1, 2, alignment=QtCore.Qt.AlignLeft)
        self.update_quality_button.hide()

        # Create export button
        self.export_button = QPushButton('Export Image')
        self.export_button.setMinimumSize(200, 40)
        self.export_button.setMaximumSize(200, 40)
        layout.addWidget(self.export_button, 8, 0, 1, 2, alignment=QtCore.Qt.AlignRight)
        self.export_button.clicked.connect(self.export_image)
        self.export_button.hide()

        # Connect button clicked signal to update_quality slot
        self.update_quality_button.clicked.connect(self.update_quality)
        # Set validators for input fields
        self.ssim_input.setValidator(InputValidator(0, 1))
        self.dssim_input.setValidator(InputValidator(0, 1))
        self.psnr_avg_input.setValidator(InputValidator(0, float('inf')))

        # Remove default title bar
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        # Create custom title bar
        title_bar = QHBoxLayout()
        title_bar.setContentsMargins(0, 0, 0, 0)

        # Add window title label
        window_title = QLabel('Check Image Quality')
        window_title.setStyleSheet("""
            QLabel {
                color: #8be9fd;
                font-family: "Fira Code", monospace;
            }
        """)
        title_bar.addWidget(window_title)

        # Add spacer item to push buttons to right side of title bar
        title_bar.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Add minimize button
        minimize_button = QPushButton('-')
        minimize_button.setFixedSize(30, 30)
        minimize_button.setStyleSheet("""
            QPushButton {
                background-color: #44475a;
                color: #8be9fd;
                border: none;
            }
            QPushButton:hover {
                background-color: #6272a4;
            }
        """)
        minimize_button.clicked.connect(self.showMinimized)
        title_bar.addWidget(minimize_button)

        # Add maximize button
        maximize_button = QPushButton('+')
        maximize_button.setFixedSize(30, 30)
        maximize_button.setStyleSheet("""
            QPushButton {
                background-color: #44475a;
                color: #8be9fd;
                border: none;
            }
            QPushButton:hover {
                background-color: #6272a4;
            }
        """)
        maximize_button.clicked.connect(self.showMaximized)
        title_bar.addWidget(maximize_button)

        # Add close button
        close_button = QPushButton('x')
        close_button.setFixedSize(30, 30)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #ff5555;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #ff6e6e;
            }
        """)
        close_button.clicked.connect(self.close)
        title_bar.addWidget(close_button)

        # Add custom title bar to main layout
        layout.addLayout(title_bar, 0, 0, 1, 2)

        # Add attribute to store position of mouse cursor when user clicks on custom title bar
        self.title_bar_mouse_pos = None

    def update_quality_from_ssim(self):
        # Parse text entered by user to obtain desired SSIM value
        try:
            desired_ssim = float(self.ssim_input.text())
        except ValueError:
            return

        # Define function to compute SSIM score for given quality value
        def compute_ssim(quality):
            # Convert input image to JPEG with current quality
            with BytesIO() as buffer:
                self.input_image.save(buffer, format='JPEG', quality=quality)
                input_data = buffer.getvalue()

            # Perform lossless optimization using mozjpeg_lossless_optimization
            output_data = mozjpeg_lossless_optimization.optimize(input_data)

            # Load output image and convert to grayscale
            output_image = Image.open(BytesIO(output_data)).convert('L')
            output_array = np.array(output_image)

            # Compute and return SSIM score between original and compressed images
            return ssim(np.array(self.input_image.convert('L')), output_array)

        # Use binary search to find value of quality slider that achieves desired SSIM value
        min_quality = 0
        max_quality = 100
        while min_quality <= max_quality:
            mid_quality = (min_quality + max_quality) // 2
            ssim_score = compute_ssim(mid_quality)
            if ssim_score < desired_ssim:
                max_quality = mid_quality - 1
            elif ssim_score > desired_ssim:
                min_quality = mid_quality + 1
            else:
                break

        # Update value of quality slider with computed value
        self.quality_slider.setValue(mid_quality)

    def update_quality_from_dssim(self):
        # Parse text entered by user to obtain desired DSSIM value
        try:
            desired_dssim = float(self.dssim_input.text())
        except ValueError:
            return

        # Define function to compute DSSIM score for given quality value
        def compute_dssim(quality):
            # Convert input image to JPEG with current quality
            with BytesIO() as buffer:
                self.input_image.save(buffer, format='JPEG', quality=quality)
                input_data = buffer.getvalue()

            # Perform lossless optimization using mozjpeg_lossless_optimization
            output_data = mozjpeg_lossless_optimization.optimize(input_data)

            # Load output image and convert to grayscale
            output_image = Image.open(BytesIO(output_data)).convert('L')
            output_array = np.array(output_image)

            # Compute and return DSSIM score between original and compressed images
            return dssim(np.array(self.input_image.convert('L')), output_array)

        # Use binary search to find value of quality slider that achieves desired DSSIM value
        min_quality = 0
        max_quality = 100
        while min_quality <= max_quality:
            mid_quality = (min_quality + max_quality) // 2
            dssim_score = compute_dssim(mid_quality)
            if dssim_score < desired_dssim:
                max_quality = mid_quality - 1
            elif dssim_score > desired_dssim:
                min_quality = mid_quality + 1
            else:
                break

        # Update value of quality slider with computed value
        self.quality_slider.setValue(mid_quality)

    def update_quality_from_psnr_avg(self):
        # Parse text entered by user to obtain desired PSNR-AVG value
        try:
            desired_psnr_avg = float(self.psnr_avg_input.text())
        except ValueError:
            return

        # Define function to compute PSNR-AVG score for given quality value
        def compute_psnr_avg(quality):
            # Convert input image to JPEG with current quality
            with BytesIO() as buffer:
                self.input_image.save(buffer, format='JPEG', quality=quality)
                input_data = buffer.getvalue()

            # Perform lossless optimization using mozjpeg_lossless_optimization
            output_data = mozjpeg_lossless_optimization.optimize(input_data)

            # Load output image and convert to grayscale
            output_image = Image.open(BytesIO(output_data)).convert('L')
            output_array = np.array(output_image)

            # Compute and return PSNR-AVG score between original and compressed images
            return psnr_avg(np.array(self.input_image.convert('L')), output_array)

        # Use binary search to find value of quality slider that achieves desired PSNR-AVG value
        min_quality = 0
        max_quality = 100
        while min_quality <= max_quality:
            mid_quality = (min_quality + max_quality) // 2
            psnr_avg_score = compute_psnr_avg(mid_quality)
            if psnr_avg_score < desired_psnr_avg:
                min_quality = mid_quality + 1
            elif psnr_avg_score > desired_psnr_avg:
                max_quality = mid_quality - 1
            else:
                break

        # Update value of quality slider with computed value
        self.quality_slider.setValue(mid_quality)

    def dropEvent(self, event):
        # Get file path of dropped image
        self.file_path = event.mimeData().urls()[0].toLocalFile()

        # Load and store full resolution version of original image
        self.input_image = Image.open(self.file_path)

        # Convert input image to JPEG with current quality
        quality = self.quality_slider.value()
        with BytesIO() as buffer:
            self.input_image.save(buffer, format='JPEG', quality=quality)
            input_data = buffer.getvalue()

        # Perform lossless optimization using mozjpeg_lossless_optimization
        output_data = mozjpeg_lossless_optimization.optimize(input_data)

        # Load output image
        output_image = Image.open(BytesIO(output_data))
        output_array = np.array(output_image.convert('L'))

        # Compute SSIM score between original and compressed images
        ssim_score = ssim(np.array(self.input_image.convert('L')), output_array)

        # Compute DSSIM score between original and compressed images
        dssim_score = dssim(np.array(self.input_image.convert('L')), output_array)

        # Compute PSNR-AVG score between original and compressed images
        psnr_avg_score = psnr_avg(np.array(self.input_image.convert('L')), output_array)

        # Extract region of image at desired zoom level
        zoom_level = 10
        x, y = output_image.size
        zoomed_region = output_image.crop((0, 0, x / zoom_level, y / zoom_level))

        # Scale zoomed region to fit size of zoomed_image_label
        data = zoomed_region.tobytes('raw', 'RGB')
        qim = QtGui.QImage(data, zoomed_region.size[0], zoomed_region.size[1], QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qim).scaled(self.zoomed_image_label.size(), QtCore.Qt.KeepAspectRatio)
        # Display zoomed region in zoomed_image_label
        self.zoomed_image_label.setPixmap(pixmap)

        # Hide drop label and display image and scores
        self.drop_label.hide()
        data = output_image.tobytes('raw', 'RGB')
        qim = QtGui.QImage(data, output_image.size[0], output_image.size[1], QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qim).scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.ssim_label.setText(f'SSIM: {ssim_score:.2f}')
        self.dssim_label.setText(f'DSSIM: {dssim_score:.2f}')
        self.psnr_avg_label.setText(f'PSNR-AVG: {psnr_avg_score:.2f}')
        self.quality_slider.show()
        self.quality_label.show()
        self.quality_label.setText(f'Quality: {quality}')
        self.file_size_label.show()
        file_size = len(output_data) / 1024
        self.file_size_label.setText(f'File Size: {file_size:.2f} KB')

        # self.ssim_input.show()
        self.dssim_input.show()
        self.psnr_avg_input.show()
        self.update_quality_button.show()
        self.export_button.show()

    def update_image(self):
        # Check if input_image attribute exists
        # if not hasattr(self, 'input_image'):
        #     return

        # Get current quality value from slider
        quality = self.quality_slider.value()

        # Convert input image to JPEG with current quality
        with BytesIO() as buffer:
            self.input_image.save(buffer, format='JPEG', quality=quality)
            input_data = buffer.getvalue()

        # Perform lossless optimization using mozjpeg_lossless_optimization
        output_data = mozjpeg_lossless_optimization.optimize(input_data)

        # Load output image and convert to grayscale
        output_image = Image.open(BytesIO(output_data)).convert('L')
        output_array = np.array(output_image)

        # Compute SSIM score between original and compressed images
        ssim_score = ssim(np.array(self.input_image.convert('L')), output_array)

        # Compute DSSIM score between original and compressed images
        dssim_score = dssim(np.array(self.input_image.convert('L')), output_array)

        # Compute PSNR-AVG score between original and compressed images
        psnr_avg_score = psnr_avg(np.array(self.input_image.convert('L')), output_array)

        # Update image and scores
        data = Image.open(BytesIO(output_data)).tobytes('raw', 'RGB')
        qim = QtGui.QImage(data, self.input_image.size[0], self.input_image.size[1], QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qim).scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.ssim_label.setText(f'SSIM: {ssim_score:.2f}')
        self.dssim_label.setText(f'DSSIM: {dssim_score:.2f}')
        self.psnr_avg_label.setText(f'PSNR-AVG: {psnr_avg_score:.2f}')
        self.quality_label.setText(f'Quality: {quality}')

        # Update file size label
        self.file_size_label.show()
        file_size = len(output_data) / 1024
        self.file_size_label.setText(f'Estimated File Size: {file_size:.2f} KB')

        # Update zoomed-in image
        self.update_zoomed_image()

    def update_zoomed_image(self):
        # Check if last_click_pos attribute exists
        if not hasattr(self, 'last_click_pos') or self.last_click_pos is None:
            return

        # Get the current value of the quality slider
        quality = self.quality_slider.value()

        # Compress the original image using the current quality value
        compressed_image = self.compress_image(self.input_image, quality)

        # Extract region of image at desired zoom level around last clicked position
        zoom_level = 10
        x_img, y_img = self.last_click_pos
        x1 = max(0, x_img - compressed_image.width // (2 * zoom_level))
        y1 = max(0, y_img - compressed_image.height // (2 * zoom_level))
        x2 = min(compressed_image.width, x_img + compressed_image.width // (2 * zoom_level))
        y2 = min(compressed_image.height, y_img + compressed_image.height // (2 * zoom_level))
        zoomed_region = compressed_image.crop((x1, y1, x2, y2))

        # Scale zoomed region to fit size of zoomed_image_label
        data = zoomed_region.tobytes('raw', 'RGB')
        qim = QtGui.QImage(data, zoomed_region.size[0], zoomed_region.size[1], QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qim).scaled(self.zoomed_image_label.size(), QtCore.Qt.KeepAspectRatio)

        # Update the zoomed-in image label with the zoomed region
        self.zoomed_image_label.setPixmap(pixmap)

    def compress_image(self, image, quality):
        # Check if the input image has an alpha channel
        if image.mode == 'RGBA':
            # Create a white background image
            background = Image.new('RGB', image.size, (255, 255, 255))

            # Composite the input image onto the white background
            image = Image.alpha_composite(background, image.convert('RGBA'))

            # Convert input image to RGB mode
            # image = image.convert('RGB')

        # Convert input image to JPEG with current quality
        with BytesIO() as buffer:
            image.save(buffer, format='JPEG', quality=quality)
            input_data = buffer.getvalue()

        # Perform lossless optimization using mozjpeg_lossless_optimization
        output_data = mozjpeg_lossless_optimization.optimize(input_data)

        # Load and return output image
        return Image.open(BytesIO(output_data))

    def dragEnterEvent(self, event):
        # Accept drag event if it contains an image file
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].isLocalFile():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.endswith(('.png', '.jpg', '.jpeg')):
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        # Check if the mouse click was within the bounds of the image label
        if self.image_label.geometry().contains(event.pos()):
            # Map the mouse position from global coordinates to image label coordinates
            label_pos = self.image_label.mapFromGlobal(event.globalPos())

            # Compute the relative position of the mouse within the image label
            x_rel = label_pos.x() / self.image_label.width()
            y_rel = label_pos.y() / self.image_label.height()

            # Compute and store the corresponding position within the original image
            x_img = int(x_rel * self.input_image.width)
            y_img = int(y_rel * self.input_image.height)
            self.last_click_pos = (x_img, y_img)

            # Update zoomed-in image
            self.update_zoomed_image()
        elif event.pos().y() < 30:
            # Store position of mouse cursor
            self.title_bar_mouse_pos = event.pos()
        else:
            # Reset position of mouse cursor
            self.title_bar_mouse_pos = None

    def export_image(self):
        # Get current quality value
        quality = self.quality_slider.value()

        # Convert input image to JPEG with current quality
        with BytesIO() as buffer:
            self.input_image.save(buffer, format='JPEG', quality=quality)
            input_data = buffer.getvalue()

        # Perform lossless optimization using mozjpeg_lossless_optimization
        output_data = mozjpeg_lossless_optimization.optimize(input_data)

        # Create file dialog for selecting export location
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter('JPEG Image (*.jpg *.jpeg)')
        if file_dialog.exec_():
            # Get selected file path
            file_path = file_dialog.selectedFiles()[0]

            # Save optimized image data to selected file path
            with open(file_path, 'wb') as f:
                f.write(output_data)

    def mouseMoveEvent(self, event):
        # Check if user is dragging custom title bar
        if self.title_bar_mouse_pos is not None:
            # Move window
            delta = event.pos() - self.title_bar_mouse_pos
            self.move(self.pos() + delta)

    def update_quality(self):
        # Check if SSIM input field contains a valid value
        try:
            float(self.ssim_input.text())
            self.update_quality_from_ssim()
            return
        except ValueError:
            pass

        # Check if DSSIM input field contains a valid value
        try:
            float(self.dssim_input.text())
            self.update_quality_from_dssim()
            return
        except ValueError:
            pass

        # Check if PSNR-AVG input field contains a valid value
        try:
            float(self.psnr_avg_input.text())
            self.update_quality_from_psnr_avg()
            return
        except ValueError:
            pass

# Create a custom QValidator class for validating input values
class InputValidator(QtGui.QValidator):
    def __init__(self, min_value, max_value, parent=None):
        super().__init__(parent)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, text, pos):
        try:
            value = float(text)
            if self.min_value <= value <= self.max_value:
                return QtGui.QValidator.Acceptable, text, pos
            else:
                return QtGui.QValidator.Intermediate, text, pos
        except ValueError:
            if text == '':
                return QtGui.QValidator.Intermediate, text, pos
            else:
                return QtGui.QValidator.Invalid, text, pos

    def fixup(self, text):
        try:
            value = float(text)
            if value < self.min_value:
                return str(self.min_value)
            elif value > self.max_value:
                return str(self.max_value)
            else:
                return text
        except ValueError:
            return ''

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Load style sheet from file
    with open('style.qss', 'r') as f:
        style = f.read()

    # Apply style sheet to application
    app.setStyleSheet(style)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
