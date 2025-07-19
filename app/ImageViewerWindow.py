import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtGui import QPixmap, QIcon

class ImageViewerWindow(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(image_path))
        self.resize(600, 600)
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "SeisSnap_logo.png")
        
        self.setWindowIcon(QIcon(icon_path))
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        layout.addWidget(scroll)

        image_label = QLabel()
        pixmap = QPixmap(image_path)
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)
        scroll.setWidget(image_label)
