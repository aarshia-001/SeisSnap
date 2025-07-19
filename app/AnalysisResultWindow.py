import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QListWidget, QMessageBox
)
from PySide6.QtCore import Qt
from CSVeditorDialogue import CsvEditorDialog
from ImageViewerWindow import ImageViewerWindow

from PySide6.QtGui import QIcon
class AnalysisResultWindow(QWidget):
    """Lists files inside <project>/analysisResult and opens each in its own
    top‑level window (multiple can stay open simultaneously)."""
    def __init__(self, directory: str, parent=None):
        super().__init__(parent)
        self.directory = directory
        self.setWindowTitle("📊 Analysis Results")
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "SeisSnap_logo.png")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(400, 500)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # Keeping references so child windows aren’t
        self._children: list[QWidget] = []

        layout = QVBoxLayout(self)

        header = QLabel("📁 Files in analysisResult:")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)

        self.file_list = QListWidget()
        self.populate_file_list()
        self.file_list.itemClicked.connect(self.open_file)
        layout.addWidget(self.file_list)

    # ---------- helpers -----------------------------------------------------

    def populate_file_list(self) -> None:
        self.file_list.clear()
        for f in os.listdir(self.directory):
            self.file_list.addItem(f)

    def _register_child(self, win: QWidget) -> None:
        """Track child windows so Python GC doesn’t close them prematurely."""
        self._children.append(win)
        win.destroyed.connect(lambda: self._children.remove(win))

    # ---------- slots -------------------------------------------------------

    def open_file(self, item) -> None:
        file_name = item.text()
        file_path = os.path.join(self.directory, file_name)

        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Error",
                                f"The file '{file_name}' no longer exists.")
            self.populate_file_list()
            return

        ext = os.path.splitext(file_name)[1].lower()

        if ext == ".csv":
            # Non‑modal CSV editor
            dlg = CsvEditorDialog(file_path)          # <-- no parent
            dlg.setAttribute(Qt.WA_DeleteOnClose)
            dlg.show()                                # non‑blocking
            self._register_child(dlg)

        elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
            # Non‑modal image viewer
            viewer = ImageViewerWindow(file_path)     # <-- no parent
            viewer.setAttribute(Qt.WA_DeleteOnClose)
            viewer.show()
            self._register_child(viewer)

        else:
            QMessageBox.information(self, "Unsupported",
                                     f"File type '{ext}' is not supported.")
