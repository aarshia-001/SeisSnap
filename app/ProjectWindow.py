import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox,
    QListWidget, QListWidgetItem, QHBoxLayout, QFrame, QInputDialog
)
from PySide6.QtCore import Qt, QFileSystemWatcher
from PreProcessing import PreprocessingWindow
from ProjectWorkSpace import WorkspaceWidget
from CSVeditorDialogue import CsvEditorDialog  # Make sure this is imported
from AnalysisResultWindow import AnalysisResultWindow
from topography_nc_viewer import MainWindow as TerrainViewerWindow
from PySide6.QtGui import QIcon
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, for PyInstaller compatibility """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)



class ProjectWindow(QWidget):
    def __init__(self, project_path):
        super().__init__()
        self.project_path = project_path
        self.project_name = os.path.basename(project_path)
        self.setWindowTitle(f"Project: {self.project_name}")
        self.resize(900, 500)


        icon_path = resource_path("app/assets/SeisSnap_logo.png")
        self.setWindowIcon(QIcon(icon_path))
        # Setup file system watcher
        self.watcher = QFileSystemWatcher()
        self.watcher.addPath(self.project_path)
        self.watcher.directoryChanged.connect(self.populate_csv_list)

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        # Title
        title = QLabel(f"📁 Project Title: {self.project_name}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        # Top Buttons
        open_grapher_btn = QPushButton("📈 Open Grapher")
        open_grapher_btn.clicked.connect(self.open_grapher)

        nc_topo_view_btn = QPushButton("🏔️ View Terrain")
        nc_topo_view_btn.clicked.connect(self.open_terrain_viewer)

        analysis_btn = QPushButton("📊 Analysis")

        top_buttons_layout = QHBoxLayout()
        for btn in [open_grapher_btn, nc_topo_view_btn, analysis_btn]:
            btn.setStyleSheet("font-size: 16px; padding: 8px;")
            top_buttons_layout.addWidget(btn)

        # --- Central Split Layout ---
        central_layout = QHBoxLayout()

        # CSV File List Panel
        csv_list_frame = QFrame()
        csv_list_frame.setFrameShape(QFrame.StyledPanel)
        csv_list_layout = QVBoxLayout()

        csv_label = QLabel("📄 CSV Files")
        csv_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.csv_list_widget = QListWidget()
        self.csv_list_widget.itemChanged.connect(self.on_csv_checkbox_change)
        self.populate_csv_list()
        self.csv_list_widget.itemDoubleClicked.connect(self.rename_csv_file)

        delete_btn = QPushButton("🗑️ Delete Selected CSV")
        delete_btn.clicked.connect(self.delete_selected_csv)

        csv_list_layout.addWidget(csv_label)
        csv_list_layout.addWidget(self.csv_list_widget)
        csv_list_layout.addWidget(delete_btn)

        csv_list_frame.setLayout(csv_list_layout)

        # Workspace Canvas
        workspace_frame = QFrame()
        workspace_frame.setFrameShape(QFrame.StyledPanel)
        workspace_layout = QVBoxLayout()

        workspace_label = QLabel("🧰 Workspace")
        workspace_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        # Pass selected CSVs during initialization
        self.workspace_widget = WorkspaceWidget(
            project_csv_paths=self.get_selected_csv_paths(),
            csv_provider_callback=self.get_selected_csv_paths
        )
        workspace_layout.addWidget(workspace_label)
        workspace_layout.addWidget(self.workspace_widget)

        workspace_frame.setLayout(workspace_layout)

        # Add both panels to central layout
        central_layout.addWidget(csv_list_frame, 1)
        central_layout.addWidget(workspace_frame, 2)

        # Final Assembly
        main_layout.addWidget(title)
        main_layout.addLayout(top_buttons_layout)
        main_layout.addLayout(central_layout)

        self.setLayout(main_layout)
        analysis_btn.clicked.connect(self.open_analysis_window)


    def open_terrain_viewer(self):
        from PySide6.QtWidgets import QFileDialog
        nc_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select NetCDF file",
            self.project_path,
            "NetCDF files (*.nc);;All files (*)"
        )
        if not nc_file:
            return  # User cancelled

        self.terrain_window = TerrainViewerWindow(nc_file)
        self.terrain_window.setAttribute(Qt.WA_DeleteOnClose)
        self.terrain_window.show()


    def open_grapher(self):
        self.preprocess_window = PreprocessingWindow(self.project_path)
        self.preprocess_window.show()

    def populate_csv_list(self):
        self.csv_list_widget.clear()
        csv_files = [f for f in os.listdir(self.project_path) if f.endswith(".csv")]
        for f in csv_files:
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(Qt.Unchecked)
            self.csv_list_widget.addItem(item)

    def delete_selected_csv(self):
        selected = self.csv_list_widget.currentItem()
        if selected:
            file_name = selected.text()
            file_path = os.path.join(self.project_path, file_name)

            if not os.path.exists(file_path):
                QMessageBox.warning(self, "Error", f"The file '{file_name}' no longer exists.")
                self.populate_csv_list()
                return

            confirm = QMessageBox.question(
                self, "Confirm Deletion",
                f"Are you sure you want to delete '{file_name}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if confirm == QMessageBox.Yes:
                try:
                    os.remove(file_path)
                    self.csv_list_widget.takeItem(self.csv_list_widget.row(selected))
                    QMessageBox.information(self, "Deleted", f"Deleted {file_name}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not delete file: {e}")
        else:
            QMessageBox.information(self, "No Selection", "Please select a CSV file to delete.")

    def rename_csv_file(self, item):
        csv_path = os.path.join(self.project_path, item.text())
        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "Error", f"The file '{item.text()}' does not exist.")
            self.populate_csv_list()
            return

        editor = CsvEditorDialog(csv_path, self)
        if editor.exec():
            self.populate_csv_list()

    def on_csv_checkbox_change(self, item):
        # Update the workspace's selected CSVs live
        self.workspace_widget.project_csv_paths = self.get_selected_csv_paths()

    def get_selected_csv_paths(self):
        selected_paths = []
        for i in range(self.csv_list_widget.count()):
            item = self.csv_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                full_path = os.path.join(self.project_path, item.text())
                if os.path.exists(full_path):
                    selected_paths.append(full_path)
        return selected_paths
    
        # --- ProjectWindow ---

    # --- inside ProjectWindow ---------------------------------------------------

    def open_analysis_window(self):
        """Launch / focus the AnalysisResultWindow in a separate top‑level window."""
        analysis_dir = os.path.join(self.project_path, "analysisResult")
        os.makedirs(analysis_dir, exist_ok=True)          # create if missing

        # If we’ve already opened one, just bring it forward
        if getattr(self, "_analysis_win", None) and self._analysis_win.isVisible():
            self._analysis_win.raise_()
            self._analysis_win.activateWindow()
            return

        # Otherwise spin up a fresh, independent window (no parent)
        self._analysis_win = AnalysisResultWindow(analysis_dir)  # <‑‑ remove “self”
        self._analysis_win.setAttribute(Qt.WA_DeleteOnClose)
        self._analysis_win.destroyed.connect(
            lambda: setattr(self, "_analysis_win", None)
        )
        self._analysis_win.show()
