import os
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox, QLineEdit, QLabel, QHBoxLayout, QInputDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

class VelocityMapPlugin(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Velocity Structure Viewer")
        self.resize(800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.load_button = QPushButton("Load CSV Files and Plot")
        self.load_button.clicked.connect(self.load_csv_files)
        self.layout.addWidget(self.load_button)

        self.save_button = QPushButton("Save Map as Image")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setEnabled(False)
        self.layout.addWidget(self.save_button)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.data = None  # Final interpolated [dist, time, slope]
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None

    def load_csv_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select CSV Files", "", "CSV Files (*.csv *.txt)")

        if not file_paths:
            return

        # Ask user for col1, col2
        col1, ok1 = QInputDialog.getText(self, "Column Input", "Enter Time-like Column Name (col1):")
        if not ok1 or not col1.strip():
            return

        col2, ok2 = QInputDialog.getText(self, "Column Input", "Enter Distance-like Column Name (col2):")
        if not ok2 or not col2.strip():
            return

        offset_val, ok3 = QInputDialog.getDouble(self, "Offset Distance", "Distance between CSVs (km):", 20.0, 0.1, 10000.0, 2)
        if not ok3:
            return

        all_data = []

        for i, path in enumerate(file_paths):
            try:
                df = pd.read_csv(path)
                if col1 not in df.columns or col2 not in df.columns:
                    QMessageBox.warning(self, "Column Error", f"File {os.path.basename(path)} is missing {col1} or {col2}. Skipped.")
                    continue

                df = df.dropna(subset=[col1, col2])
                df = df.drop_duplicates(subset=col1)
                df = df.sort_values(by=col1).reset_index(drop=True)

                slope = []
                for j in range(len(df) - 1):
                    dt = df.loc[j + 1, col1] - df.loc[j, col1]
                    dx = df.loc[j + 1, col2] - df.loc[j, col2]
                    slope.append(abs(dx / dt) if dt != 0 else None)
                slope.append(None)  # Pad last
                df["SlopeVelocity"] = slope
                df = df.dropna(subset=["SlopeVelocity"])

                x_offset = i * offset_val
                for _, row in df.iterrows():
                    all_data.append([x_offset, row[col1], row["SlopeVelocity"]])

            except Exception as e:
                QMessageBox.warning(self, "File Error", f"Failed to process {os.path.basename(path)}:\n{str(e)}")

        if not all_data:
            QMessageBox.critical(self, "No Data", "No valid data found in any file.")
            return

        self.data = np.array(all_data)
        self.plot_velocity_map()
        self.save_button.setEnabled(True)

    def plot_velocity_map(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]

        self.grid_x, self.grid_y = np.mgrid[
            min(x):max(x):100j,
            min(y):max(y):100j
        ]
        self.grid_z = griddata((x, y), z, (self.grid_x, self.grid_y), method='linear')

        # Optional: smoothing
        if not np.isnan(self.grid_z).all():
            self.grid_z = gaussian_filter(self.grid_z, sigma=1.2)

        contourf = ax.contourf(self.grid_x, self.grid_y, self.grid_z, levels=20, cmap="jet")
        contour = ax.contour(self.grid_x, self.grid_y, self.grid_z, levels=10, colors='black', linewidths=0.8)
        ax.clabel(contour, inline=True, fontsize=8, fmt="%.1f")

        ax.set_title("Velocity Structure Map (Slope)")
        ax.set_xlabel("Distance Offset (km)")
        ax.set_ylabel("Time (s)")
        ax.invert_yaxis()
        self.figure.colorbar(contourf, ax=ax, label="Velocity (km/s)")
        self.canvas.draw()

    def save_plot(self):
        if self.grid_z is None:
            QMessageBox.warning(self, "No Plot", "Nothing to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "velocity_map.png", "PNG Files (*.png);;PDF Files (*.pdf)")
        if not file_path:
            return

        try:
            self.figure.savefig(file_path)
            QMessageBox.information(self, "Saved", f"Plot saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save image: {str(e)}")


    def load_velocity_data_from_file(self, file_path):
        try:
            df = pd.read_csv(file_path, header=None, names=["x", "y", "slope"])
            self.data = df.to_numpy()
            self.plot_velocity_map()
            self.save_button.setEnabled(True)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load Failed", f"Could not load velocity data:\n{e}")

