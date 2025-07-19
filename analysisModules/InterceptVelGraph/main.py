MODULE_NAME = "InterceptVelGrapher"
DESCRIPTION = "Plots lateral velocity structure using selected CSVs with smoothing and export."
TYPE = "multi"

import os
import importlib.util

def load_velocity_map_plugin():
    """Dynamically load VelocityMapPlugin from sibling file."""
    plugin_path = os.path.join(os.path.dirname(__file__), "VelocityMapPlugin.py")
    plugin_path = os.path.abspath(plugin_path)
    spec = importlib.util.spec_from_file_location("VelocityMapPlugin", plugin_path)
    plugin_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin_mod)
    return plugin_mod.VelocityMapPlugin

def configure(parent=None, sample_path=None):
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
        QPushButton, QDoubleSpinBox, QCheckBox, QFileDialog
    )

    class ConfigDialog(QDialog):
        def launch_viewer(csv_path):
            from PySide6.QtWidgets import QApplication
            import sys

            VelocityMapPlugin = load_velocity_map_plugin()
            app = QApplication.instance() or QApplication(sys.argv)
            viewer = VelocityMapPlugin()
            viewer.show()
            viewer.load_velocity_data_from_file(csv_path)
            app.exec()

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Velocity Structure Settings")
            self.resize(400, 200)
            layout = QVBoxLayout()

            self.offset_input = QDoubleSpinBox()
            self.offset_input.setValue(1.0)
            self.offset_input.setDecimals(4)
            layout.addWidget(QLabel("Offset Multiplier:"))
            layout.addWidget(self.offset_input)

            self.col1_input = QLineEdit()
            self.col2_input = QLineEdit()
            layout.addWidget(QLabel("Enter Time-like Column (col1):"))
            layout.addWidget(self.col1_input)
            layout.addWidget(QLabel("Enter Distance-like Column (col2):"))
            layout.addWidget(self.col2_input)

            self.spacing_input = QDoubleSpinBox()
            self.spacing_input.setRange(0, 100000)
            self.spacing_input.setValue(20.0)
            layout.addWidget(QLabel("Distance between CSV files:"))
            layout.addWidget(self.spacing_input)

            self.smooth_checkbox = QCheckBox("Apply smoothing to contour")
            layout.addWidget(self.smooth_checkbox)

            self.save_path = ""
            def choose_save_path():
                self.save_path, _ = QFileDialog.getSaveFileName(self, "Save Plot As", "", "PNG (*.png);;PDF (*.pdf)")
            save_btn = QPushButton("Choose Save Location (Optional)")
            save_btn.clicked.connect(choose_save_path)
            layout.addWidget(save_btn)

            buttons = QHBoxLayout()
            ok = QPushButton("OK")
            cancel = QPushButton("Cancel")
            ok.clicked.connect(self.accept)
            cancel.clicked.connect(self.reject)
            buttons.addWidget(ok)
            buttons.addWidget(cancel)
            layout.addLayout(buttons)

            self.setLayout(layout)

    dialog = ConfigDialog(parent)
    if dialog.exec() == QDialog.Accepted:
        return {
            "offset": dialog.offset_input.value(),
            "col1": dialog.col1_input.text().strip(),
            "col2": dialog.col2_input.text().strip(),
            "dist_between": dialog.spacing_input.value(),
            "smooth": dialog.smooth_checkbox.isChecked(),
            "save_path": dialog.save_path
        }

    return {}

def run(file_list, settings):
    import pandas as pd
    import numpy as np
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    import tempfile
    import os
    import matplotlib.pyplot as plt

    offset = settings.get("offset")
    col1 = settings.get("col1")
    col2 = settings.get("col2")
    dist_between = settings.get("dist_between")
    save_path = settings.get("save_path")
    smooth = settings.get("smooth")

    if isinstance(file_list, pd.DataFrame):
        raise ValueError(
            "[InterceptVelGrapher] This plugin expects multiple files (list of paths), but received a single DataFrame. Ensure TYPE = 'multi' is set."
        )
    if not isinstance(file_list, list) or not file_list:
        raise ValueError("File list is empty or invalid.")

    for path in file_list:
        if not isinstance(path, str) or not os.path.isfile(path):
            raise ValueError(f"Invalid file path: {path}")

    all_data = []

    for i, path in enumerate(file_list):
        try:
            df = pd.read_csv(path)

            if col1 not in df.columns or col2 not in df.columns:
                print(f"[!] Skipping {os.path.basename(path)} — missing columns: {col1}, {col2}")
                continue

            df = df.dropna(subset=[col1, col2])
            df = df.drop_duplicates(subset=col1)
            df = df.sort_values(by=col1).reset_index(drop=True)

            slope = []
            for j in range(len(df) - 1):
                dt = df.loc[j + 1, col1] - df.loc[j, col1]
                dx = df.loc[j + 1, col2] - df.loc[j, col2]
                slope.append(abs(offset * dx / dt) if dt != 0 else None)
            slope.append(None)
            df["SlopeVelocity"] = slope
            df = df.dropna(subset=["SlopeVelocity"]).reset_index(drop=True)

            x = i * dist_between
            for _, row in df.iterrows():
                all_data.append([x, row[col1], row["SlopeVelocity"]])

        except Exception as e:
            print(f"[!] Skipped file {path}: {e}")

    if not all_data:
        raise ValueError("No valid data to plot.")

    data = np.array(all_data)
    grid_x, grid_y = np.mgrid[
        0 : dist_between * (len(file_list) - 1) : 100j,
        min(data[:, 1]) : max(data[:, 1]) : 100j
    ]
    grid_z = griddata((data[:, 0], data[:, 1]), data[:, 2], (grid_x, grid_y), method='linear')

    if smooth:
        grid_z = gaussian_filter(grid_z, sigma=1.2)

    # --- Save interpolated data for plugin ---
    temp_path = os.path.join(tempfile.gettempdir(), "velocity_map_data.csv")
    with open(temp_path, "w") as f:
        for row in data:
            f.write(f"{row[0]},{row[1]},{row[2]}\n")

    # --- Define result directory ---
    result_dir = os.path.join(os.path.dirname(file_list[0]), "analysisResult")
    os.makedirs(result_dir, exist_ok=True)

    # --- Define save path for image ---
    if not save_path:
        save_path = os.path.join(result_dir, "velocity_map.png")

    # --- Save static velocity structure plot ---
    plt.figure(figsize=(8, 5))
    contourf = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap="jet")
    contour = plt.contour(grid_x, grid_y, grid_z, levels=10, colors='black', linewidths=0.8)
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.1f")
    plt.colorbar(contourf, label="Velocity (km/s)")
    plt.xlabel("Distance (km)")
    plt.ylabel("Time (s)")
    plt.title("Intercept Velocity Structure Map")
    plt.xticks(np.arange(0, dist_between * len(file_list) + 1, dist_between))
    plt.gca().invert_yaxis()
    plt.tight_layout()

    ext = os.path.splitext(save_path)[1].lower()
    if ext in [".png", ".pdf"]:
        plt.savefig(save_path)
        print(f"[✓] Plot saved to: {save_path}")
    else:
        print(f"[!] Unsupported save format: {ext}")
    plt.close()

    # --- Save contour level Y-values ---
    plt.contour(grid_x, grid_y, grid_z, levels=10)
    level_path = os.path.join(result_dir, "vel_list.txt")
    with open(level_path, "w") as f:
        for c in plt.gca().collections:
            if hasattr(c, 'get_paths'):
                for p in c.get_paths():
                    try:
                        f.write(f"{p.vertices[0][1]:.2f}\n")
                    except:
                        continue
    plt.close()
    print(f"[✓] Contour levels saved to: {level_path}")

    # --- Launch interactive viewer ---
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer

    VelocityMapPlugin = load_velocity_map_plugin()
    def launch_plugin():
        viewer = VelocityMapPlugin()
        viewer.show()
        viewer.load_velocity_data_from_file(temp_path)

    QTimer.singleShot(0, launch_plugin)
