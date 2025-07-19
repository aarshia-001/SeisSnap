from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QPushButton, QLineEdit,
    QCheckBox, QComboBox, QMessageBox, QSpinBox
)
from PySide6.QtCore import Qt
from seismic_viewer_2 import SeismicViewer
from PySide6.QtGui import QIcon
import os


class PreprocessingWindow(QWidget):
    def __init__(self, project_path):
        super().__init__()
        self.project_path = project_path
        self.setWindowTitle(f"Preprocessing - {project_path}")
        self.resize(500, 400)
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "SeisSnap_logo.png")
        self.setWindowIcon(QIcon(icon_path))
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- Gain Controls ---
        self.gain_active = QCheckBox("Enable Gain")
        self.gain_mode = QComboBox()
        self.gain_mode.addItems(["constant gain", "add more feature"])
        self.gain_value = QSpinBox()
        self.gain_value.setRange(1, 1000)
        self.gain_value.setValue(10)

        gain_layout = QVBoxLayout()
        gain_layout.addWidget(self.gain_active)
        gain_layout.addWidget(self.gain_mode)
        gain_layout.addWidget(QLabel("Gain Value:"))
        gain_layout.addWidget(self.gain_value)

        # --- Bandpass Inputs ---
        self.lowCut_input = QLineEdit()
        self.midlowCut_input = QLineEdit()
        self.midhighCut_input = QLineEdit()
        self.highCut_input = QLineEdit()
        self.lowCut_input.setPlaceholderText("Low Cut Frequency (e.g. 10.0)")
        self.midlowCut_input.setPlaceholderText("Gain Low Cut Frequency (e.g. 12.0)")
        self.midhighCut_input.setPlaceholderText("Gain High Cut Frequency (e.g. 17.0)")
        self.highCut_input.setPlaceholderText("High Cut Frequency (e.g. 20.0)")

        bandwidth_layout = QVBoxLayout()
        bandwidth_layout.addWidget(QLabel("Bandwidth Filters"))
        bandwidth_layout.addWidget(self.lowCut_input)
        bandwidth_layout.addWidget(self.midlowCut_input)
        bandwidth_layout.addWidget(self.midhighCut_input)
        bandwidth_layout.addWidget(self.highCut_input)

        # --- Reduction Velocity Inputs ---
        self.reduction_active = QCheckBox("Enable Reduction Velocity")
        self.reduction_velocity = QSpinBox()
        self.reduction_velocity.setRange(1, 20000)
        self.reduction_velocity.setValue(1500)

        self.dist_traces = QSpinBox()
        self.dist_traces.setRange(1, 90000)
        self.dist_traces.setValue(125)

        reduction_layout = QVBoxLayout()
        reduction_layout.addWidget(self.reduction_active)
        reduction_layout.addWidget(QLabel("Reduction Velocity (m/s):"))
        reduction_layout.addWidget(self.reduction_velocity)
        reduction_layout.addWidget(QLabel("Distance between Traces (m):"))
        reduction_layout.addWidget(self.dist_traces)

        # Disable inputs initially
        self.gain_mode.setEnabled(False)
        self.gain_value.setEnabled(False)
        self.reduction_velocity.setEnabled(False)
        self.dist_traces.setEnabled(False)

        # Toggle state enable/disable
        self.gain_active.stateChanged.connect(
            lambda state: (
                self.gain_mode.setEnabled(state),
                self.gain_value.setEnabled(state)
            )
        )
        self.reduction_active.stateChanged.connect(
            lambda state: (
                self.reduction_velocity.setEnabled(state),
                self.dist_traces.setEnabled(state)
            )
        )

        # --- Next Button ---
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.launch_viewer)

        # --- Assemble Layout ---
        layout.addLayout(gain_layout)
        layout.addSpacing(10)
        layout.addLayout(bandwidth_layout)
        layout.addSpacing(10)
        layout.addLayout(reduction_layout)
        layout.addSpacing(20)
        layout.addWidget(next_btn)
        self.setLayout(layout)

    def launch_viewer(self):
        try:
            gain_enabled = self.gain_active.isChecked()
            gain_mode = self.gain_mode.currentText() if gain_enabled else None
            gain_value = self.gain_value.value() if gain_enabled and gain_mode == "constant gain" else None

            lowCut = float(self.lowCut_input.text())
            highCut = float(self.highCut_input.text())
            midHighCut_=float(self.midhighCut_input.text())
            midLowCut_= float(self.midlowCut_input.text())

            reduct_enabled = self.reduction_active.isChecked()
            reduct_vel = self.reduction_velocity.value() if reduct_enabled else None
            reduct_dist = self.dist_traces.value() if reduct_enabled else 1

            self.viewer = SeismicViewer(
                output_dir=self.project_path,
                gain_isActive=gain_enabled,
                gain=gain_value,
                reductVel_isActive=reduct_enabled,
                reductVel=reduct_vel,
                dist_Trace=reduct_dist,
                lowCutFreq=lowCut,
                highCutFreq=highCut,
                midHighCut=midHighCut_,
                midLowCut=midLowCut_,
            )
            self.viewer.resize(1200, 800)
            self.viewer.show()
            self.close()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for lowCut/highCut frequencies.")
