import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QWidget, QPushButton,
    QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt
from scipy.fft import rfft, rfftfreq, fft2, fftshift
from scipy.signal import stft

from PySide6.QtGui import QIcon

import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, for PyInstaller compatibility """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class SpectralAnalysisWindow(QDialog):
    """
    Quick‑look spectral tools (1‑D FFT, 2‑D FFT, STFT, Radon placeholder).

    * Expects `data` in (trace, sample) order.  If not, it is transposed.
    * Uses the `time_vector` to derive the sampling interval.
    """
    def __init__(self, method_name, data, time_vector, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Spectral Analysis – {method_name}")
        self.resize(1000, 600)

        icon_path = resource_path("app/assets/SeisSnap_logo.png")
        self.setWindowIcon(QIcon(icon_path))

        # --- stash & sanity‑check ----------------------------------------
        if data.shape[0] < data.shape[1]:          # likely (samples, traces)
            data = data.T
        self.data  = data                          # (n_traces, n_samples)
        self.time  = time_vector
        self.dt    = float(self.time[1] - self.time[0])
        self.method_name = method_name

        # --- ui scaffolding ----------------------------------------------
        self.layout = QVBoxLayout(self)
        self.controls = QFormLayout()
        self.controls.setAlignment(Qt.AlignTop)

        self.control_panel = QWidget()
        self.control_panel.setLayout(self.controls)
        self.layout.addWidget(self.control_panel)

        self.reload_btn = QPushButton("Reload")
        self.reload_btn.clicked.connect(self.compute_and_plot)
        self.layout.addWidget(self.reload_btn)

        self.line_plot  = pg.PlotWidget()
        self.image_plot = pg.ImageView()
        self.layout.addWidget(self.line_plot)
        self.layout.addWidget(self.image_plot)

        # --- parameter widgets -------------------------------------------
        self._build_param_widgets()

        self._toggle_plot_widgets()
        self.compute_and_plot()

    # ------------------------------------------------------------------ #
    def _build_param_widgets(self):
        """Create / populate widgets shown on the left panel."""
        m = self.method_name

        if m.startswith("1D FFT") or m.startswith("STFT"):
            self.trace_idx = QSpinBox()
            self.trace_idx.setMaximum(self.data.shape[0] - 1)
            self.controls.addRow("Trace index", self.trace_idx)

        if m.startswith("STFT"):
            self.nperseg = QSpinBox();  self.nperseg.setRange(16, 4096);  self.nperseg.setValue(256)
            self.noverlp = QSpinBox();  self.noverlp.setRange(0, 4096);   self.noverlp.setValue(128)
            self.controls.addRow("Window size", self.nperseg)
            self.controls.addRow("Overlap",      self.noverlp)

        if m.startswith("Radon"):
            self.pmin    = QDoubleSpinBox(); self.pmin.setRange(-1.0, 0.0); self.pmin.setDecimals(3); self.pmin.setValue(-0.4)
            self.pmax    = QDoubleSpinBox(); self.pmax.setRange( 0.0, 1.0); self.pmax.setDecimals(3); self.pmax.setValue( 0.4)
            self.np_vals = QSpinBox();       self.np_vals.setRange(50, 2000); self.np_vals.setValue(400)
            self.controls.addRow("p min", self.pmin)
            self.controls.addRow("p max", self.pmax)
            self.controls.addRow("# p",   self.np_vals)

    # ------------------------------------------------------------------ #
    def _toggle_plot_widgets(self):
        """Show line plot for 1‑D FFT otherwise show image view."""
        if self.method_name.startswith("1D FFT"):
            self.line_plot.show();  self.image_plot.hide()
        else:
            self.line_plot.hide();  self.image_plot.show()

    # ------------------------------------------------------------------ #
    def compute_and_plot(self):
        self._toggle_plot_widgets()
        m = self.method_name

        if m.startswith("1D FFT"):
            self.line_plot.clear()

            idx   = self.trace_idx.value()
            trace = self.data[idx, :].astype(float)

            # --- detrend & window ---------------------------------------
            trace -= trace.mean()
            trace *= np.hanning(len(trace))

            yf = np.abs(rfft(trace))
            xf = rfftfreq(len(trace), d=self.dt)

            self.line_plot.plot(xf, yf, pen='y')
            self.line_plot.setTitle(f"1‑D FFT – trace {idx}")
            self.line_plot.setLabel("left",   "Amplitude")
            self.line_plot.setLabel("bottom", "Frequency (Hz)")

        elif m.startswith("2D FFT"):
            spec2d = fftshift(np.abs(fft2(self.data)))
            self.image_plot.setImage(np.log1p(spec2d.T), autoLevels=True)
            self.image_plot.setPredefinedGradient("viridis")

        elif m.startswith("STFT"):
            idx   = self.trace_idx.value()
            f, t, Z = stft(
                self.data[idx], fs=1.0 / self.dt,
                nperseg=self.nperseg.value(),
                noverlap=self.noverlp.value(),
                window='hann'
            )
            mag = np.abs(Z)
            self.image_plot.setImage(np.log1p(mag), autoLevels=True)
            self.image_plot.setPredefinedGradient("viridis")
            self.image_plot.setLabel('bottom', 'Time (s)')
            self.image_plot.setLabel('left',   'Freq (Hz)')

        elif m.startswith("Radon"):
            # placeholder
            rnd = np.random.rand(300, 300)
            self.image_plot.setImage(rnd, autoLevels=True)
            self.image_plot.setPredefinedGradient("magma")
