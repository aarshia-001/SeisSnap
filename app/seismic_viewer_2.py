import sys, struct, numpy as np, traceback, csv, os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QLabel,
    QVBoxLayout, QWidget, QHBoxLayout, QSpinBox, QCheckBox,
    QInputDialog, QStatusBar, QLineEdit, QComboBox, QFrame, QGraphicsProxyWidget
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent
from scipy.signal import butter, filtfilt
import pyqtgraph as pg
import os
import sys
import obspy


def resource_path(relative_path):
    """ Get absolute path to resource, for PyInstaller compatibility """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)



# Add fallback path for obspy/lib manually
lib_dir = os.path.join(os.path.dirname(obspy.__file__), "lib")
if not os.path.isdir(lib_dir) and hasattr(sys, '_MEIPASS'):
    # Inside PyInstaller bundle
    fallback = os.path.join(sys._MEIPASS, "obspy", "lib")
    if os.path.exists(fallback):
        os.environ["PATH"] += os.pathsep + fallback

from obspy.io.segy.segy import _read_segy
from obspy import read as obspy_read
from scipy.ndimage import uniform_filter1d
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PySide6.QtWidgets import QGraphicsRectItem       # instead of QtGui.QGraphicsRectItem
from PySide6.QtGui import QIcon

HEADER_BYTES = 240
DT = 0.004
ORDER = 10
WIGGLE_SCALE = 0.3
VISIBLE_MARGIN = 20
PAN_THROTTLE_MS = 300

import numpy as np
from numpy.fft import rfft, irfft, rfftfreq

def bandpass_filter(traces, dt, f1, f2, f3, f4, *, agc_norm=False):
    """
    Seismic‑Unix–style trapezoidal band‑pass.

    Parameters
    ----------
    traces : (n_traces, n_samples) ndarray
        Gather to filter.
    dt     : float
        Sample interval in seconds (e.g. 0.004).
    f1,f2,f3,f4 : float
        Corner frequencies in Hz (f1 < f2 < f3 < f4).
    agc_norm : bool, optional
        If True every trace is scaled to max(abs(trace)) == 1 *after*
        filtering.  Leave False for quantitative work (spectral, AVO …).

    Returns
    -------
    out : ndarray  (same shape & dtype as input)
    """
    # ---- FFT parameters --------------------------------------------------
    n_tr, n_samp = traces.shape
    nyq   = 0.5 / dt
    freqs = rfftfreq(n_samp, dt)      # length n_freq = n_samp//2 + 1

    # ---- build SU trapezoid ---------------------------------------------
    amp = np.zeros_like(freqs)
    # 0 … f1   : 0
    # f1…f2    : cosine taper 0→1
    taper1 = (freqs >= f1) & (freqs < f2)
    amp[taper1] = 0.5 * (1 - np.cos(np.pi * (freqs[taper1] - f1) / (f2 - f1)))

    # f2…f3    : 1
    amp[(freqs >= f2) & (freqs <= f3)] = 1.0

    # f3…f4    : cosine taper 1→0
    taper2 = (freqs > f3) & (freqs <= f4)
    amp[taper2] = 0.5 * (1 + np.cos(np.pi * (freqs[taper2] - f3) / (f4 - f3)))

    # f4…nyq   : 0  (already zeros)

    # ---- apply to every trace -------------------------------------------
    out = np.empty_like(traces, dtype=float)
    for i, tr in enumerate(traces):
        spec = rfft(tr)
        spec *= amp              # frequency‑domain multiplication
        filt = irfft(spec, n=n_samp)
        if agc_norm:
            m = np.max(np.abs(filt)) + 1e-12
            filt /= m
        out[i] = filt

    return out


#AGC

def apply_agc_fast(traces, window_size):
    """
    Vectorized AGC using RMS envelope computed with uniform_filter1d.
    Assumes traces is a 2D array: shape (n_traces, n_samples)
    """
    traces = np.asarray(traces)
    power = traces ** 2
    rms = np.sqrt(uniform_filter1d(power, size=window_size, axis=1, mode='reflect'))
    return traces / (rms + 1e-10)

#constant gain
def apply_gain(traces, gain_value):
    if gain_value is None:
        return traces
    return np.clip(traces * gain_value, -1, 1)

def apply_reduction_velocity(traces, velocity, distBwTrace):
    if velocity is None or velocity<=0:
        return traces
    corrected = []
    for i, tr in enumerate(traces):
        offset = i * distBwTrace
        time_shift = offset / velocity
        sample_shift = int(np.round(time_shift / DT))
        shifted_data = np.roll(tr, -sample_shift)
        if sample_shift > 0:
            shifted_data[-sample_shift:] = 0
        elif sample_shift < 0:
            shifted_data[:-sample_shift] = 0
        corrected.append(shifted_data)
    return np.array(corrected)
#mcs
def load_mcs_traces(filename, start, end):
    traces = []
    try:
        with open(filename, 'rb') as f:
            index = 0
            while True:
                header = f.read(240)  # Optional: skip if not present
                if not header or len(header) < 240:
                    break
                if len(header) >= 116:
                    ns = struct.unpack('>h', header[114:116])[0]
                else:
                    ns = 1024  # default fallback

                data = f.read(ns * 4)
                if len(data) != ns * 4:
                    break
                if start <= index < end:
                    trace = np.frombuffer(data, dtype='>f4')
                    traces.append(trace)
                index += 1
                if index >= end:
                    break
        return np.array(traces)
    except Exception as e:
        print(f"Failed to read MCS: {e}")
        return np.empty((0,))
#seg-d
def load_seg_traces(filename, start, end):
    try:
        # Heuristic: SEG seismic trace data (float32)
        traces = []
        with open(filename, 'rb') as f:
            index = 0
            while True:
                header = f.read(240)  # skip header (optional)
                if not header or len(header) < 240:
                    break
                ns = struct.unpack('>h', header[114:116])[0] if len(header) >= 116 else 1024
                data = f.read(ns * 4)
                if len(data) != ns * 4:
                    break
                if start <= index < end:
                    trace = np.frombuffer(data, dtype='>f4')
                    traces.append(trace)
                index += 1
                if index >= end:
                    break
        return np.array(traces)
    except Exception as e:
        print(f"Failed to read SEG data: {e}")
        return np.empty((0,))

# su seismic unix
def load_su_traces(filename, start, end):
    traces = []
    with open(filename, 'rb') as f:
        index = 0
        while True:
            header = f.read(HEADER_BYTES)
            if not header or len(header) < HEADER_BYTES:
                break
            ns = struct.unpack('>h', header[114:116])[0]
            data = f.read(ns * 4)
            if len(data) != ns * 4:
                break
            if start <= index < end:
                trace = np.frombuffer(data, dtype='>f4')
                traces.append(trace)
            index += 1
            if index >= end:
                break
    return np.array(traces)
#segy sgy

def load_segy_traces(filename, start, end):
    try:
        st = _read_segy(filename)
        selected = st.traces[start:end]
        traces = [tr.data.astype(np.float32) for tr in selected]
        return np.array(traces)
    except Exception as e:
        print(f"Failed to read SEG-Y: {e}")
        return np.empty((0,))



#miniSEEd miniseed miniseed
def load_mseed_traces(filename, start, end):
    try:
        st = obspy_read(filename, format='MSEED')
        traces = [tr.data.astype(np.float32) for tr in st[start:end] if tr.stats.npts > 0]
        return np.array(traces)
    except Exception as e:
        print(f"Failed to read MiniSEED: {e}")
        return np.empty((0,))


def load_traces(filename, start, end):
    ext = os.path.splitext(filename)[-1].lower()
    if ext == '.su':
        return load_su_traces(filename, start, end)
    elif ext in ['.segy', '.sgy']:
        return load_segy_traces(filename, start, end)
    elif ext == '.mseed':
        return load_mseed_traces(filename, start, end)
    elif ext == '.mcs':
        return load_mcs_traces(filename, start, end)
    elif ext == '.seg':
        return load_seg_traces(filename, start, end)
    else:
        raise ValueError("Unsupported file format: " + ext)


def find_first_peak(trace, s_idx, e_idx):
    segment = trace[s_idx:e_idx]
    if segment.size == 0:
        return s_idx
    return s_idx + np.argmax(np.abs(segment))

class SeismicViewer(QMainWindow):
    def __init__(self, output_dir=None, gain_isActive=False, gain=1,
             reductVel_isActive=False, reductVel=None, lowCutFreq=10.0, highCutFreq=20.0, dist_Trace=1, midHighCut=17.0, midLowCut=12.0):
        super().__init__()
        self.setWindowTitle("Graph Seismic Viewer")

        icon_path = resource_path("app/assets/SeisSnap_logo.png")
        self.setWindowIcon(QIcon(icon_path))



        self.output_dir = output_dir
        self.filename = None
        self.time = None
        self.nsamples = None
        self.total_traces = None
        self.loaded_start = 0
        self.loaded_end = 0
        self.traces_cache = {}
        self.snap_points = []
        self.added_points = []
        self.roi_lines = []
        self.draw_mode = False
        self.add_mode = False
        self.current_point = None
        self.history = []
        self.selected_point = None
        self.gain_isActive = gain_isActive
        self.gain = gain
        self.reductVel_isActive = reductVel_isActive
        self.reductVel = reductVel
        self.lowCutFreq = lowCutFreq
        self.highCutFreq = highCutFreq

        self.midHighCut=midHighCut
        self.midLowCut= midLowCut
        self.distTrace = dist_Trace
        self.agc_isActive = False
        self.agc_window = 100

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.viewbox = self.plot_widget.getViewBox()
        self.viewbox.invertY(True)
        self.viewbox.setMouseEnabled(x=True, y=True)


                # --- navigator (overview) -------------------------------------------------
        self.nav_viewbox = pg.ViewBox(invertY=True)
        self.nav_widget  = pg.PlotWidget(viewBox=self.nav_viewbox, enableMenu=False)
        self.nav_widget.setFixedSize(220, 160)
        self.nav_widget.setMouseEnabled(x=False, y=False)
        self.nav_widget.hideAxis('left')
        self.nav_widget.hideAxis('bottom')

        self.nav_img = pg.ImageItem()
        self.nav_viewbox.addItem(self.nav_img)

        from PySide6.QtWidgets import QGraphicsRectItem        # already imported
        self.nav_rect = QGraphicsRectItem()
        self.nav_rect.setPen(pg.mkPen('r', width=1))
        self.nav_rect.setBrush(pg.mkBrush(255, 0, 0, 40))
        self.nav_viewbox.addItem(self.nav_rect)

        # ⬇️ 1. **add the proxy into the scene**
        self.nav_proxy = self.plot_widget.scene().addWidget(self.nav_widget)
        self.nav_proxy.setZValue(50)      # top layer

        # ⬇️ 2. **hook a resize handler so the proxy stays in the corner**
        def _reposition_proxy(ev=None):
            pw_size = self.plot_widget.size()
            # 10‑px margin
            self.nav_proxy.setPos(10, pw_size.height() - self.nav_widget.height() - 10)

        # call once now, and every time the PlotWidget resizes
        _reposition_proxy()
        self.plot_widget.sigResized = pg.SignalProxy(self.plot_widget.sigDeviceRangeChanged,
                                                    slot=_reposition_proxy)




        # === Ribbon Controls ===
        def vline():
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)
            return line

        load_btn = QPushButton("Load")
        load_btn.setFixedWidth(80)
        load_btn.clicked.connect(self.select_file)

        self.start_spin = QSpinBox()
        self.start_spin.setMaximum(1000000)
        self.start_spin.setValue(0)
        self.start_spin.setFixedWidth(70)

        self.end_spin = QSpinBox()
        self.end_spin.setMaximum(1000000)
        self.end_spin.setValue(200)
        self.end_spin.setFixedWidth(70)

        plot_btn = QPushButton("Plot")
        plot_btn.clicked.connect(self.scroll_to_range)

        self.draw_checkbox = QCheckBox("Draw")
        self.draw_checkbox.stateChanged.connect(self.toggle_draw_mode)

        self.add_checkbox = QCheckBox("Add")
        self.add_checkbox.stateChanged.connect(self.toggle_add_mode)

        self.agc_checkbox = QCheckBox("AGC")
        self.agc_checkbox.stateChanged.connect(self.toggle_agc)

        self.agc_spinbox = QSpinBox()
        self.agc_spinbox.setRange(10, 10000)
        self.agc_spinbox.setValue(self.agc_window)
        self.agc_spinbox.setFixedWidth(90)
        self.agc_spinbox.setSuffix(" smpl")
        self.agc_spinbox.valueChanged.connect(self.set_agc_window)

        snap_btn = QPushButton("Snap")
        snap_btn.clicked.connect(self.snap_to_peaks)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_csv)

        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.undo)

        from seismicFreqAnal import SpectralAnalysisWindow

        self.spectral_dropdown = QComboBox()
        self.spectral_dropdown.setFixedWidth(160)
        self.spectral_dropdown.addItems([
            "1D FFT", "2D FFT", "STFT", "Radon"
        ])

        self.visualise_btn = QPushButton("Spec")
        self.visualise_btn.clicked.connect(self.open_spectral_analysis)

        self.velocity_input = QLineEdit()
        self.velocity_input.setPlaceholderText("Velocity (m/s)")
        self.velocity_input.setFixedWidth(130)
        self.velocity_input.textChanged.connect(self.on_velocity_changed)

        # === Horizontal Ribbon Layout ===
        ribbon = QHBoxLayout()
        ribbon.setSpacing(10)
        ribbon.addWidget(load_btn)
        ribbon.addWidget(vline())

        ribbon.addWidget(QLabel("Start"))
        ribbon.addWidget(self.start_spin)
        ribbon.addWidget(QLabel("End"))
        ribbon.addWidget(self.end_spin)
        ribbon.addWidget(self.velocity_input)
        ribbon.addWidget(plot_btn)
        ribbon.addWidget(vline())



        
        ribbon.addWidget(self.draw_checkbox)
        ribbon.addWidget(self.add_checkbox)
        ribbon.addWidget(snap_btn)
        ribbon.addWidget(save_btn)
        ribbon.addWidget(undo_btn)
        ribbon.addWidget(vline())

        
        ribbon.addWidget(self.agc_checkbox)
        ribbon.addWidget(self.agc_spinbox)
        ribbon.addWidget(vline())


        ribbon.addWidget(vline())
        ribbon.addWidget(self.spectral_dropdown)
        ribbon.addWidget(self.visualise_btn)
        ribbon.addWidget(vline())
        ribbon.addWidget(vline())
        ribbon.addStretch()

        ribbon_widget = QWidget()
        ribbon_widget.setLayout(ribbon)

        # === Final Layout ===
        layout = QVBoxLayout()
        layout.addWidget(ribbon_widget)
        layout.addWidget(self.plot_widget)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.plot_widget.scene().sigMouseClicked.connect(self.handle_mouse_click)
        self.viewbox.sigXRangeChanged.connect(self.delayed_pan_load)

        self.pan_timer = QTimer()
        self.pan_timer.setSingleShot(True)
        self.pan_timer.timeout.connect(self.load_visible_data)

        self.point_items = []
        self.plot_widget.setFocusPolicy(Qt.StrongFocus)


    def toggle_agc(self, state):
        self.agc_isActive = bool(state)
        self.status.showMessage("AGC " + ("enabled" if self.agc_isActive else "disabled"))
        self.load_visible_data()

    def set_agc_window(self, value):
        self.agc_window = value
        if self.agc_isActive:
            self.status.showMessage(f"AGC window set to {value} samples")
            self.load_visible_data()

    def on_velocity_changed(self, text):
        text = text.strip()
        if text:
            try:
                value = float(text)
                if value <= 0:
                    raise ValueError
                self.reductVel = value
                self.reductVel_isActive = True
                self.status.showMessage(f"Reduction velocity set to {value} m/s")
                self.load_visible_data()  # Trigger update
            except ValueError:
                self.status.showMessage("Invalid velocity input. Must be a positive number.")
                self.reductVel_isActive = False
        else:
            self.reductVel_isActive = False
            self.status.showMessage("Reduction velocity disabled.")
            self.load_visible_data()  # Refresh with velocity turned off

    def keyPressEvent(self, event: QKeyEvent):
        if self.selected_point:
            if self.selected_point in self.point_items:
                idx = self.point_items.index(self.selected_point)
            else:
                return

            x, y = self.selected_point.pos().x(), self.selected_point.pos().y()

            if event.key() == Qt.Key_Up:
                y -= DT
            elif event.key() == Qt.Key_Down:
                y += DT
            elif event.key() == Qt.Key_Delete:
                self.store_history()
                self.plot_widget.removeItem(self.selected_point)
                if (x, y) in self.snap_points:
                    self.snap_points.remove((x, y))
                elif (x, y) in self.added_points:
                    self.added_points.remove((x, y))
                self.point_items.pop(idx)
                self.selected_point = None
                return

            self.store_history()
            self.selected_point.setPos(x, y)
            old = (x, y)
            new = (x, y)
            if old in self.snap_points:
                i = self.snap_points.index(old)
                self.snap_points[i] = new
            elif old in self.added_points:
                i = self.added_points.index(old)
                self.added_points[i] = new


    def store_history(self):
        self.history.append((list(self.snap_points), list(self.added_points)))

    def undo(self):
        if not self.history:
            return
        self.snap_points, self.added_points = self.history.pop()
        self.plot_snapped_points()

    def delayed_pan_load(self):
        self.pan_timer.start(PAN_THROTTLE_MS)

    def select_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Seismic File", "", "Seismic Files (*.su *.segy *.sgy *.mseed *.mcs *.seg)")

        if file:
            self.filename = file
            self.loaded_start = self.loaded_end = 0
            self.traces_cache.clear()
            self.snap_points.clear()
            self.added_points.clear()
            self.roi_lines.clear()
            self.plot_widget.clear()
            self.time = None
            self.status.showMessage("File loaded. Scroll or zoom to trigger load.")

    def load_visible_data(self):
        if not self.filename:
            return
        x_min, x_max = map(int, self.viewbox.viewRange()[0])
        x_min = max(0, x_min - VISIBLE_MARGIN)
        x_max += VISIBLE_MARGIN

        if self.loaded_start <= x_min and self.loaded_end >= x_max:
            return
        
        
        try:
            traces = load_traces(self.filename, x_min, x_max)
            if traces.size == 0:
                self.status.showMessage("No traces loaded.")
                return

            self.total_traces = max(self.total_traces or 0, x_max)
            self.nsamples = traces.shape[1]
            self.time = np.arange(self.nsamples) * DT

            if self.gain_isActive:
                traces = apply_gain(traces, self.gain)
            
            if self.agc_isActive:
                traces = apply_agc_fast(traces, self.agc_window)



            if self.reductVel_isActive:
                if self.reductVel is not None and self.reductVel > 0:
                    traces = apply_reduction_velocity(traces, self.reductVel, self.distTrace)
                    self.status.showMessage(f"Loaded with reduction velocity {self.reductVel} m/s")
                else:
                    self.status.showMessage("Invalid or missing reduction velocity. Skipping reduction.")


            # filtered = bandpass_filter(traces,0.004, self.lowCutFreq, self.midLowCut,self.midHighCut, self.highCutFreq, agc_norm=True)
            # self.traces_cache = {i: filtered[i - x_min] for i in range(x_min, x_max)}
            # self.loaded_start, self.loaded_end = x_min, x_max


                        # --- trapezoidal band‑pass, SU style -------------------------------
            f1, f2, f3, f4 = self.lowCutFreq, self.midLowCut, self.midHighCut, self.highCutFreq

            # pretty version for wiggle plot (unit amplitude)
            filtered_display  = bandpass_filter(traces, DT, f1, f2, f3, f4, agc_norm=True)
            # true‑amplitude version for analysis
            filtered_analysis = bandpass_filter(traces, DT, f1, f2, f3, f4, agc_norm=False)

            # use each where it belongs
            self.traces_cache    = {i: filtered_display[i - x_min]  for i in range(x_min, x_max)}
            self.analysis_cache  = {i: filtered_analysis[i - x_min] for i in range(x_min, x_max)}
            self.plot_traces()

            # ---- refresh navigator --------------------------------------------------
            if self.time is not None:
                #  a very light decimation so the overview is fast:
                dec_step = max(1, traces.shape[0] // 600)        # ≈ 600 wiggles max
                prev = traces[::dec_step]                        # (Ndec, Nsamples)

                # shift each wiggle to its absolute trace‑index:
                offs = np.arange(x_min, x_max, dec_step, dtype=float)[:, None]
                prev_disp = prev * WIGGLE_SCALE + offs           # shape: (Ndec, Nsamples)

                # make an image: x ≡ trace index, y ≡ sample
                # (we flip vertically so time increases downward like the main plot)
                img = np.flipud(prev_disp)
                self.nav_img.setImage(img, autoLevels=False, levels=(np.min(img), np.max(img)))

                # scale the image so one pixel in x == one trace, and y == DT seconds
                self.nav_img.resetTransform()
                self.nav_img.setTransform(pg.QtGui.QTransform().scale(1, DT))


                # lock the overview box to full data extents once
                self.nav_viewbox.setRange(
                    xRange=[0, self.total_traces],
                    yRange=[0, self.time[-1]],
                    padding=0
                )

                # Update the red rectangle
                self._update_nav_rect()

        except Exception as e:
            self.status.showMessage(f"Load failed: {e}")
            traceback.print_exc()



    # ------------------------------------------------------------------
    def _plot_resize_event(self, evt):
        """Keep the navigator glued to bottom‑left inside the plot."""
        sz = self.plot_widget.size()
        # 10 px margin from left & bottom
        self.nav_proxy.setPos(10, sz.height() - self.nav_widget.height() - 10)
        # don’t forget the original resize default
        pg.PlotWidget.resizeEvent(self.plot_widget, evt)

    def _update_nav_rect(self):
        (xmin, xmax), (ymin, ymax) = self.viewbox.viewRange()
        # note: nav image is flipped vertically → same y values still work
        self.nav_rect.setRect(xmin, ymin, xmax - xmin, ymax - ymin)



    def plot_traces(self):
        self.plot_widget.clear()
        for i, trace in self.traces_cache.items():
            self.plot_widget.plot(trace * WIGGLE_SCALE + i, self.time, pen=pg.mkPen('k', width=0.5))
        for roi in self.roi_lines:
            self.viewbox.addItem(roi)
        self.plot_snapped_points()

    def plot_snapped_points(self):
        for item in self.point_items:
            self.plot_widget.removeItem(item)
        self.point_items.clear()

        for x, y in self.snap_points:
            point = pg.ScatterPlotItem([x], [y], symbol='o', brush='deeppink', pen='black', size=10)
            point.sigClicked.connect(self.make_point_selectable(point))
            self.point_items.append(point)
            self.plot_widget.addItem(point)

        for x, y in self.added_points:
            point = pg.ScatterPlotItem([x], [y], symbol='x', brush='magenta', pen='black', size=10)
            point.sigClicked.connect(self.make_point_selectable(point))
            self.point_items.append(point)
            self.plot_widget.addItem(point)

    def make_point_selectable(self, item):
        def on_click(plot, pts):
            for pt in self.point_items:
                pt.setSize(10)
            item.setSize(14)
            self.selected_point = item
        return on_click

    def scroll_to_range(self):
        start = self.start_spin.value()
        end = self.end_spin.value()
        if start >= end:
            self.status.showMessage("Invalid start/end range.")
            return

        # Apply user-entered velocity if provided
        vel_text = self.velocity_input.text().strip()
        if vel_text:
            try:
                velocity = float(vel_text)
                if velocity <= 0:
                    raise ValueError("Velocity must be positive.")
                self.reductVel = velocity
                self.reductVel_isActive = True
            except ValueError:
                self.status.showMessage("Invalid velocity input. Enter a number in m/s.")
                return
        else:
            self.reductVel_isActive = False  # disable reduction if nothing is entered

        # ⬇⬇ FORCE RELOAD when new velocity entered
        self.loaded_start = 0
        self.loaded_end = 0
        self.traces_cache.clear()
        self.load_visible_data()

        self.viewbox.setXRange(start, end, padding=0.05)

    def toggle_draw_mode(self, state):
        self.draw_mode = bool(state)
        self.current_point = None

    def toggle_add_mode(self, state):
        self.add_mode = bool(state)

    def handle_mouse_click(self, event):
        if event.button() != Qt.LeftButton or not self.traces_cache:
            return
        pos = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
        x, y = pos.x(), pos.y()

        if self.add_mode:
            self.store_history()
            nearest_trace = int(round(x))
            self.added_points.append((nearest_trace, y))
            self.plot_snapped_points()
        elif self.draw_mode:
            if self.current_point is None:
                self.current_point = (x, y)
            else:
                x0, y0 = self.current_point
                try:
                    line = pg.LineSegmentROI([[x0, y0], [x, y]], pen=pg.mkPen('b', width=2))
                except Exception as e:
                    self.status.showMessage(f"Error drawing line: {e}")
                self.viewbox.addItem(line)
                self.roi_lines.append(line)
                self.current_point = (x, y)

    def snap_to_peaks(self):
        if not self.traces_cache or not self.roi_lines:
            return
        self.store_history()
        self.snap_points.clear()
        for roi in self.roi_lines:
            pos = roi.getSceneHandlePositions()
            if len(pos) != 2:
                continue
            (x0, y0), (x1, y1) = [self.plot_widget.plotItem.vb.mapSceneToView(
                pg.QtCore.QPointF(p[1].x(), p[1].y())).toTuple() for p in pos]
            if x1 < x0:
                x0, x1 = x1, x0
                y0, y1 = y1, y0
            for ix in range(int(x0), int(x1) + 1):
                if ix in self.traces_cache:
                    t = (ix - x0) / (x1 - x0) if abs(x1 - x0) > 1e-6 else 0.5
                    y_interp = (1 - t) * y0 + t * y1
                    idx = int(y_interp / DT)
                    win = int(0.25 / DT)
                    s, e = max(0, idx - win), min(len(self.traces_cache[ix]), idx + win)
                    peak_idx = find_first_peak(self.traces_cache[ix], s, e)
                    self.snap_points.append((ix, peak_idx * DT))
        self.plot_snapped_points()

    def save_csv(self):
        if not self.snap_points and not self.added_points:
            return
        default_dir = self.output_dir or "."
        name, ok = QInputDialog.getText(self, "Save CSV", "Enter filename:")
        if not ok or not name.strip():
            return
        if not name.endswith(".csv"):
            name += ".csv"
        full_path = os.path.join(default_dir, name)
        try:
            with open(full_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Trace Index", "Time", "Type"])
                for x, y in self.snap_points:
                    writer.writerow([x, round(y, 6), "snapped"])
                for x, y in self.added_points:
                    writer.writerow([x, round(y, 6), "added"])
            self.status.showMessage(f"Saved to {full_path}")
        except Exception as e:
            self.status.showMessage(f"Failed to save: {e}")


    def open_spectral_analysis(self):
        from seismicFreqAnal import SpectralAnalysisWindow

        if not getattr(self, "analysis_cache", None):
            self.status.showMessage("No data loaded to analyze.")
            return

        data = np.array([self.analysis_cache[i] for i in sorted(self.analysis_cache)])
        method = self.spectral_dropdown.currentText()
        win = SpectralAnalysisWindow(method, data, self.time, parent=self)
        win.exec()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SeismicViewer()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())
