"""
nc_pyside_viewer.py
-------------------
PySide6 + PyVista NetCDF DEM viewer

• Handles huge rasters, auto‑projects lon/lat°→m
• Vertical‑exaggeration toggle
• Area analysis (planar & surface)
• **NEW:** Add ▸ Plane  → draggable horizon plane + sidebar live area stats
"""

import os, sys, argparse
from typing import Optional
import numpy as np
import xarray as xr
import pyvista as pv
from PySide6.QtGui import QIcon


def resource_path(relative_path):
    """ Get absolute path to resource, for PyInstaller compatibility """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)




pv.OFF_SCREEN = False
pv.set_plot_theme("document")
os.environ["QT_API"] = "pyside6"

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QWidget, QVBoxLayout, QLabel, QDockWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from pyvistaqt import QtInteractor

# ─── constants ─────────────────────────────────────────────
ELEV_VARS = ["elevation", "z", "topo", "Band1", "hgt", "bed",
             "surface", "topography"]
LAT_VARS  = ["lat", "latitude", "y"]
LON_VARS  = ["lon", "longitude", "x"]

R_EARTH = 6_378_137.0          # m
TARGET_PREVIEW_NPX = 2_048

# ─── helpers ───────────────────────────────────────────────
def pick_var(ds, pool, user=None, kind="variable", ref_shape=None):
    if user and user in ds:
        return ds[user]
    for name in pool:
        if name in ds:
            return ds[name]
    for nm, var in ds.data_vars.items():
        if "elevation" in var.attrs.get("standard_name", "").lower() or \
           "elevation" in var.attrs.get("long_name", "").lower():
            return var
        if ref_shape and var.shape == ref_shape:
            return var
    raise KeyError(f"No suitable {kind} found.")

def lonlat_to_m(lon_deg: np.ndarray, lat_deg: np.ndarray):
    lat0 = np.deg2rad(lat_deg.mean())
    x = np.deg2rad(lon_deg) * R_EARTH * np.cos(lat0)
    y = np.deg2rad(lat_deg) * R_EARTH
    return x, y

def cell_planar_area(lon, lat):
    """Return per‑cell (ny‑1, nx‑1) matrix of planimetric areas in m²."""
    dlon = np.deg2rad(np.diff(lon[0, :])[0])
    dlat = np.deg2rad(np.diff(lat[:, 0])[0])
    row_area = (R_EARTH**2 * dlon * dlat *
                np.cos(np.deg2rad(0.5 * (lat[:-1, 0] + lat[1:, 0]))))
    return np.repeat(row_area[:, None], lon.shape[1]-1, axis=1)

def horizontal_area_fast(cell_area):
    return float(cell_area.sum())
def per_cell_surface_area(lon, lat, elev):
    """
    Return an (ny‑1, nx‑1) array whose element [i,j] is the 3‑D area (m²)
    of the DEM quad bounded by (i,j) … (i+1,j+1).
    Uses dask but materialises the result as a NumPy array once.
    """
    import dask.array as da
    lon = da.from_array(np.deg2rad(lon), chunks="auto")
    lat = da.from_array(np.deg2rad(lat), chunks="auto")
    r   = R_EARTH + da.from_array(elev, chunks="auto")
    x = r * da.cos(lat) * da.cos(lon)
    y = r * da.cos(lat) * da.sin(lon)
    z = r * da.sin(lat)

    A = da.stack([x[:-1, :-1], y[:-1, :-1], z[:-1, :-1]], axis=-1)
    B = da.stack([x[:-1, 1: ], y[:-1, 1: ], z[:-1, 1: ]], axis=-1)
    C = da.stack([x[1: , 1: ], y[1: , 1: ], z[1: , 1: ]], axis=-1)
    D = da.stack([x[1: , :-1], y[1: , :-1], z[1: , :-1]], axis=-1)

    def _cross(u, v):
        return da.stack([
            u[..., 1]*v[..., 2] - u[..., 2]*v[..., 1],
            u[..., 2]*v[..., 0] - u[..., 0]*v[..., 2],
            u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]
        ], axis=-1)

    tri1 = 0.5 * da.linalg.norm(_cross(B - A, C - A), axis=-1)
    tri2 = 0.5 * da.linalg.norm(_cross(C - A, D - A), axis=-1)
    return (tri1 + tri2).compute()          # → NumPy array

# -------------------------------------------------------------------------
def surface_area_dask(lon, lat, elev):
    """True 3‑D area (m²) – no da.cross required."""
    import dask.array as da
    lon = da.from_array(np.deg2rad(lon), chunks="auto")
    lat = da.from_array(np.deg2rad(lat), chunks="auto")
    r   = R_EARTH + da.from_array(elev, chunks="auto")
    x = r * da.cos(lat) * da.cos(lon)
    y = r * da.cos(lat) * da.sin(lon)
    z = r * da.sin(lat)

    A = da.stack([x[:-1,:-1], y[:-1,:-1], z[:-1,:-1]], axis=-1)
    B = da.stack([x[:-1,1:],  y[:-1,1:],  z[:-1,1:]],  axis=-1)
    C = da.stack([x[1:,1:],   y[1:,1:],   z[1:,1:]],   axis=-1)
    D = da.stack([x[1:,:-1],  y[1:,:-1],  z[1:,:-1]],  axis=-1)

    def _cross(u, v):
        return da.stack([
            u[..., 1]*v[..., 2] - u[..., 2]*v[..., 1],
            u[..., 2]*v[..., 0] - u[..., 0]*v[..., 2],
            u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]
        ], axis=-1)

    cross1 = _cross(B - A, C - A)
    cross2 = _cross(C - A, D - A)

    area = 0.5 * (da.linalg.norm(cross1, axis=-1) +
                  da.linalg.norm(cross2, axis=-1))
    return float(area.sum().compute())

# ─── main window ───────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, nc_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("View Terrain🏂🗺️")
        icon_path = resource_path("app/assets/SeisSnap_logo.png")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(1200, 800)

        # full‑res arrays
        self._lon_full = self._lat_full = self._z_full = None
        self._cell_area = None           # per‑cell planar area

        # preview arrays
        self._lon = self._lat = self._z = None
        self._user_exag = 1.0
        self._h2v_scale = 1.0

        self._current_mesh = None
        self._plane_actor = None   # pv.PolyData plane currently in the scene
        self._plane_slider = None  # slider widget
        self._plane_height = None  # un‑scaled height in DEM units (metres)


        # --- UI ----------------------------------------------------------------
        central = QWidget()
        layout = QVBoxLayout(central)
        self._plotter = QtInteractor(central)
        layout.addWidget(self._plotter.interactor)
        self.setCentralWidget(central)

        self._sidebar = QLabel("No plane yet.")
        dock = QDockWidget("Plane Stats", self)
        dock.setWidget(self._sidebar)
        dock.setMinimumWidth(220)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)


        

        self._make_menu()
        if nc_path:
            self.open_nc(nc_path)

    # menu ------------------------------------------------------------------
    def _make_menu(self):

        mb = self.menuBar()
        # File
        fm = mb.addMenu("&File")
        fm.addAction(QAction("Open…", self, triggered=self._browse))
        fm.addSeparator()
        fm.addAction(QAction("Quit", self, triggered=self.close))
        # View
        vm = mb.addMenu("&View")
        vm.addAction(QAction("Reset Camera",
                             self, triggered=lambda: self._plotter.reset_camera()))
        vm.addAction(QAction("Toggle Vertical Exaggeration",
                             self, triggered=self._toggle_exag))
        vis = QAction("Toggle View Visibility", self, checkable=True, checked=True)
        vis.triggered.connect(lambda v: self._plotter.interactor.setVisible(v))
        vm.addAction(vis)
        # Analysis
        am = mb.addMenu("&Analysis")
        am.addAction(QAction("Compute Area…", self, triggered=self._compute_area))
        # Export
        em = mb.addMenu("&Export")
        em.addAction(QAction("Export OBJ…", self, triggered=self._export_obj))
        # Add
        addm = mb.addMenu("&Add")

        # … inside _make_menu after you create addm …
        addm.addAction(QAction("Plane", self, triggered=self._add_plane))
        addm.addAction(QAction("Remove Plane", self, triggered=self._remove_plane))

    # file I/O --------------------------------------------------------------
    def _browse(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open NetCDF DEM", "", "NetCDF (*.nc);;All files (*)")
        if fn:
            self.open_nc(fn)

    def open_nc(self, path):
        try:
            ds = xr.open_dataset(path, chunks="auto")
            lat = pick_var(ds, LAT_VARS, kind="latitude")
            lon = pick_var(ds, LON_VARS, kind="longitude")

            lon2d, lat2d = (np.meshgrid(lon.values, lat.values)
                            if lat.ndim == lon.ndim == 1 else (lon.values, lat.values))

            elev_da = pick_var(ds, ELEV_VARS, kind="elevation",
                               ref_shape=lon2d.shape)

            # full‑res arrays
            self._lon_full, self._lat_full = lon2d, lat2d
            self._z_full = elev_da.compute().values \
                if hasattr(elev_da.data, "compute") else elev_da.values
            self._cell_area = cell_planar_area(lon2d, lat2d)

            # preview stride
            n_pts = lon2d.size
            stride = 100 if n_pts > 5_000_000 else 50 if n_pts > 1_000_000 \
                     else 10 if n_pts > 250_000 else 1
            if stride == 1 and max(lon2d.shape) > TARGET_PREVIEW_NPX:
                stride = max(lon2d.shape) // TARGET_PREVIEW_NPX + 1

            self._lon = lon2d[::stride, ::stride]
            self._lat = lat2d[::stride, ::stride]
            self._z   = self._z_full[::stride, ::stride]
            self._surface_cell_area = per_cell_surface_area(lon2d, lat2d, self._z_full)
            self._surface_area_total = float(self._surface_cell_area.sum())


            # h:v scale
            x_m, y_m = lonlat_to_m(self._lon, self._lat)
            horiz_span = max(x_m.ptp(), y_m.ptp())
            z_range = np.nanmax(self._z) - np.nanmin(self._z)
            self._h2v_scale = 0 if z_range == 0 else horiz_span / (20 * z_range)

            self._update_plot()
            self.statusBar().showMessage(
                f"{os.path.basename(path)}  stride {stride}  "
                f"({self._lon.shape[0]}×{self._lon.shape[1]})")

            ds.close()
        except Exception as e:
            QMessageBox.critical(self, "Open error", str(e))

    # ----------------------------------------------------------------------
    def _update_plot(self):
        if self._lon is None:
            return

        # ------------------------------------------------------------------ mesh
        x_m, y_m = lonlat_to_m(self._lon, self._lat)
        z_m = self._z * self._h2v_scale * self._user_exag

        grid = pv.StructuredGrid(
            x_m.astype(np.float32),
            y_m.astype(np.float32),
            z_m.astype(np.float32),
        )
        grid["elev"] = self._z.ravel(order="F")

        poly = grid.extract_surface().triangulate()
        try:
            dec = poly.decimate(0.5);  dec["elev"] = poly["elev"]
        except Exception:
            dec = poly
        self._current_mesh = dec

        # ---------------------------------------------------------------- window
        self._plotter.clear()
        self._plotter.clear_slider_widgets()         # drop stale slider widgets

        zmin, zmax = float(np.nanmin(self._z)), float(np.nanmax(self._z))
        self._plotter.add_mesh(
            dec, scalars="elev", cmap="terrain",
            clim=(zmin, zmax), nan_color="black",
            smooth_shading=True,
            scalar_bar_args=dict(title="Elevation (m)", n_labels=5),
        )

        # ---------------------------------------------------------------- plane
        if self._plane_actor is not None and self._plane_height is not None:
            # re‑add actor
            self._plotter.add_mesh(self._plane_actor, color="cyan",
                                opacity=0.4, nan_color="cyan")
            # update its Z to current exaggeration scale
            self._refresh_plane_position()
            # recreate slider so it remains functional
            zmin, zmax = float(np.nanmin(self._z_full)), float(np.nanmax(self._z_full))
            self._plane_slider = self._plotter.add_slider_widget(
                callback=self._move_plane,
                rng=[zmin, zmax],
                value=self._plane_height,
                title="Plane height (m)",
                pointa=(0.025, 0.1), pointb=(0.35, 0.1),
                style="modern",
            )

        # ---------------------------------------------------------------- camera
        self._plotter.view_isometric()
        self._plotter.reset_camera()



    def _refresh_plane_position(self):
        """Move existing plane to follow current h:v scale / exaggeration."""
        if self._plane_actor and self._plane_height is not None:
            z_scaled = self._plane_height * self._h2v_scale * self._user_exag
            # update Z in‑place (plane has only one unique Z)
            self._plane_actor.points[:, 2] = z_scaled
            self._plotter.render()

    # ----------------------------------------------------------------------
    def _toggle_exag(self):
        # flip factor
        self._user_exag = 5.0 if self._user_exag == 1.0 else 1.0
        # redraw DEM
        self._update_plot()
        # move plane (if any) to new scale
        self._refresh_plane_position()
        # update sidebar numbers
        if self._plane_height is not None:
            self._update_plane_stats(self._plane_height)

    # ----------------------------------------------------------------------
    def _export_obj(self):
        if self._current_mesh is None:
            QMessageBox.warning(self, "No mesh", "Render first.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export OBJ", "", "OBJ (*.obj)")
        if fn:
            try:
                self._current_mesh.save(fn)
                QMessageBox.information(self, "Exported", fn)
            except Exception as e:
                QMessageBox.critical(self, "Export error", str(e))

    # ----------------------------------------------------------------------
    def _compute_area(self):
        if self._lon_full is None:
            QMessageBox.warning(self, "No data", "Load a DEM first.")
            return
        try:
            planar = horizontal_area_fast(self._cell_area)
            surf   = surface_area_dask(self._lon_full, self._lat_full, self._z_full)
            QMessageBox.information(
                self, "Area",
                f"Planimetric: {planar/1e6:,.2f} km²\n"
                f"Surface:     {surf/1e6:,.2f} km²")
        except Exception as e:
            QMessageBox.critical(self, "Area error", str(e))

    # ─────────── Plane tool ─────────────────────────────────
    def _add_plane(self):
        if self._current_mesh is None:
            QMessageBox.warning(self, "No DEM", "Load and render a surface first.")
            return

        # Remove any previous plane + slider
        self._remove_plane()

        # Height to start at = mean elevation of full‑res DEM
        z0 = float(np.nanmean(self._z_full))
        self._plane_height = z0

        # ----- build plane actor centred on the DEM --------------------------
        b = self._current_mesh.bounds
        x_mid, y_mid = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
        plane = pv.Plane(
            center=(x_mid, y_mid, z0 * self._h2v_scale * self._user_exag),
            direction=(0, 0, 1),
            i_size=b[1] - b[0],
            j_size=b[3] - b[2],
        )
        self._plane_actor = plane
        # add with nan‑safe colour so it never “holes” over missing DEM data
        self._plotter.add_mesh(plane, color="cyan", opacity=0.4, nan_color="cyan")

        # ----- slider widget --------------------------------------------------
        zmin, zmax = float(np.nanmin(self._z_full)), float(np.nanmax(self._z_full))
        self._plane_slider = self._plotter.add_slider_widget(
            callback=self._move_plane,
            rng=[zmin, zmax],
            value=z0,
            title="Plane height (m)",
            pointa=(0.025, 0.1), pointb=(0.35, 0.1),
            style="modern",
        )

        self._update_plane_stats(z0)
        self._plotter.render()            # immediate visual feedback



    def _remove_plane(self):
        if self._plane_actor:
            self._plotter.remove_actor(self._plane_actor)
            self._plane_actor = None
        if self._plane_slider:
            self._plotter.remove_slider_widget(self._plane_slider)
            self._plane_slider = None
        self._plane_height = None
        self._sidebar.setText("No plane yet.")
        self._plotter.render()



    def _create_plane(self, height):
        """Create/replace a horizontal plane at `height` (DEM metres)."""
        self._plane_height = height                       # ↞ store raw height
        z_scaled = height * self._h2v_scale * self._user_exag

        b = self._current_mesh.bounds
        x_mid, y_mid = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2   # mesh centre

        plane = pv.Plane(
            center=(x_mid, y_mid, z_scaled),
            direction=(0, 0, 1),
            i_size=b[1] - b[0],
            j_size=b[3] - b[2],
        )

        if self._plane_actor:
            self._plotter.remove_actor(self._plane_actor)
        self._plane_actor = plane
        self._plotter.add_mesh(plane, color="cyan", opacity=0.4)
        self._plotter.render() 


    def _move_plane(self, value):
        self._plane_height = value
        z_scaled = value * self._h2v_scale * self._user_exag
        self._plane_actor.points[:, 2] = z_scaled   # move Z only
        self._plotter.render()                      #  ← ADDED
        self._update_plane_stats(value)



    def _update_plane_stats(self, z_val):
        if self._surface_cell_area is None:
            return
        mask = self._z_full[:-1, :-1] > z_val          # same shape as cell areas
        area_above = float(self._surface_cell_area[mask].sum())
        area_below = self._surface_area_total - area_above
        self._sidebar.setText(
            f"<b>Plane @ {z_val:,.0f} m</b><br>"
            f"Surface area above: {area_above/1e6:,.2f} km²<br>"
            f"Surface area below: {area_below/1e6:,.2f} km²"
        )


# entry -----------------------------------------------------
def main_runner():
    app = QApplication.instance() or QApplication(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("ncfile", nargs="?", help="NetCDF file")
    args = parser.parse_args()

    if not args.ncfile:
        f, _ = QFileDialog.getOpenFileName(None, "Select NetCDF", "",
                                           "NetCDF (*.nc);;All files (*)")
        if not f:
            sys.exit("No file selected")
        args.ncfile = f

    win = MainWindow(args.ncfile)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main_runner()
