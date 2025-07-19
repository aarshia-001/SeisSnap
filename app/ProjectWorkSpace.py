import os
import importlib.util
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QListWidget, QListWidgetItem, QListView, QMessageBox
)
from PySide6.QtCore import Qt

#absolute path to 'analysisModules'
import sys

def get_module_dir():
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller: analysisModules bundled next to the EXE
        return os.path.join(sys._MEIPASS, "analysisModules")
    else:
        # Dev mode: analysisModules is sibling to app/
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root_dir, "analysisModules")

MODULE_DIR = get_module_dir()


class WorkspaceWidget(QWidget):
    def __init__(self, project_csv_paths=None, csv_provider_callback=None):
        super().__init__()
        self.project_csv_paths = project_csv_paths or []
        self.csv_provider_callback = csv_provider_callback
        self.loaded_modules = {}      # name -> {module, settings}
        self.available_modules = {}   # folder_name -> Python module
        self.module_folder = MODULE_DIR

        # ----- Layouts -----
        main_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()

        # ----- Dropdown + Buttons -----
        self.analysis_dropdown = QComboBox()
        self.load_available_modules()
        self.analysis_dropdown.addItem("Select Analysis")
        self.analysis_dropdown.addItems(self.available_modules.keys())

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_analysis)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_selected_analysis)

        controls_layout.addWidget(QLabel("Add Tool:"))
        controls_layout.addWidget(self.analysis_dropdown)
        
        add_btn.setStyleSheet("font-size: 12px; padding: 5px;")
        delete_btn.setStyleSheet("font-size: 12px; padding: 5px;")
        controls_layout.addWidget(add_btn)
        controls_layout.addWidget(delete_btn)

        # ----- Workspace List -----
        self.analysis_list = QListWidget()
        self.analysis_list.setDragDropMode(QListView.InternalMove)

        # ----- Analyse Button -----
        self.analyse_btn = QPushButton("Analyse")
        self.analyse_btn.setStyleSheet(" QPushButton{font-size: 15px; padding: 10px;} QPushButton:hover{font-size: 15px; padding: 10px; background: '#ddd'}")
        self.analyse_btn.clicked.connect(self.run_analysis)

        # ----- Layout Assembly -----
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(QLabel("🔧 Workspace Modules:"))
        main_layout.addWidget(self.analysis_list)
        main_layout.addWidget(self.analyse_btn)

        self.setLayout(main_layout)

    def load_available_modules(self):
        """Load available module folders dynamically from analysisModules/"""
        if not os.path.exists(self.module_folder):
            print("Module folder not found:", self.module_folder)
            return

        for name in os.listdir(self.module_folder):
            folder_path = os.path.join(self.module_folder, name)
            script_path = os.path.join(folder_path, "main.py")
            if os.path.isdir(folder_path) and os.path.exists(script_path):
                spec = importlib.util.spec_from_file_location(name, script_path)
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    self.available_modules[name] = mod
                except Exception as e:
                    print(f"Failed to load module '{name}': {e}")


    def add_analysis(self):
        selected_name = self.analysis_dropdown.currentText()
        if selected_name == "Select Analysis":
            QMessageBox.warning(self, "Invalid Selection", "Select a module to add.")
            return
        if selected_name in self.loaded_modules:
            QMessageBox.information(self, "Already Added", f"{selected_name} already in workspace.")
            return

        mod = self.available_modules[selected_name]
        settings = {}
        if hasattr(mod, "configure"):
            try:
                settings = mod.configure(parent=self)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error in configure(): {e}")
                return

        self.loaded_modules[selected_name] = {"module": mod, "settings": settings}
        self.analysis_list.addItem(QListWidgetItem(selected_name))

    def delete_selected_analysis(self):
        selected_item = self.analysis_list.currentItem()
        if selected_item:
            name = selected_item.text()
            self.loaded_modules.pop(name, None)
            self.analysis_list.takeItem(self.analysis_list.row(selected_item))
        else:
            QMessageBox.information(self, "No Selection", "Please select a module to delete.")

    def run_analysis(self):
        """
        Execute the workspace pipeline.

        ▶  For each CSV the user selected, build a list:
              items = [{ "path": <str>, "df": <DataFrame> }, …]

        ▶  Iterate over modules in drag‑drop order:
              • If TYPE == "multi":
                     – first try module.run(list[DataFrame], settings)
                     – if that raises, retry with list[str] (paths)
                     – interpret return value (list[DF] | list[str] | None)
              • Else (single‑file):
                     – for each item: try DF first, then path
                     – interpret return value (DF | str | None)

        ▶  After every successful call that ends with a DataFrame in memory
            (either returned or in‑place edit), save it to
            <csv_dir>/analysisResult/<base>_<step>.csv and propagate that path
            to downstream modules.

        A module therefore NEVER needs to know what ran before it: it always
        receives what it wants (DFs or paths) in the user‑defined order.
        """
        import os, pandas as pd
        from traceback import format_exc

        # ---------- helpers -------------------------------------------------
        def next_filename(old_path: str, step: str) -> str:
            base = os.path.basename(old_path).rsplit(".csv", 1)[0]
            safe = step.replace(" ", "_")
            return f"{base}_{safe}.csv"

        def save_df(df: pd.DataFrame, old_path: str, step: str) -> str:
            """Save DF and return new absolute path."""
            out_dir = os.path.join(os.path.dirname(old_path), "analysisResult")
            os.makedirs(out_dir, exist_ok=True)
            new_path = os.path.join(out_dir, next_filename(old_path, step))
            df.to_csv(new_path, index=False)
            return new_path

        # ---------- 1. gather inputs ---------------------------------------
        csv_paths = (self.csv_provider_callback()
                     if self.csv_provider_callback
                     else self.project_csv_paths)
        if not csv_paths:
            QMessageBox.warning(self, "No CSVs",
                                "No CSVs selected to run analysis on.")
            return

        steps = [self.analysis_list.item(i).text()
                 for i in range(self.analysis_list.count())]
        if not steps:
            QMessageBox.information(self, "Empty",
                                     "Add modules before running analysis.")
            return

        items = []
        try:
            for p in csv_paths:
                items.append({"path": p, "df": pd.read_csv(p)})
        except Exception as e:
            QMessageBox.critical(self, "Load error",
                                 f"Failed reading CSVs:\n{e}")
            return

        # ---------- 2. run pipeline ----------------------------------------
        for step in steps:
            info = self.loaded_modules.get(step)
            if not info:          # should never happen
                continue

            mod, settings = info["module"], info["settings"]
            mtype = getattr(mod, "TYPE", "").lower()

            # ===== MULTI‑FILE MODULES ======================================
            if mtype == "multi":
                try:
                    # try with DataFrames first
                    result = mod.run([it["df"] for it in items], settings)
                    used_df_input = True
                except Exception as e_df:
                    try:
                        # fallback → paths
                        result = mod.run([it["path"] for it in items], settings)
                        used_df_input = False
                    except Exception as e_path:
                        QMessageBox.critical(
                            self, "Analysis error",
                            f"[{step}] failed with both input forms:\n"
                            f"--- DataFrame error ---\n{e_df}\n\n"
                            f"--- Path error ---\n{e_path}"
                        )
                        return

                # -------- interpret return value --------------------------
                if used_df_input:
                    # We *started* with DFs
                    if result is None:
                        # in‑place edits → save each DF
                        for it in items:
                            it["path"] = save_df(it["df"], it["path"], step)
                    elif (isinstance(result, list)
                          and len(result) == len(items)
                          and all(isinstance(r, pd.DataFrame) for r in result)):
                        for it, new_df in zip(items, result):
                            it["df"] = new_df
                            it["path"] = save_df(new_df, it["path"], step)
                    else:
                        QMessageBox.critical(
                            self, "Analysis error",
                            f"[{step}] returned an unexpected value.")
                        return
                else:
                    # We *started* with paths
                    if result is None:
                        # assume module mutated files in place → reload
                        try:
                            for it in items:
                                it["df"] = pd.read_csv(it["path"])
                        except Exception as e:
                            QMessageBox.critical(
                                self, "Reload error",
                                f"Could not reload CSVs after [{step}]:\n{e}"
                            )
                            return
                    elif (isinstance(result, list)
                          and len(result) == len(items)
                          and all(isinstance(r, str) for r in result)):
                        for it, new_path in zip(items, result):
                            it["path"] = new_path
                            try:
                                it["df"] = pd.read_csv(new_path)
                            except Exception as e:
                                QMessageBox.critical(
                                    self, "Reload error",
                                    f"[{step}] produced {new_path} "
                                    f"but it could not be read:\n{e}"
                                )
                                return
                    else:
                        QMessageBox.critical(
                            self, "Analysis error",
                            f"[{step}] returned an unexpected value.")
                        return
                continue   # next step in pipeline

            # ===== SINGLE‑FILE MODULES =====================================
            for it in items:
                # -- try DataFrame first ------------------------------------
                try:
                    out = mod.run(it["df"], settings)
                    used_df_input = True
                except Exception as e_df:
                    # fallback → path
                    try:
                        out = mod.run(it["path"], settings)
                        used_df_input = False
                    except Exception as e_path:
                        QMessageBox.critical(
                            self, "Analysis error",
                            f"[{step}] failed for {it['path']}:\n\n"
                            f"--- DataFrame error ---\n{e_df}\n\n"
                            f"--- Path error ---\n{e_path}"
                        )
                        return

                # -- interpret result --------------------------------------
                if used_df_input:
                    if out is None:                 # in‑place edits
                        it["path"] = save_df(it["df"], it["path"], step)
                    elif isinstance(out, pd.DataFrame):
                        it["df"]   = out
                        it["path"] = save_df(out, it["path"], step)
                    elif isinstance(out, str):
                        it["path"] = out
                        try:
                            it["df"] = pd.read_csv(out)
                        except Exception as e:
                            QMessageBox.critical(
                                self, "Reload error",
                                f"{out} could not be read:\n{e}")
                            return
                    else:
                        QMessageBox.critical(
                            self, "Analysis error",
                            f"[{step}] returned unsupported type "
                            f"{type(out)}")
                        return
                else:   # used path input
                    if out is None:
                        # module wrote over same file → reload
                        try:
                            it["df"] = pd.read_csv(it["path"])
                        except Exception as e:
                            QMessageBox.critical(
                                self, "Reload error",
                                f"Could not reload {it['path']}:\n{e}")
                            return
                    elif isinstance(out, pd.DataFrame):
                        it["df"]   = out
                        it["path"] = save_df(out, it["path"], step)
                    elif isinstance(out, str):
                        it["path"] = out
                        try:
                            it["df"] = pd.read_csv(out)
                        except Exception as e:
                            QMessageBox.critical(
                                self, "Reload error",
                                f"{out} could not be read:\n{e}")
                            return
                    else:
                        QMessageBox.critical(
                            self, "Analysis error",
                            f"[{step}] returned unsupported type "
                            f"{type(out)}")
                        return

        # ---------- 3. done -----------------------------------------------
        QMessageBox.information(self, "Done", "Analysis completed.")
