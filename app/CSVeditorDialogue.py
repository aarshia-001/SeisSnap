
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QMessageBox,
    QAbstractItemView, QFileDialog, QToolButton,
    QInputDialog, QComboBox, QWidget, QFormLayout
)
from PySide6.QtGui import QIcon
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtGui import QIcon

class CsvEditorDialog(QDialog):
    def __init__(self, csv_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit CSV")
        self.resize(1000, 650)
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "SeisSnap_logo.png")
        self.setWindowIcon(QIcon(icon_path))
        self.csv_path = csv_path
        self.original_name = os.path.basename(csv_path)
        self.df = pd.read_csv(csv_path)
        self.history = []
        self.future = []

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Rename Section
        rename_layout = QHBoxLayout()
        rename_label = QLabel("Rename File:")
        rename_label.setStyleSheet("padding:10px; font-size: 15px")
        self.rename_input = QLineEdit(self.original_name)
        self.rename_input.setStyleSheet("padding:10px; font-size: 15px")
        rename_layout.addWidget(rename_label)
        rename_layout.addWidget(self.rename_input)
        self.layout.addLayout(rename_layout)

        # Toolbar
        toolbar_layout = QHBoxLayout()

        def make_btn(text, tooltip, callback):
            btn = QToolButton()
            btn.setText(text)
            btn.setToolTip(tooltip)
            btn.clicked.connect(callback)
            return btn

        buttons = [
            make_btn("↑ Asc", "Sort Ascending", self.sort_ascending),
            make_btn("↓ Desc", "Sort Descending", self.sort_descending),
            make_btn("⎀ Dupl Rmv", "Detect & Remove Duplicates", self.remove_duplicates),
            make_btn("↩ Uno", "Undo", self.undo),
            make_btn("↪ Redo", "Redo", self.redo),
            make_btn("❌ Del Nan", "Delete None/Empty Rows", self.delete_empty_rows),
            make_btn("+ Add Col", "Add Column", self.add_column),
            make_btn("- Del Col", "Delete Column", self.delete_column),
            make_btn("📈 Plot Setup", "Plot Data", self.show_plot_dialog),
        ]
        for b in buttons:
            toolbar_layout.addWidget(b)
        self.layout.addLayout(toolbar_layout)

        # Search Box
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.textChanged.connect(self.filter_table)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        self.layout.addLayout(search_layout)

        # Table and Plot in side-by-side layout
        table_plot_layout = QHBoxLayout()

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.df.columns))
        self.table.setHorizontalHeaderLabels(self.df.columns)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setDragDropMode(QAbstractItemView.InternalMove)
        self.table.itemChanged.connect(self.update_plot)
        table_plot_layout.addWidget(self.table, 3)  # Wider

        # Plot
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumWidth(250)
        table_plot_layout.addWidget(self.canvas, 1)  # Thinner

        self.layout.addLayout(table_plot_layout)

        # Delete row button
        delete_btn = QPushButton("Delete Selected Row ⛔")
        delete_btn.setStyleSheet("padding:10px; font-size: 15px")
        delete_btn.clicked.connect(self.delete_selected_row)
        self.layout.addWidget(delete_btn)

        # Bottom Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save Changes")
        cancel_btn = QPushButton("Cancel")
        save_btn.clicked.connect(self.save_changes)
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("padding:10px; font-size: 15px")
        save_btn.setStyleSheet("padding:10px; font-size: 15px")
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        self.layout.addLayout(button_layout)

        self.populate_table()

    def backup_state(self):
        self.history.append(self.df.copy())
        self.future.clear()

    def undo(self):
        if self.history:
            self.future.append(self.df.copy())
            self.df = self.history.pop()
            self.populate_table()

    def redo(self):
        if self.future:
            self.history.append(self.df.copy())
            self.df = self.future.pop()
            self.populate_table()

    def populate_table(self):
        self.table.setColumnCount(len(self.df.columns))
        self.table.setRowCount(len(self.df))
        self.table.setHorizontalHeaderLabels(self.df.columns)
        for row_idx, row in self.df.iterrows():
            for col_idx, value in enumerate(row):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
        self.update_plot()

    def filter_table(self):
        text = self.search_input.text().lower()
        for row in range(self.table.rowCount()):
            match = any(text in (self.table.item(row, col).text().lower() if self.table.item(row, col) else "")
                        for col in range(self.table.columnCount()))
            self.table.setRowHidden(row, not match)

    def delete_selected_row(self):
        selected_row = self.table.currentRow()
        if selected_row >= 0:
            self.backup_state()
            self.df.drop(self.df.index[selected_row], inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self.populate_table()

    def save_changes(self):
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        headers = [self.table.horizontalHeaderItem(i).text() for i in range(cols)]
        new_data = []
        for row in range(rows):
            new_row = []
            for col in range(cols):
                item = self.table.item(row, col)
                new_row.append(item.text() if item else "")
            new_data.append(new_row)

        try:
            new_df = pd.DataFrame(new_data, columns=headers)
            new_df = new_df.apply(pd.to_numeric, errors='ignore')
            new_name = self.rename_input.text().strip()
            if not new_name:
                QMessageBox.warning(self, "Invalid Name", "Filename cannot be empty.")
                return
            if not new_name.endswith(".csv"):
                new_name += ".csv"
            new_path = os.path.join(os.path.dirname(self.csv_path), new_name)

            if new_path != self.csv_path and os.path.exists(new_path):
                QMessageBox.warning(self, "File Exists", f"'{new_name}' already exists.")
                return

            new_df.to_csv(new_path, index=False)
            if new_path != self.csv_path:
                os.remove(self.csv_path)

            QMessageBox.information(self, "Success", f"Saved to '{new_name}'")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def sort_ascending(self):
        col, ok = QInputDialog.getItem(self, "Sort Ascending", "Column:", list(self.df.columns), 0, False)
        if ok:
            self.backup_state()
            self.df.sort_values(by=col, ascending=True, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self.populate_table()

    def sort_descending(self):
        col, ok = QInputDialog.getItem(self, "Sort Descending", "Column:", list(self.df.columns), 0, False)
        if ok:
            self.backup_state()
            self.df.sort_values(by=col, ascending=False, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self.populate_table()

    def remove_duplicates(self):
        cols, ok = QInputDialog.getMultiLineText(self, "Duplicate Check Columns", "Enter comma-separated column names:")
        if ok:
            selected_cols = [c.strip() for c in cols.split(",") if c.strip() in self.df.columns]
            if selected_cols:
                self.backup_state()
                self.df.drop_duplicates(subset=selected_cols, keep='first', inplace=True)
                self.df.reset_index(drop=True, inplace=True)
                self.populate_table()

    def delete_empty_rows(self):
        self.backup_state()
        self.df.dropna(how='any', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.populate_table()

    def add_column(self):
        col_name, ok = QInputDialog.getText(self, "Add Column", "Enter new column name:")
        if ok and col_name:
            self.backup_state()
            self.df[col_name] = ""
            self.populate_table()

    def delete_column(self):
        col, ok = QInputDialog.getItem(self, "Delete Column", "Select column:", list(self.df.columns), 0, False)
        if ok:
            self.backup_state()
            self.df.drop(columns=[col], inplace=True)
            self.populate_table()

    def show_plot_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Plot")
        layout = QFormLayout(dialog)

        x_axis = QComboBox()
        y_axis = QComboBox()
        for col in self.df.columns:
            x_axis.addItem(col)
            y_axis.addItem(col)

        title_input = QLineEdit()
        plot_type = QComboBox()
        plot_type.addItems(["Line", "Scatter"])

        export_btn = QPushButton("Export")
        export_btn.clicked.connect(lambda: self.export_plot())

        layout.addRow("X Axis:", x_axis)
        layout.addRow("Y Axis:", y_axis)
        layout.addRow("Title:", title_input)
        layout.addRow("Plot Type:", plot_type)
        layout.addRow(export_btn)

        def on_apply():
            self.plot_config = {
                "x": x_axis.currentText(),
                "y": y_axis.currentText(),
                "title": title_input.text(),
                "type": plot_type.currentText()
            }
            self.update_plot()
            dialog.accept()

        apply_btn = QPushButton("Plot")
        apply_btn.clicked.connect(on_apply)
        layout.addRow(apply_btn)

        dialog.exec()

    def update_plot(self):
        self.ax.clear()
        if not hasattr(self, "plot_config"):
            self.canvas.draw()
            return

        x_col = self.plot_config.get("x")
        y_col = self.plot_config.get("y")
        if x_col not in self.df.columns or y_col not in self.df.columns:
            return

        x_idx = self.df.columns.get_loc(x_col)
        y_idx = self.df.columns.get_loc(y_col)

        x_vals, y_vals = [], []

        for row in range(self.table.rowCount()):
            try:
                x_item = self.table.item(row, x_idx)
                y_item = self.table.item(row, y_idx)
                if x_item and y_item:
                    x_val = float(x_item.text())
                    y_val = float(y_item.text())
                    x_vals.append(x_val)
                    y_vals.append(y_val)
            except ValueError:
                continue

        if not x_vals or not y_vals:
            self.canvas.draw()
            return

        if self.plot_config.get("type", "Line") == "Line":
            self.ax.plot(x_vals, y_vals)
        else:
            self.ax.scatter(x_vals, y_vals)

        self.ax.set_title(self.plot_config.get("title", ""))
        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.canvas.draw()

    def export_plot(self):
        if not hasattr(self, "plot_config"):
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Plot", "", "PNG Image (*.png);;PDF File (*.pdf)")
        if file_path:
            self.figure.savefig(file_path)
