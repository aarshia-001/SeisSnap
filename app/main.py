import os
import sys
import shutil

import stat          
import subprocess 

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QInputDialog, QMessageBox, QListWidget, QListWidgetItem,
    QMainWindow, QStackedWidget, QHBoxLayout, QMenu,
    QFileDialog
)
from PySide6.QtCore import Qt, QPoint
from ProjectWindow import ProjectWindow
from PySide6.QtGui import QIcon

from PySide6.QtCore import QStandardPaths

def get_project_root():
    data_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    project_dir = os.path.join(data_dir, "SeisSnapProjects")
    os.makedirs(project_dir, exist_ok=True)
    return project_dir



def resource_path(relative_path):
    """ Get absolute path to resource, for PyInstaller compatibility """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)



class MyProjects(QWidget):
    def __init__(self, switch_to_menu):
        super().__init__()
        self.switch_to_menu = switch_to_menu
        self.project_list = QListWidget()
        self.project_list.setVisible(False)
        self.project_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.project_list.customContextMenuRequested.connect(self.show_folder_context_menu)
        self.project_list.itemDoubleClicked.connect(self.open_project_window)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        title = QLabel("📁 My Projects")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; color: '#fff'; background: '#000'")

        new_project_btn = QPushButton("➕ New Project")
        new_project_btn.clicked.connect(self.create_new_project)

        open_projects_btn = QPushButton("📂 Open Projects")
        open_projects_btn.clicked.connect(self.load_project_list)


        back_btn = QPushButton("⬅ Back to Menu")
        back_btn.clicked.connect(self.switch_to_menu)

        quit_btn = QPushButton("❌ Quit")
        quit_btn.clicked.connect(QApplication.quit)

        for btn in [new_project_btn, open_projects_btn, back_btn, quit_btn]:
            btn.setStyleSheet("font-size: 16px; padding: 8px;")

        layout.addWidget(title)
        layout.addWidget(new_project_btn)
        layout.addWidget(open_projects_btn)
        layout.addWidget(self.project_list)
        layout.addWidget(back_btn)
        layout.addWidget(quit_btn)

        self.setLayout(layout)


    def create_new_project(self):
        name, ok = QInputDialog.getText(self, "New Project", "Enter project name:")
        if ok and name.strip():
            full_path = os.path.join(get_project_root(), name.strip())
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                QMessageBox.information(self, "Success", f"Project '{name}' created.")
            else:
                QMessageBox.warning(self, "Exists", f"Project '{name}' already exists.")

    def load_project_list(self):
        self.project_list.clear()
        self.project_list.setVisible(True)

        folders = [
            d for d in os.listdir(get_project_root())
            if os.path.isdir(os.path.join(get_project_root(), d))
        ]
        if not folders:
            self.project_list.addItem("No projects found.")
        else:
            for name in folders:
                item = QListWidgetItem(f"📁 {name}")
                item.setData(Qt.UserRole, os.path.join(get_project_root(), name))
                self.project_list.addItem(item)

    def open_project_window(self, item):
        project_path = item.data(Qt.UserRole)
        if project_path:
            self.project_window = ProjectWindow(project_path)
            self.project_window.show()

    def show_folder_context_menu(self, position: QPoint):
        item = self.project_list.itemAt(position)
        if not item:
            return

        menu = QMenu()
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        action = menu.exec(self.project_list.viewport().mapToGlobal(position))

        folder_name = item.text().replace("📁 ", "")
        folder_path = os.path.join(get_project_root(), folder_name)

        if action == rename_action:
            new_name, ok = QInputDialog.getText(self, "Rename Project", "Enter new name:", text=folder_name)
            if ok and new_name.strip() and new_name != folder_name:
                new_path = os.path.join(get_project_root(), new_name)
                if os.path.exists(new_path):
                    QMessageBox.warning(self, "Exists", "A project with that name already exists.")
                    return
                try:
                    os.rename(folder_path, new_path)
                    item.setText(f"📁 {new_name}")
                    item.setData(Qt.UserRole, new_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not rename folder:\n{str(e)}")

        elif action == delete_action:
            confirm = QMessageBox.question(
                self, "Delete Project",
                f"Are you sure you want to delete '{folder_name}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if confirm == QMessageBox.Yes:
                try:
                    shutil.rmtree(folder_path)
                    self.project_list.takeItem(self.project_list.row(item))
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not delete folder:\n{str(e)}")


class MainMenu(QWidget):
    def __init__(self, switch_to_menu):
        super().__init__()
        layout = QVBoxLayout()

        label = QLabel("Welcome to the SeisSnap !")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 24px;")
        

        start_btn = QPushButton("Start")
        start_btn.setStyleSheet("font-size: 18px; padding: 10px; background: '#222'; color: '#fff'; border-radius: 8px")
        start_btn.clicked.connect(switch_to_menu)

        layout.addStretch()
        layout.addWidget(label)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.setLayout(layout)


class MenuScreen(QWidget):
    def __init__(self, switch_to_main, parent_window):
        super().__init__()
        self.parent_window = parent_window
        layout = QVBoxLayout()

        projects_btn = QPushButton("📁 My Projects")
        projects_btn.clicked.connect(self.parent_window.show_projects)

        upload_an_module_btn = QPushButton("➕ Upload Analysis Module")
        upload_an_module_btn.clicked.connect(self.update_an_module)

        back_btn = QPushButton("⬅ Back")
        back_btn.clicked.connect(switch_to_main)

        quit_btn = QPushButton("❌ Quit")
        quit_btn.clicked.connect(QApplication.quit)

        for btn in [projects_btn, upload_an_module_btn]:
            btn.setStyleSheet("font-size: 30px; padding: 60px; color: '#fff'; background: '#000'; border-radius: 4px")
            layout.addWidget(btn)

        button_layout = QHBoxLayout()
        for i in [back_btn, quit_btn]:
            i.setStyleSheet("""
                font-size: 16px;
                padding: 8px;
                width: 40px;
                color: #fff;
                background: #333;
                border-radius: 4px;
            """)
            button_layout.addWidget(i)

        layout.addLayout(button_layout)
        self.setLayout(layout)


    def update_an_module(self):
        """
        Pick a folder and install it into  <Seismic App>/analysisModules/<name>.
        After copying, give the current user read/write on every file and
        read/write/execute on every directory; on Windows also strip the
        Zone.Identifier “blocked” flag so Python can import without errors.
        """
        # 1. ask for a directory
        src_dir = QFileDialog.getExistingDirectory(
            self, "Select analysis‑module folder", os.path.expanduser("~")
        )
        if not src_dir:
            return  # user cancelled

        # 2. destination = one level above App.py + /analysisModules
        app_root = os.path.expanduser("~")  # or use AppData for a cleaner approach

        dest_root = os.path.join(app_root, "SeisSnapModules")

        os.makedirs(dest_root, exist_ok=True)
        dest_dir  = os.path.join(dest_root, os.path.basename(src_dir))

        # 3. overwrite check
        if os.path.exists(dest_dir):
            if QMessageBox.question(
                    self, "Overwrite?",
                    f"A module named '{os.path.basename(src_dir)}' exists.\n"
                    "Replace it?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            ) == QMessageBox.No:
                return
            try:
                shutil.rmtree(dest_dir)
            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"Could not remove old module:\n{e}")
                return

        # 4. copy
        try:
            shutil.copytree(src_dir, dest_dir)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to copy module:\n{e}")
            return

        # 5. grant permissions
        def _grant_permissions(root: str):
            for dirpath, dirnames, filenames in os.walk(root):
                # dirs → rwx for user
                os.chmod(dirpath,
                         stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                for f in filenames:
                    fpath = os.path.join(dirpath, f)
                    # files → rw for user
                    os.chmod(fpath,
                             stat.S_IRUSR | stat.S_IWUSR)

                    # Windows: remove Mark‑of‑the‑Web
                    if os.name == "nt":
                        try:
                            subprocess.run(
                                ["powershell", "-Command",
                                 f'if (Test-Path -LiteralPath "{fpath}:Zone.Identifier") '
                                 f'{{ Remove-Item -LiteralPath "{fpath}:Zone.Identifier" }}'],
                                capture_output=True, check=False
                            )
                        except Exception:
                            pass  # best‑effort

        try:
            _grant_permissions(dest_dir)
        except Exception as e:
            QMessageBox.warning(
                self, "Permission Warning",
                f"Module copied but permission adjustment hit a problem:\n{e}"
            )

        # 6. success message
        QMessageBox.information(
            self, "Success",
            f"Module installed to:\n{dest_dir}\n\n"
            "It will appear in the analysis drop‑down automatically."
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SeisSnap ")
        self.resize(600, 400)



        icon_path = resource_path("app/assets/SeisSnap_logo.png")

        self.setWindowIcon(QIcon(icon_path))
        
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.start_screen = MainMenu(self.show_menu)
        self.menu_screen = MenuScreen(self.show_start, self)
        self.projects_screen = MyProjects(self.show_menu)

        self.stack.addWidget(self.start_screen)
        self.stack.addWidget(self.menu_screen)
        self.stack.addWidget(self.projects_screen)

        self.show_start()

    def show_start(self):
        self.stack.setCurrentIndex(0)

    def show_menu(self):
        self.stack.setCurrentIndex(1)

    def show_projects(self):
        self.stack.setCurrentIndex(2)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
