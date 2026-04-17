"""Menu bar management for the Image Tag Editor."""

from PySide6.QtWidgets import QMainWindow, QMenu, QMessageBox
from PySide6.QtGui import QAction

from .image_manager import ImageManager


class MenuManager:
    """Manages menu bar creation and actions."""

    def __init__(self, parent: QMainWindow):
        self.parent = parent
        self.image_manager = ImageManager(parent)

    def create_menu_bar(self):
        """Create the menu bar with all menus and actions."""
        menubar = self.parent.menuBar()

        # File menu
        self._create_file_menu(menubar)

        # Edit menu
        self._create_edit_menu(menubar)

        # View menu
        self._create_view_menu(menubar)

    def _create_file_menu(self, menubar: QMenu):
        """Create the File menu."""
        file_menu = menubar.addMenu('File')

        open_folder_action = QAction('Open Folder', self.parent)
        open_folder_action.setShortcut('Ctrl+O')
        open_folder_action.triggered.connect(self.parent.open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self.parent)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.parent.close)
        file_menu.addAction(exit_action)

    def _create_edit_menu(self, menubar: QMenu):
        """Create the Edit menu."""
        edit_menu = menubar.addMenu('Edit')

        crop_action = QAction('Crop', self.parent)
        crop_action.setShortcut('Ctrl+R')
        crop_action.triggered.connect(self.parent.apply_crop)
        edit_menu.addAction(crop_action)

        delete_action = QAction('Delete Image', self.parent)
        delete_action.setShortcut('Delete')
        delete_action.triggered.connect(self.parent.delete_image)
        edit_menu.addAction(delete_action)

        duplicate_action = QAction('Duplicate Image', self.parent)
        duplicate_action.setShortcut('Ctrl+D')
        duplicate_action.triggered.connect(self.parent.duplicate_image)
        edit_menu.addAction(duplicate_action)

    def _create_view_menu(self, menubar: QMenu):
        """Create the View menu."""
        view_menu = menubar.addMenu('View')

        zoom_in_action = QAction('Zoom In', self.parent)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(self.parent.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction('Zoom Out', self.parent)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.parent.zoom_out)
        view_menu.addAction(zoom_out_action)

        reset_zoom_action = QAction('Reset Zoom', self.parent)
        reset_zoom_action.setShortcut('Ctrl+0')
        reset_zoom_action.triggered.connect(self.parent.reset_zoom)
        view_menu.addAction(reset_zoom_action)

        fit_window_action = QAction('Fit to Window', self.parent)
        fit_window_action.setShortcut('Ctrl+F')
        fit_window_action.triggered.connect(self.parent.fit_to_window)
        view_menu.addAction(fit_window_action)
