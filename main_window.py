"""
Main window for the Image Tag Editor application with zoom functionality.

This is the entry point for the application and coordinates all UI modules.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QHBoxLayout, QSplitter, QWidget, QVBoxLayout, QFrame,
    QLabel, QTextEdit, QPushButton, QMessageBox
)
from PySide6.QtGui import QAction
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from image_canvas import ImageCanvas
from config import (
    LEFT_PANEL_DEFAULT_WIDTH, RIGHT_PANEL_DEFAULT_WIDTH,
    THUMBNAIL_MODE_ENABLED
)
from modules.ui import (
    MenuManager,
    LeftPanelManager,
    PaintToolsManager,
    TriggerWordsManager,
    TagsManager,
    ImageManager,
)


class ImageTagEditor(QMainWindow):
    """Main application window that coordinates all UI components."""

    def __init__(self, initial_folder=None):
        super().__init__()
        self.current_folder = None
        self.current_image_path = None
        self.image_extensions = None  # Set in init_ui
        self.thumbnail_mode = THUMBNAIL_MODE_ENABLED
        self.image_canvas = None
        self.tags_edit = None
        self.save_changes_btn = None

        # Initialize managers
        self.menu_manager = MenuManager(self)
        self.left_panel_manager = LeftPanelManager(self, thumbnail_mode=THUMBNAIL_MODE_ENABLED)
        self.paint_tools_manager = PaintToolsManager(self)
        self.trigger_words_manager = TriggerWordsManager(self)
        self.tags_manager = TagsManager(self)
        self.image_manager = ImageManager(self)

        # Install event filter to capture arrow keys globally
        self.installEventFilter(self)
        QApplication.instance().installEventFilter(self)

        self.init_ui()
        self.setup_connections()

        # Open initial folder if provided
        if initial_folder:
            QTimer.singleShot(100, lambda: self.open_folder_path(Path(initial_folder)))

    def init_ui(self):
        """Initialize the user interface."""
        self.image_extensions = self.left_panel_manager.image_extensions
        self.setWindowTitle("Image Tag Editor with Paint, Crop and Zoom Tools")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create menu bar
        self.menu_manager.create_menu_bar()

        # Create main horizontal layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Create left panel (image list)
        self.left_panel_manager.create_left_panel(splitter)

        # Create right panel (image display and tools)
        self.create_right_panel(splitter)

        # Set splitter proportions
        splitter.setSizes([LEFT_PANEL_DEFAULT_WIDTH, RIGHT_PANEL_DEFAULT_WIDTH])

        # Status bar
        self.statusBar().showMessage("Ready - Open a folder to start")

    def create_right_panel(self, splitter):
        """Create the right panel with image canvas and tools."""
        # Right panel container
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)

        # Filename label
        self.filename_label = QLabel("No image selected")
        self.filename_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.filename_label.setStyleSheet("color: #ddd; border-bottom: 1px solid #555;")
        right_layout.addWidget(self.filename_label)

        # Image canvas
        self.image_canvas = ImageCanvas()
        right_layout.addWidget(self.image_canvas, 1)

        # Paint tools section
        self.paint_tools_manager.create_paint_tools(right_layout)

        # Trigger words section
        self.trigger_words_manager.create_trigger_words_section(right_layout)

        # Tags section
        self.tags_manager.create_tags_section(right_layout)

        splitter.addWidget(right_widget)

    def setup_connections(self):
        """Set up signal/slot connections."""
        self.left_panel_manager.image_list.itemClicked.connect(self.image_manager.on_image_selected)
        self.tags_manager.tags_edit.textChanged.connect(self.tags_manager.on_tags_changed)
        self.tags_manager.tags_edit.textChanged.connect(self.tags_manager.auto_resize_text_edit)
        self.tags_manager.tags_edit.textChanged.connect(self.tags_manager.update_unsaved_status)

        # Connect zoom signal
        self.image_canvas.zoom_changed.connect(self.update_zoom_display)

        # Connect image needs save signal if available
        if hasattr(self.image_canvas, 'image_needs_save_changed'):
            self.image_canvas.image_needs_save_changed.connect(self.tags_manager.update_unsaved_status)

    def open_folder(self):
        """Open folder dialog and load selected folder."""
        from PySide6.QtWidgets import QFileDialog
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder", "")
        if folder_path:
            self.open_folder_path(Path(folder_path))

    def open_folder_path(self, folder_path: Path):
        """Open a specific folder path."""
        try:
            self.current_folder = Path(folder_path)
            self.left_panel_manager.current_folder = self.current_folder

            # Load trigger words first
            self.trigger_words_manager.load_trigger_words()
            self.trigger_words_manager.refresh_trigger_words_ui()

            # Load images
            self.left_panel_manager.load_images()
            self.statusBar().showMessage(f"Loaded folder: {folder_path}")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error Loading Folder",
                f"Failed to load folder: {folder_path}\n\nError: {str(e)}"
            )
            self.statusBar().showMessage("Failed to load folder")

    # Wrapper methods that delegate to managers
    def update_zoom_display(self, zoom_factor):
        """Update zoom percentage display."""
        try:
            percentage = int(zoom_factor * 100)
            self.paint_tools_manager.zoom_label.setText(f"{percentage}%")

            # Update status message
            if self.current_image_path:
                current_row = self.left_panel_manager.image_list.currentRow()
                total_count = self.left_panel_manager.image_list.count()
                self.statusBar().showMessage(
                    f"Image {current_row + 1} of {total_count}: {self.current_image_path.name} - Zoom: {percentage}%"
                )
        except Exception as e:
            print(f"Error updating zoom display: {e}")

    def highlight_trigger_words(self):
        """Highlight trigger words in tags field."""
        self.trigger_words_manager.highlight_trigger_words()

    # Delegate paint/crop methods to paint_tools_manager
    def choose_color(self):
        self.paint_tools_manager.choose_color()

    def update_color_display(self, color):
        self.paint_tools_manager.update_color_display(color)

    def toggle_dropper(self, checked):
        self.paint_tools_manager.toggle_dropper(checked)

    def toggle_crop_mode(self, checked):
        self.paint_tools_manager.toggle_crop_mode(checked)

    def update_crop_buttons(self):
        self.paint_tools_manager.update_crop_buttons()

    def apply_crop(self):
        self.paint_tools_manager.apply_crop()

    def cancel_crop(self):
        self.paint_tools_manager.cancel_crop()

    def update_brush_size(self, size):
        self.paint_tools_manager.update_brush_size(size)

    def clear_paint(self):
        self.paint_tools_manager.clear_paint()

    def zoom_in(self):
        self.paint_tools_manager.zoom_in()

    def zoom_out(self):
        self.paint_tools_manager.zoom_out()

    def reset_zoom(self):
        self.paint_tools_manager.reset_zoom()

    def fit_to_window(self):
        self.paint_tools_manager.fit_to_window()

    # Delegate utility methods
    def has_unsaved_changes(self):
        """Delegates to tags_manager."""
        return self.tags_manager.has_unsaved_changes()

    def update_unsaved_status(self):
        """Delegates to tags_manager."""
        self.tags_manager.update_unsaved_status()

    # Delegate image methods to image_manager
    def on_image_selected(self, item):
        self.image_manager.on_image_selected(item)

    def navigate_to_image(self, row: int):
        self.image_manager.navigate_to_image(row)

    def prompt_save_before_navigate(self, key):
        return self.image_manager.prompt_save_before_navigate(key)

    def discard_changes(self):
        self.image_manager.discard_changes()

    def duplicate_image(self):
        self.image_manager.duplicate_image()

    def delete_image(self):
        self.image_manager.delete_image()

    def _save_image_pixmap(self, pixmap):
        return self.image_manager._save_image_pixmap(pixmap)

    def save_changes(self):
        """Delegates to tags_manager."""
        from PySide6.QtGui import QTextCursor
        self.tags_manager.save_changes()

    # Delegate load_tags to tags_manager
    def load_tags(self, image_path: Path):
        self.tags_manager.load_tags(image_path)

    # Delegate left_panel methods
    def toggle_thumbnail_mode(self, checked):
        self.left_panel_manager.toggle_thumbnail_mode(checked)

    def refresh_image_list(self):
        """Delegates to left_panel_manager."""
        self.left_panel_manager.refresh_image_list()

    # Delegate trigger_words methods
    def refresh_trigger_words_ui(self):
        """Delegates to trigger_words_manager."""
        self.trigger_words_manager.refresh_trigger_words_ui()

    # Event filter for keyboard navigation
    def eventFilter(self, obj, event):
        """Handle keyboard events for navigation."""
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QTextCursor

        if event.type() == QEvent.KeyPress:
            # Delete key
            if event.key() == Qt.Key_Delete:
                if self.current_image_path and self.paint_tools_manager.delete_btn.isEnabled():
                    self.delete_image()
                return True

            # Arrow keys for navigation
            elif event.key() in (Qt.Key_Up, Qt.Key_Down):
                if self.left_panel_manager.image_list.count() > 0:
                    if self.has_unsaved_changes():
                        if not self.prompt_save_before_navigate(event.key()):
                            return True

                    current_row = self.left_panel_manager.image_list.currentRow()

                    if current_row == -1:
                        new_row = 0
                    elif event.key() == Qt.Key_Up:
                        new_row = max(0, current_row - 1)
                    else:
                        new_row = min(self.left_panel_manager.image_list.count() - 1, current_row + 1)

                    if new_row != current_row:
                        self.navigate_to_image(new_row)

                return True

        return super().eventFilter(obj, event)
