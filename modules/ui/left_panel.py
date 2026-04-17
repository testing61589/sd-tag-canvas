"""Left panel (image list) management for the Image Tag Editor."""

from pathlib import Path
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QLabel, QPushButton, QCheckBox, QWidget, QVBoxLayout, QListView
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QSize, Qt, QTimer

from config import (
    IMAGE_EXTENSIONS, LEFT_PANEL_MIN_WIDTH,
    LEFT_PANEL_DEFAULT_WIDTH, THUMBNAIL_SIZE, THUMBNAIL_MODE_ENABLED
)


class LeftPanelManager:
    """Manages the left panel with image list and folder navigation."""

    def __init__(self, parent, thumbnail_mode: bool = THUMBNAIL_MODE_ENABLED):
        self.parent = parent
        self.thumbnail_mode = thumbnail_mode
        self.image_extensions = IMAGE_EXTENSIONS
        self.current_folder = None

        # Widgets
        self.left_widget = None
        self.image_list = None
        self.open_folder_btn = None
        self.thumbnail_checkbox = None

    def create_left_panel(self, splitter) -> QWidget:
        """Create and return the left panel widget."""
        # Left panel container
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)

        # Title
        title_label = QLabel("Image Files")
        title_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
        left_layout.addWidget(title_label)

        # Open folder button
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.parent.open_folder)
        left_layout.addWidget(self.open_folder_btn)

        # Thumbnail mode checkbox
        self.thumbnail_checkbox = QCheckBox("Thumbnail View")
        self.thumbnail_checkbox.setChecked(self.thumbnail_mode)
        self.thumbnail_checkbox.toggled.connect(self.toggle_thumbnail_mode)
        left_layout.addWidget(self.thumbnail_checkbox)

        # Image list
        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(LEFT_PANEL_MIN_WIDTH)
        self.image_list.setSelectionMode(QListWidget.SingleSelection)
        left_layout.addWidget(self.image_list)

        splitter.addWidget(self.left_widget)

        return self.left_widget

    def toggle_thumbnail_mode(self, checked: bool):
        """Toggle thumbnail mode and refresh image list."""
        self.thumbnail_mode = checked
        if self.current_folder:
            self.refresh_image_list()

    def load_images(self):
        """Load all images from the current folder."""
        self.image_list.clear()

        if not self.current_folder:
            return

        if self.thumbnail_mode:
            self._setup_thumbnail_mode()
        else:
            self._setup_list_mode()

        # Find all image files in the folder
        image_files = []
        for file_path in self.current_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                image_files.append(file_path)

        # Sort files by name
        image_files.sort(key=lambda x: x.name.lower())

        # Add to list widget
        for image_path in image_files:
            self._add_image_item(image_path)

        # Select the first image if any images were found
        if image_files:
            self.image_list.setCurrentRow(0)
            first_item = self.image_list.item(0)
            if first_item:
                self.parent.on_image_selected(first_item)
        else:
            self.parent.filename_label.setText("No image files found")
            self.parent.statusBar().showMessage("No image files found in the selected folder")

        if image_files:
            QTimer.singleShot(3000, lambda: self.parent.statusBar().showMessage(
                f"Found {len(image_files)} image files - Use ↑/↓ keys to navigate, Ctrl+wheel to zoom"
            ))

    def _setup_thumbnail_mode(self):
        """Configure list widget for thumbnail view."""
        self.image_list.setWrapping(True)
        self.image_list.setWordWrap(True)
        self.image_list.setTextElideMode(Qt.ElideRight)
        self.image_list.setViewMode(QListWidget.IconMode)
        self.image_list.setResizeMode(QListView.Adjust)
        self.image_list.setFlow(QListView.LeftToRight)
        self.image_list.setSpacing(4)
        self.image_list.setUniformItemSizes(True)
        self.image_list.setGridSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.image_list.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))

    def _setup_list_mode(self):
        """Configure list widget for list view."""
        self.image_list.setViewMode(QListWidget.ListMode)
        self.image_list.setWrapping(False)
        self.image_list.setWordWrap(False)
        self.image_list.setTextElideMode(Qt.ElideRight)
        self.image_list.setResizeMode(QListView.Adjust)
        self.image_list.setFlow(QListView.TopToBottom)
        self.image_list.setSpacing(0)
        self.image_list.setGridSize(QSize())
        self.image_list.setUniformItemSizes(True)
        self.image_list.setIconSize(QSize(0, 0))

    def _add_image_item(self, image_path: Path):
        """Add a single image item to the list."""
        item = QListWidgetItem(image_path.name)
        item.setData(Qt.UserRole, str(image_path))
        if self.thumbnail_mode:
            try:
                pixmap = QPixmap(str(image_path)).scaled(
                    THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                item.setIcon(QIcon(pixmap))
            except Exception:
                pass  # Skip invalid images
            item.setTextAlignment(Qt.AlignHCenter)
            item.setSizeHint(QSize(THUMBNAIL_SIZE + 20, 120))
        else:
            item.setIcon(QIcon())
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            item.setSizeHint(QSize(250, 20))
        self.image_list.addItem(item)

    def refresh_image_list(self):
        """Refresh the image list and try to preserve current selection."""
        current_path = self.parent.current_image_path
        self.load_images()

        if current_path and current_path.exists():
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if Path(item.data(Qt.UserRole)) == current_path:
                    self.image_list.setCurrentRow(i)
                    self.parent.on_image_selected(item)
                    return

        # Select first image if available
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
            self.parent.on_image_selected(self.image_list.item(0))
        else:
            # No images left
            self._clear_all_state()

    def _clear_all_state(self):
        """Clear all image state when no images remain."""
        self.parent.image_canvas._clear_image_data()
        self.parent.filename_label.setText("No image selected")
        self.parent.image_canvas.update()
        self.parent.tags_manager.tags_edit.clear()
        self._disable_zoom_controls()
        self.parent.tags_manager.tags_edit.setEnabled(False)
        self.parent.tags_manager.save_changes_btn.setEnabled(False)
        self.parent.paint_tools_manager.delete_btn.setEnabled(False)
        self.parent.paint_tools_manager.duplicate_btn.setEnabled(False)
        self.parent.current_image_path = None
        self.parent.statusBar().showMessage("No more images in folder")

    def _disable_zoom_controls(self):
        """Disable zoom controls when no image is loaded."""
        self.parent.zoom_in_btn.setEnabled(False)
        self.parent.zoom_out_btn.setEnabled(False)
        self.parent.reset_zoom_btn.setEnabled(False)
        self.parent.fit_window_btn.setEnabled(False)
        self.parent.zoom_label.setText("—")
