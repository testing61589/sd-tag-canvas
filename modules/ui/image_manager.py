"""Image lifecycle management for the Image Tag Editor."""

from pathlib import Path
import shutil
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCharFormat, QColor, QTextCursor


class ImageManager:
    """Manages image loading, saving, duplication, and deletion."""

    def __init__(self, parent):
        self.parent = parent

    def on_image_selected(self, item):
        """Handle when an image is selected from the list."""
        image_path = Path(item.data(Qt.UserRole))
        self.parent.current_image_path = image_path
        self.parent.filename_label.setText(image_path.name)

        # Unload old tags first
        self._unload_tags()

        # Display image in canvas
        success = self.parent.image_canvas.set_image(image_path)
        self.parent.image_canvas.fit_to_window()
        if not success:
            self.parent.filename_label.setText(f"Failed to load: {image_path.name}")
            self.parent.statusBar().showMessage(f"Failed to load image: {image_path.name}")
            self._disable_zoom_controls()
            return

        # Enable zoom controls
        self._enable_zoom_controls()

        # Reset crop mode when switching images
        if self.parent.paint_tools_manager.crop_btn.isChecked():
            self.parent.paint_tools_manager.crop_btn.setChecked(False)

        # Load and display tags
        self.parent.load_tags(image_path)

        # Enable tags editing and action buttons
        self.parent.tags_manager.tags_edit.setEnabled(True)
        self.parent.paint_tools_manager.delete_btn.setEnabled(True)
        self.parent.paint_tools_manager.duplicate_btn.setEnabled(True)

        # Update status bar with position and zoom info
        current_row = self.parent.left_panel_manager.image_list.currentRow()
        total_count = self.parent.left_panel_manager.image_list.count()
        zoom_percentage = self.parent.image_canvas.get_zoom_percentage()
        self.parent.statusBar().showMessage(
            f"Image {current_row + 1} of {total_count}: {image_path.name} - Zoom: {zoom_percentage}%"
        )

    def _unload_tags(self):
        """Unload current tags before loading new ones."""
        if not hasattr(self.parent, 'tags_manager') or not self.parent.tags_manager.tags_edit:
            self.parent.original_tags_content = ""
            return

        self.parent.tags_manager.tags_edit.blockSignals(True)
        plain_format = QTextCharFormat()
        plain_format.setBackground(QColor())
        cursor = QTextCursor(self.parent.tags_manager.tags_edit.document())
        cursor.select(QTextCursor.Document)
        cursor.mergeCharFormat(plain_format)
        self.parent.tags_manager.tags_edit.clear()
        self.parent.tags_manager.tags_edit.blockSignals(False)
        self.parent.original_tags_content = ""

    def _enable_zoom_controls(self):
        """Enable zoom controls when image is loaded."""
        self.parent.paint_tools_manager.zoom_in_btn.setEnabled(True)
        self.parent.paint_tools_manager.zoom_out_btn.setEnabled(True)
        self.parent.paint_tools_manager.reset_zoom_btn.setEnabled(True)
        self.parent.paint_tools_manager.fit_window_btn.setEnabled(True)

    def _disable_zoom_controls(self):
        """Disable zoom controls when no image is loaded."""
        self.parent.paint_tools_manager.zoom_in_btn.setEnabled(False)
        self.parent.paint_tools_manager.zoom_out_btn.setEnabled(False)
        self.parent.paint_tools_manager.reset_zoom_btn.setEnabled(False)
        self.parent.paint_tools_manager.fit_window_btn.setEnabled(False)
        self.parent.paint_tools_manager.zoom_label.setText("—")

    def navigate_to_image(self, row: int):
        """Navigate to the image at the given row index."""
        self.parent.left_panel_manager.image_list.setCurrentRow(row)
        item = self.parent.left_panel_manager.image_list.item(row)
        if item:
            self.on_image_selected(item)

    def prompt_save_before_navigate(self, key):
        """Prompt user to save changes before navigating."""
        msg_box = QMessageBox(self.parent)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Unsaved Changes")
        msg_box.setText("You have unsaved changes.")
        msg_box.setInformativeText("Do you want to save them before navigating?")

        save_btn = msg_box.addButton("&Save", QMessageBox.YesRole)
        discard_btn = msg_box.addButton("&Discard", QMessageBox.NoRole)
        cancel_btn = msg_box.addButton(QMessageBox.Cancel)

        msg_box.setDefaultButton(save_btn)
        msg_box.exec()

        clicked = msg_box.clickedButton()
        if clicked == save_btn:
            self.parent.save_changes()
            return True
        elif clicked == discard_btn:
            self.discard_changes()
            return True
        else:
            return False

    def discard_changes(self):
        """Discard unsaved changes by reloading original state."""
        if self.parent.current_image_path:
            self.parent.load_tags(self.parent.current_image_path)
            if hasattr(self.parent.image_canvas, 'set_image'):
                self.parent.image_canvas.set_image(self.parent.current_image_path)
        self.parent.update_unsaved_status()

    def duplicate_image(self):
        """Duplicate the current image and its associated tag file."""
        if not self.parent.current_image_path or not self.parent.current_image_path.exists():
            QMessageBox.warning(self.parent, "No Image", "No image selected or image file not found.")
            return

        try:
            # Create duplicate image filename
            image_stem = self.parent.current_image_path.stem
            image_suffix = self.parent.current_image_path.suffix
            duplicate_image_path = self.parent.current_image_path.parent / f"{image_stem}_copy{image_suffix}"

            # Handle case where duplicate already exists
            counter = 1
            while duplicate_image_path.exists():
                duplicate_image_path = self.parent.current_image_path.parent / f"{image_stem}_copy{counter}{image_suffix}"
                counter += 1

            # Copy image file
            shutil.copy(str(self.parent.current_image_path), str(duplicate_image_path))

            # Copy tag file if it exists
            tag_file = self.parent.current_image_path.with_suffix('.txt')
            duplicate_tag_path = duplicate_image_path.with_suffix('.txt')
            if tag_file.exists():
                shutil.copy(str(tag_file), str(duplicate_tag_path))

            # Update UI
            self.parent.statusBar().showMessage(f"Duplicated '{self.parent.current_image_path.name}' -> '{duplicate_image_path.name}'")

            # Reload the images list and select the duplicate
            self.parent.refresh_image_list()

        except Exception as e:
            QMessageBox.critical(
                self.parent, "Duplicate Failed",
                f"Failed to duplicate image: {str(e)}"
            )

    def delete_image(self):
        """Move current image and associated tag file to .deleted subfolder."""
        if not self.parent.current_image_path or not self.parent.current_image_path.exists():
            QMessageBox.warning(self.parent, "No Image", "No image selected or image file not found.")
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self.parent, "Confirm Delete",
            f"Move '{self.parent.current_image_path.name}' and its associated tag file to .deleted folder?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Create .deleted subfolder if it doesn't exist
                deleted_folder = self.parent.current_image_path.parent / ".deleted"
                deleted_folder.mkdir(exist_ok=True)

                # Move image file
                deleted_image_path = deleted_folder / self.parent.current_image_path.name
                shutil.move(str(self.parent.current_image_path), str(deleted_image_path))

                # Move tag file if it exists
                tag_file = self.parent.current_image_path.with_suffix('.txt')
                if tag_file.exists():
                    deleted_tag_path = deleted_folder / tag_file.name
                    shutil.move(str(tag_file), str(deleted_tag_path))

                # Update UI
                self.parent.refresh_image_list()
                self.parent.statusBar().showMessage(f"Moved '{self.parent.current_image_path.name}' to .deleted folder")

            except Exception as e:
                QMessageBox.critical(
                    self.parent, "Delete Failed",
                    f"Failed to delete image: {str(e)}"
                )

    def _handle_image_deleted(self):
        """Handle UI updates after image deletion."""
        current_row = self.parent.left_panel_manager.image_list.currentRow()

        # Remove item from list
        self.parent.left_panel_manager.image_list.takeItem(current_row)

        # Select next image if available, otherwise previous
        new_count = self.parent.left_panel_manager.image_list.count()
        if new_count > 0:
            if current_row < new_count:
                new_row = current_row
            else:
                new_row = new_count - 1

            self.parent.left_panel_manager.image_list.setCurrentRow(new_row)
            new_item = self.parent.left_panel_manager.image_list.item(new_row)
            if new_item:
                self.on_image_selected(new_item)
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

    def _save_image_pixmap(self, pixmap):
        """Save a pixmap to the current image path."""
        from config import JPEG_QUALITY
        if not self.parent.current_image_path or not pixmap:
            return False

        file_ext = self.parent.current_image_path.suffix.lower()
        image_format = None
        if file_ext in ['.jpg', '.jpeg']:
            image_format = 'JPEG'
        elif file_ext == '.png':
            image_format = 'PNG'
        elif file_ext == '.bmp':
            image_format = 'BMP'

        if image_format:
            return pixmap.save(str(self.parent.current_image_path), image_format, JPEG_QUALITY)
        else:
            return pixmap.save(str(self.parent.current_image_path))
