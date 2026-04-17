"""Paint tools and zoom controls management for the Image Tag Editor."""

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QColorDialog, QMessageBox, QListView, QVBoxLayout, QFrame
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

from config import (
    DEFAULT_PAINT_COLOR, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE,
    PAINT_TOOLS_MAX_HEIGHT, JPEG_QUALITY
)


class PaintToolsManager:
    """Manages paint tools, crop tools, and zoom controls."""

    def __init__(self, parent):
        self.parent = parent
        self.current_paint_color = QColor(*DEFAULT_PAINT_COLOR)
        self.dropper_active = False
        self.crop_active = False

    def create_paint_tools(self, layout):
        """Create paint tools UI and add to layout."""
        # Paint tools frame
        tools_frame = QFrame()
        tools_frame.setFrameStyle(QFrame.Box)
        tools_frame.setMaximumHeight(PAINT_TOOLS_MAX_HEIGHT)
        tools_layout = QVBoxLayout(tools_frame)

        # First row of tools
        row1_layout = QHBoxLayout()

        # Tools label
        tools_label = QLabel("Tools:")
        tools_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        row1_layout.addWidget(tools_label)

        # Color picker button
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(40, 30)
        self.color_btn.setStyleSheet(f"background-color: {self.current_paint_color.name()}; border: 2px solid #333;")
        self.color_btn.clicked.connect(self.choose_color)
        self.color_btn.setToolTip("Choose paint color")
        row1_layout.addWidget(self.color_btn)

        # Color dropper button
        self.dropper_btn = QPushButton("🎨")
        self.dropper_btn.setFixedSize(40, 30)
        self.dropper_btn.setCheckable(True)
        self.dropper_btn.setToolTip("Color dropper - click to pick color from image")
        self.dropper_btn.toggled.connect(self.toggle_dropper)
        row1_layout.addWidget(self.dropper_btn)

        # Crop tool button
        self.crop_btn = QPushButton("✂️")
        self.crop_btn.setFixedSize(40, 30)
        self.crop_btn.setCheckable(True)
        self.crop_btn.setToolTip("Crop tool - drag to select area to crop")
        self.crop_btn.toggled.connect(self.toggle_crop_mode)
        row1_layout.addWidget(self.crop_btn)

        # Brush size controls
        self.create_brush_size_controls(row1_layout)

        row1_layout.addStretch()

        # Second row of tools
        row2_layout = QHBoxLayout()
        self.create_action_buttons(row2_layout)

        # Add zoom controls to second row
        self.create_zoom_controls(row2_layout)

        row2_layout.addStretch()

        tools_layout.addLayout(row1_layout)
        tools_layout.addLayout(row2_layout)

        layout.addWidget(tools_frame)

    def create_brush_size_controls(self, layout):
        """Create brush size controls."""
        # Brush size
        brush_label = QLabel("Size:")
        layout.addWidget(brush_label)

        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
        self.brush_size_slider.setValue(15)
        self.brush_size_slider.setMaximumWidth(100)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)
        layout.addWidget(self.brush_size_slider)

        self.brush_size_label = QLabel("15")
        self.brush_size_label.setMinimumWidth(30)
        layout.addWidget(self.brush_size_label)

    def create_action_buttons(self, layout):
        """Create action buttons for paint and crop operations."""
        # Clear paint button
        clear_paint_btn = QPushButton("Clear Paint")
        clear_paint_btn.clicked.connect(self.parent.clear_paint)
        clear_paint_btn.setToolTip("Clear all paint marks")
        layout.addWidget(clear_paint_btn)

        # Apply crop button
        self.apply_crop_btn = QPushButton("Apply Crop")
        self.apply_crop_btn.clicked.connect(self.parent.apply_crop)
        self.apply_crop_btn.setEnabled(False)
        self.apply_crop_btn.setToolTip("Apply the selected crop area (Ctrl+R)")
        layout.addWidget(self.apply_crop_btn)

        # Cancel crop button
        self.cancel_crop_btn = QPushButton("Cancel Crop")
        self.cancel_crop_btn.clicked.connect(self.parent.cancel_crop)
        self.cancel_crop_btn.setEnabled(False)
        self.cancel_crop_btn.setToolTip("Cancel crop selection")
        layout.addWidget(self.cancel_crop_btn)

        # Delete button
        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.clicked.connect(self.parent.delete_image)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setToolTip("Move image and tags to .deleted folder (Delete key)")
        layout.addWidget(self.delete_btn)

        # Duplicate button
        self.duplicate_btn = QPushButton("Duplicate")
        self.duplicate_btn.clicked.connect(self.parent.duplicate_image)
        self.duplicate_btn.setEnabled(False)
        self.duplicate_btn.setToolTip("Duplicate image and tag file")
        layout.addWidget(self.duplicate_btn)

    def create_zoom_controls(self, layout):
        """Create zoom controls."""
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Zoom label
        zoom_label = QLabel("Zoom:")
        layout.addWidget(zoom_label)

        # Zoom out button
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_out_btn.clicked.connect(self.parent.zoom_out)
        self.zoom_out_btn.setToolTip("Zoom out (Ctrl+-)")
        self.zoom_out_btn.setEnabled(False)
        layout.addWidget(self.zoom_out_btn)

        # Zoom percentage display
        self.zoom_label = QLabel("—")
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet("border: 1px solid #666; padding: 2px;")
        layout.addWidget(self.zoom_label)

        # Zoom in button
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_in_btn.clicked.connect(self.parent.zoom_in)
        self.zoom_in_btn.setToolTip("Zoom in (Ctrl++)")
        self.zoom_in_btn.setEnabled(False)
        layout.addWidget(self.zoom_in_btn)

        # Reset zoom button
        self.reset_zoom_btn = QPushButton("1:1")
        self.reset_zoom_btn.setFixedSize(35, 30)
        self.reset_zoom_btn.clicked.connect(self.parent.reset_zoom)
        self.reset_zoom_btn.setToolTip("Reset zoom to 100% (Ctrl+0)")
        self.reset_zoom_btn.setEnabled(False)
        layout.addWidget(self.reset_zoom_btn)

        # Fit to window button
        self.fit_window_btn = QPushButton("Fit")
        self.fit_window_btn.setFixedSize(35, 30)
        self.fit_window_btn.clicked.connect(self.parent.fit_to_window)
        self.fit_window_btn.setToolTip("Fit image to window (Ctrl+F)")
        self.fit_window_btn.setEnabled(False)
        layout.addWidget(self.fit_window_btn)

        # Update zoom controls based on current state
        self._update_zoom_controls()

    def choose_color(self):
        """Open color chooser dialog."""
        color = QColorDialog.getColor(self.current_paint_color, self.parent, "Choose Paint Color")
        if color.isValid():
            self.current_paint_color = color
            self.update_color_display(color)

    def update_color_display(self, color: QColor):
        """Update the color button display."""
        self.current_paint_color = color
        self.color_btn.setStyleSheet(f"background-color: {color.name()}; border: 2px solid #333;")

    def toggle_dropper(self, checked: bool):
        """Toggle color dropper mode."""
        if checked:
            # Disable crop mode if it's active
            self.crop_btn.setChecked(False)

        self.parent.image_canvas.set_dropper_mode(checked)
        if checked:
            self.dropper_btn.setText("🎯")
            self.parent.statusBar().showMessage("Dropper active - click on image to pick color")
        else:
            self.dropper_btn.setText("🎨")
            self.parent.statusBar().showMessage("Paint mode active")

    def toggle_crop_mode(self, checked: bool):
        """Toggle crop mode."""
        if checked:
            # Disable dropper mode if it's active
            self.dropper_btn.setChecked(False)

        self.parent.image_canvas.set_crop_mode(checked)
        self.apply_crop_btn.setEnabled(checked)
        self.cancel_crop_btn.setEnabled(checked)

        if checked:
            self.parent.statusBar().showMessage("Crop mode active - drag handles to adjust crop area")
        else:
            self.parent.statusBar().showMessage("Paint mode active")

    def update_crop_buttons(self):
        """Update crop button states based on selection."""
        if self.parent.image_canvas.crop_active:
            has_selection = self.parent.image_canvas.has_crop_selection_ready()
            self.apply_crop_btn.setEnabled(has_selection)

    def update_brush_size(self, size: int):
        """Update brush size."""
        self.brush_size_label.setText(str(size))
        self.parent.image_canvas.set_brush_size(size)

    def clear_paint(self):
        """Clear all paint marks."""
        self.parent.image_canvas.clear_paint()
        self.parent.statusBar().showMessage("Paint marks cleared")

    def apply_crop(self):
        """Apply the crop selection."""
        if not self.parent.image_canvas.has_crop_selection_ready():
            QMessageBox.information(self.parent, "No Changes", "The crop area matches the full image. No cropping needed.")
            return

        # Confirm crop operation
        reply = QMessageBox.question(
            self.parent, "Confirm Crop",
            "This will permanently crop the image. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            success = self.parent.image_canvas.perform_crop()
            if success:
                self.parent.statusBar().showMessage("Image cropped successfully")
                self.crop_btn.setChecked(False)
            else:
                QMessageBox.warning(self.parent, "Crop Failed", "Failed to crop the image.")

    def cancel_crop(self):
        """Cancel crop selection."""
        self.parent.image_canvas.clear_crop_selection()
        self.crop_btn.setChecked(False)
        self.parent.statusBar().showMessage("Crop cancelled")

    # Zoom-related methods
    def zoom_in(self):
        """Zoom in on the image."""
        if self.parent.image_canvas and self.parent.image_canvas.image_pixmap:
            self.parent.image_canvas.zoom_in()

    def zoom_out(self):
        """Zoom out on the image."""
        if self.parent.image_canvas and self.parent.image_canvas.image_pixmap:
            self.parent.image_canvas.zoom_out()

    def reset_zoom(self):
        """Reset zoom to 100%."""
        if self.parent.image_canvas and self.parent.image_canvas.image_pixmap:
            self.parent.image_canvas.reset_zoom()

    def fit_to_window(self):
        """Fit image to window."""
        if self.parent.image_canvas and self.parent.image_canvas.image_pixmap:
            self.parent.image_canvas.fit_to_window()

    def _update_zoom_controls(self):
        """Update the enabled state of zoom controls and labels."""
        if not self.parent.current_image_path:
            self._disable_all_zoom_controls()
            return

        # Enable zoom controls if image is loaded
        self._enable_all_zoom_controls()

        # Update zoom percentage
        zoom_percentage = self.parent.image_canvas.get_zoom_percentage()
        self.zoom_label.setText(f"{zoom_percentage}%")

    def _enable_all_zoom_controls(self):
        """Enable all zoom controls."""
        self.zoom_in_btn.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self.reset_zoom_btn.setEnabled(True)
        self.fit_window_btn.setEnabled(True)

    def _disable_all_zoom_controls(self):
        """Disable all zoom controls."""
        self.zoom_in_btn.setEnabled(False)
        self.zoom_out_btn.setEnabled(False)
        self.reset_zoom_btn.setEnabled(False)
        self.fit_window_btn.setEnabled(False)
        self.zoom_label.setText("—")
