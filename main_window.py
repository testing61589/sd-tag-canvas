"""
Main window for the Image Tag Editor application with zoom functionality.
"""

from pathlib import Path
import shutil
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QLabel, QCheckBox, QTextEdit, QPushButton, QFileDialog,
    QSplitter, QMessageBox, QListWidgetItem, QColorDialog,
    QSlider, QFrame, QApplication, QScrollArea, QListView, QInputDialog,
    QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QEvent, QSize
from PySide6.QtGui import QFont, QAction, QTextOption, QColor, QIcon, QPixmap, QTextCursor, QTextCharFormat

from image_canvas import ImageCanvas
from config import (
    IMAGE_EXTENSIONS, DEFAULT_PAINT_COLOR, LEFT_PANEL_MIN_WIDTH,
    LEFT_PANEL_DEFAULT_WIDTH, RIGHT_PANEL_DEFAULT_WIDTH,
    TAGS_EDITOR_MIN_HEIGHT, TAGS_EDITOR_MAX_HEIGHT,
    PAINT_TOOLS_MAX_HEIGHT, JPEG_QUALITY, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE,
    THUMBNAIL_SIZE, THUMBNAIL_MODE_ENABLED,
    TRIGGER_WORD_BUTTON_WIDTH, TRIGGER_WORD_BUTTON_HEIGHT, BASE_TRIGGER_COLOR
)


class ImageTagEditor(QMainWindow):
    def __init__(self, initial_folder=None):
        super().__init__()
        self.current_folder = None
        self.current_image_path = None
        self.image_extensions = IMAGE_EXTENSIONS
        self.thumbnail_mode = THUMBNAIL_MODE_ENABLED
        self.current_paint_color = QColor(*DEFAULT_PAINT_COLOR)
        self.original_tags_content = ""
        self.save_changes_btn = None  # Will be renamed

        # Trigger words
        self.trigger_words = []
        self.trigger_words_container = None
        self._highlighting = False  # Re-entrancy guard
        
        self.init_ui()
        self.setup_connections()
        
        # Install event filter to capture arrow keys globally
        self.installEventFilter(self)
        QApplication.instance().installEventFilter(self)
        
        # Open initial folder if provided
        if initial_folder:
            # Use QTimer to ensure UI is fully initialized before opening folder
            QTimer.singleShot(100, lambda: self.open_folder_path(initial_folder))
        
    def init_ui(self):
        self.setWindowTitle("Image Tag Editor with Paint, Crop and Zoom Tools")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Image list
        self.create_left_panel(splitter)
        
        # Right panel - Image display and tags
        self.create_right_panel(splitter)
        
        # Set splitter proportions (25% left, 75% right)
        splitter.setSizes([LEFT_PANEL_DEFAULT_WIDTH, RIGHT_PANEL_DEFAULT_WIDTH])
        
        # Status bar
        self.statusBar().showMessage("Ready - Open a folder to start")
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_folder_action = QAction('Open Folder', self)
        open_folder_action.setShortcut('Ctrl+O')
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        
        crop_action = QAction('Crop', self)
        crop_action.setShortcut('Ctrl+R')
        crop_action.triggered.connect(self.apply_crop)
        edit_menu.addAction(crop_action)

        delete_action = QAction('Delete Image', self)
        delete_action.setShortcut('Delete')
        delete_action.triggered.connect(self.delete_image)
        edit_menu.addAction(delete_action)

        duplicate_action = QAction('Duplicate Image', self)
        duplicate_action.setShortcut('Ctrl+D')
        duplicate_action.triggered.connect(self.duplicate_image)
        edit_menu.addAction(duplicate_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        zoom_in_action = QAction('Zoom In', self)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction('Zoom Out', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        reset_zoom_action = QAction('Reset Zoom', self)
        reset_zoom_action.setShortcut('Ctrl+0')
        reset_zoom_action.triggered.connect(self.reset_zoom)
        view_menu.addAction(reset_zoom_action)
        
        fit_window_action = QAction('Fit to Window', self)
        fit_window_action.setShortcut('Ctrl+F')
        fit_window_action.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_window_action)
        
    def toggle_thumbnail_mode(self, checked):
        """Toggle thumbnail mode and refresh image list"""
        self.thumbnail_mode = checked
        if self.current_folder:
            self.refresh_image_list()

    def create_left_panel(self, parent):
        # Left panel container
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Title
        title_label = QLabel("Image Files")
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        left_layout.addWidget(title_label)
        
        # Open folder button
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.open_folder)
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
        
        parent.addWidget(left_widget)
        
    def create_right_panel(self, parent):
        # Right panel container
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)
        
        # Image display area (QGraphicsView handles scrolling)
        self.filename_label = QLabel("No image selected")
        self.filename_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.filename_label.setStyleSheet("color: #ddd; border-bottom: 1px solid #555;")
        right_layout.addWidget(self.filename_label)

        self.image_canvas = ImageCanvas()
        right_layout.addWidget(self.image_canvas, 1)  # Stretch to fill available space
        
        # Paint tools section - moved between image and tags
        self.create_paint_tools(right_layout)

        # Trigger words section - above tags
        self.create_trigger_words_section(right_layout)

        # Tags section - moved to bottom
        self.create_tags_section(right_layout)
        
        parent.addWidget(right_widget)
        
        # Initialize canvas with proper brush size
        # Brush size set via slider, no init needed
    
    def create_paint_tools(self, layout):
        """Create paint tools UI"""
        # Paint tools frame
        tools_frame = QFrame()
        tools_frame.setFrameStyle(QFrame.Box)
        tools_frame.setMaximumHeight(PAINT_TOOLS_MAX_HEIGHT)
        tools_layout = QVBoxLayout(tools_frame)
        
        # First row of tools
        row1_layout = QHBoxLayout()
        
        # Tools label
        tools_label = QLabel("Tools:")
        tools_label.setFont(QFont("Arial", 9, QFont.Bold))
        row1_layout.addWidget(tools_label)
        
        # Color picker button
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(40, 30)
        self.color_btn.setStyleSheet(f"background-color: {self.current_paint_color.name()}; border: 2px solid #333;")
        self.color_btn.clicked.connect(self.choose_color)
        self.color_btn.setToolTip("Choose paint color")
        row1_layout.addWidget(self.color_btn)
        
        # Color dropper button
        self.dropper_btn = QPushButton("🎨")  # dropper icon
        self.dropper_btn.setFixedSize(40, 30)
        self.dropper_btn.setCheckable(True)
        self.dropper_btn.setToolTip("Color dropper - click to pick color from image")
        self.dropper_btn.toggled.connect(self.toggle_dropper)
        row1_layout.addWidget(self.dropper_btn)
        
        # Crop tool button
        self.crop_btn = QPushButton("✂️")  # scissors icon
        self.crop_btn.setFixedSize(40, 30)
        self.crop_btn.setCheckable(True)
        self.crop_btn.setToolTip("Crop tool - drag to select area to crop")
        self.crop_btn.toggled.connect(self.toggle_crop_mode)
        row1_layout.addWidget(self.crop_btn)
        
        # Brush size controls
        self.create_brush_size_controls(row1_layout)
        
        row1_layout.addStretch()  # Push everything to the left
        
        # Second row of tools
        row2_layout = QHBoxLayout()
        self.create_action_buttons(row2_layout)
        
        # Add zoom controls to second row
        self.create_zoom_controls(row2_layout)
        
        row2_layout.addStretch()  # Push everything to the left
        
        tools_layout.addLayout(row1_layout)
        tools_layout.addLayout(row2_layout)
        
        layout.addWidget(tools_frame)
    
    def create_brush_size_controls(self, layout):
        """Create brush size controls"""
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
        """Create action buttons for paint and crop operations"""
        # Clear paint button
        clear_paint_btn = QPushButton("Clear Paint")
        clear_paint_btn.clicked.connect(self.clear_paint)
        clear_paint_btn.setToolTip("Clear all paint marks")
        layout.addWidget(clear_paint_btn)
        
        # Apply crop button
        self.apply_crop_btn = QPushButton("Apply Crop")
        self.apply_crop_btn.clicked.connect(self.apply_crop)
        self.apply_crop_btn.setEnabled(False)
        self.apply_crop_btn.setToolTip("Apply the selected crop area (Ctrl+R)")
        layout.addWidget(self.apply_crop_btn)
        
        # Cancel crop button
        self.cancel_crop_btn = QPushButton("Cancel Crop")
        self.cancel_crop_btn.clicked.connect(self.cancel_crop)
        self.cancel_crop_btn.setEnabled(False)
        self.cancel_crop_btn.setToolTip("Cancel crop selection")
        layout.addWidget(self.cancel_crop_btn)

        # Delete button
        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.clicked.connect(self.delete_image)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setToolTip("Move image and tags to .deleted folder (Delete key)")
        layout.addWidget(self.delete_btn)

        # Duplicate button
        self.duplicate_btn = QPushButton("Duplicate")
        self.duplicate_btn.clicked.connect(self.duplicate_image)
        self.duplicate_btn.setEnabled(False)
        self.duplicate_btn.setToolTip("Duplicate image and tag file")
        layout.addWidget(self.duplicate_btn)
    
    def create_zoom_controls(self, layout):
        """Create zoom controls"""
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Zoom label
        zoom_label = QLabel("Zoom:")
        layout.addWidget(zoom_label)

        # Zoom out button
        self.zoom_out_btn = QPushButton("−")  # minus sign
        self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_out_btn.setToolTip("Zoom out (Ctrl+-)")
        self.zoom_out_btn.setEnabled(False)  # Initially disabled
        layout.addWidget(self.zoom_out_btn)

        # Zoom percentage display
        self.zoom_label = QLabel("—")  # Em dash for no image
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet("border: 1px solid #666; padding: 2px;")
        layout.addWidget(self.zoom_label)

        # Zoom in button
        self.zoom_in_btn = QPushButton("+")  # plus sign
        self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_in_btn.setToolTip("Zoom in (Ctrl++)")
        self.zoom_in_btn.setEnabled(False)  # Initially disabled
        layout.addWidget(self.zoom_in_btn)

        # Reset zoom button
        self.reset_zoom_btn = QPushButton("1:1")
        self.reset_zoom_btn.setFixedSize(35, 30)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        self.reset_zoom_btn.setToolTip("Reset zoom to 100% (Ctrl+0)")
        self.reset_zoom_btn.setEnabled(False)  # Initially disabled
        layout.addWidget(self.reset_zoom_btn)

        # Fit to window button
        self.fit_window_btn = QPushButton("Fit")
        self.fit_window_btn.setFixedSize(35, 30)
        self.fit_window_btn.clicked.connect(self.fit_to_window)
        self.fit_window_btn.setToolTip("Fit image to window (Ctrl+F)")
        self.fit_window_btn.setEnabled(False)  # Initially disabled
        layout.addWidget(self.fit_window_btn)

        # Update zoom controls based on current state
        self._update_zoom_controls()

    def _update_zoom_controls(self):
        """Update the enabled state of zoom controls and labels"""
        if not self.current_image_path:
            self.zoom_in_btn.setEnabled(False)
            self.zoom_out_btn.setEnabled(False)
            self.reset_zoom_btn.setEnabled(False)
            self.fit_window_btn.setEnabled(False)
            self.zoom_label.setText("—")
            return

        # Enable zoom controls if image is loaded
        self.zoom_in_btn.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self.reset_zoom_btn.setEnabled(True)
        self.fit_window_btn.setEnabled(True)

        # Update zoom percentage
        zoom_percentage = self.image_canvas.get_zoom_percentage()
        self.zoom_label.setText(f"{zoom_percentage}%")
    
    def create_trigger_words_section(self, layout):
        """Create trigger words section above tags"""
        try:
            # Trigger words container with horizontal flow
            self.trigger_words_container = QWidget()
            trigger_layout = QHBoxLayout(self.trigger_words_container)
            trigger_layout.setContentsMargins(0, 0, 0, 0)
            trigger_layout.setSpacing(8)
            trigger_layout.setAlignment(Qt.AlignLeft)

            # Add to main layout
            layout.addWidget(self.trigger_words_container)
            layout.addSpacing(4)

            # Load and create buttons
            if self.current_folder:
                self.load_trigger_words()
                self.refresh_trigger_words_ui()
        except Exception as e:
            self.statusBar().showMessage(f"Error creating trigger words section: {e}")
            import traceback
            traceback.print_exc()

    def create_tags_section(self, layout):
        """Create the tags editing section"""
        tags_label = QLabel("Tags:")
        tags_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(tags_label)

        # Tags text edit
        self.tags_edit = QTextEdit()
        self.tags_edit.setMinimumHeight(TAGS_EDITOR_MIN_HEIGHT)
        self.tags_edit.setMaximumHeight(TAGS_EDITOR_MAX_HEIGHT)
        self.tags_edit.setPlaceholderText("Enter tags here (one per line or comma-separated)...")
        self.tags_edit.setEnabled(False)
        self.tags_edit.setWordWrapMode(QTextOption.WordWrap)
        layout.addWidget(self.tags_edit)

        # Save tags button
        self.save_changes_btn = QPushButton("Save Changes")
        self.save_changes_btn.setEnabled(False)
        self.save_changes_btn.clicked.connect(self.save_changes)
        layout.addWidget(self.save_changes_btn)
    
    def choose_color(self):
        """Open color chooser dialog"""
        color = QColorDialog.getColor(self.current_paint_color, self, "Choose Paint Color")
        if color.isValid():
            self.current_paint_color = color
            self.update_color_display(color)
            self.image_canvas.set_paint_color(color)
    
    def update_color_display(self, color):
        """Update the color button display"""
        self.current_paint_color = color
        self.color_btn.setStyleSheet(f"background-color: {color.name()}; border: 2px solid #333;")
        self.image_canvas.set_paint_color(color)
    
    def toggle_dropper(self, checked):
        """Toggle color dropper mode"""
        if checked:
            # Disable crop mode if it's active
            self.crop_btn.setChecked(False)
            
        self.image_canvas.set_dropper_mode(checked)
        if checked:
            self.dropper_btn.setText("🎯")  # Target icon when active
            self.statusBar().showMessage("Dropper active - click on image to pick color")
        else:
            self.dropper_btn.setText("🎨")  # Palette icon when inactive
            self.statusBar().showMessage("Paint mode active")
    
    def toggle_crop_mode(self, checked):
        """Toggle crop mode"""
        if checked:
            # Disable dropper mode if it's active
            self.dropper_btn.setChecked(False)
            
        self.image_canvas.set_crop_mode(checked)
        self.apply_crop_btn.setEnabled(checked)
        self.cancel_crop_btn.setEnabled(checked)
        
        if checked:
            self.statusBar().showMessage("Crop mode active - drag handles to adjust crop area")
        else:
            self.statusBar().showMessage("Paint mode active")
    
    def update_crop_buttons(self):
        """Update crop button states based on selection"""
        if self.image_canvas.crop_active:
            has_selection = self.image_canvas.has_crop_selection_ready()
            # Apply button is enabled when crop mode is active and selection differs from full image
            self.apply_crop_btn.setEnabled(has_selection)
    
    def apply_crop(self):
        """Apply the crop selection"""
        if not self.image_canvas.has_crop_selection_ready():
            QMessageBox.information(self, "No Changes", "The crop area matches the full image. No cropping needed.")
            return
        
        # Confirm crop operation
        reply = QMessageBox.question(
            self, "Confirm Crop", 
            "This will permanently crop the image. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.image_canvas.perform_crop()
            if success:
                self.statusBar().showMessage("Image cropped successfully")
                # Disable crop mode after successful crop
                self.crop_btn.setChecked(False)
                self.toggle_crop_mode(False)
            else:
                QMessageBox.warning(self, "Crop Failed", "Failed to crop the image.")
    
    def cancel_crop(self):
        """Cancel crop selection"""
        self.image_canvas.clear_crop_selection()
        self.crop_btn.setChecked(False)
        self.toggle_crop_mode(False)
        self.statusBar().showMessage("Crop cancelled")
    
    def update_brush_size(self, size):
        """Update brush size"""
        self.brush_size_label.setText(str(size))
        self.image_canvas.set_brush_size(size)
    
    def clear_paint(self):
        """Clear all paint marks"""
        self.image_canvas.clear_paint()
        self.statusBar().showMessage("Paint marks cleared")
    
    # Zoom-related methods
    def zoom_in(self):
        """Zoom in on the image"""
        if self.image_canvas and self.image_canvas.image_pixmap:
            self.image_canvas.zoom_in()
    
    def zoom_out(self):
        """Zoom out on the image"""
        if self.image_canvas and self.image_canvas.image_pixmap:
            self.image_canvas.zoom_out()
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        if self.image_canvas and self.image_canvas.image_pixmap:
            self.image_canvas.reset_zoom()
    
    def fit_to_window(self):
        """Fit image to window"""
        if self.image_canvas and self.image_canvas.image_pixmap:
            self.image_canvas.fit_to_window()
    
    def update_zoom_display(self, zoom_factor):
        """Update zoom percentage display"""
        try:
            percentage = int(zoom_factor * 100)
            self.zoom_label.setText(f"{percentage}%")
            
            # Update status message
            if hasattr(self, 'current_image_path') and self.current_image_path:
                current_row = self.image_list.currentRow()
                total_count = self.image_list.count()
                self.statusBar().showMessage(
                    f"Image {current_row + 1} of {total_count}: {self.current_image_path.name} - Zoom: {percentage}%"
                )
        except Exception as e:
            print(f"Error updating zoom display: {e}")
        
    def setup_connections(self):
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.tags_edit.textChanged.connect(self.on_tags_changed)
        self.tags_edit.textChanged.connect(self.auto_resize_text_edit)
        self.tags_edit.textChanged.connect(self.update_unsaved_status)

        # Connect zoom signal
        self.image_canvas.zoom_changed.connect(self.update_zoom_display)

        # Connect image needs save signal if available
        if hasattr(self.image_canvas, 'image_needs_save_changed'):
            self.image_canvas.image_needs_save_changed.connect(self.update_unsaved_status)
        
    def open_folder(self):
        """Open folder dialog and load selected folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", ""
        )
        
        if folder_path:
            self.open_folder_path(Path(folder_path))
    
    def open_folder_path(self, folder_path):
        """Open a specific folder path (used for both dialog and command line)"""
        try:
            self.current_folder = Path(folder_path)

            # Load trigger words first so they're available for tag highlighting
            self.load_trigger_words()
            self.refresh_trigger_words_ui()

            self.load_images()
            self.statusBar().showMessage(f"Loaded folder: {folder_path}")
        except Exception as e:
            QMessageBox.warning(
                self, 
                "Error Loading Folder", 
                f"Failed to load folder: {folder_path}\n\nError: {str(e)}"
            )
            self.statusBar().showMessage("Failed to load folder")
            
    def load_images(self):
        self.image_list.clear()

        if not self.current_folder:
            return
        
        if self.thumbnail_mode:
            # Icon (thumbnail) mode settings
            self.image_list.setWrapping(True)
            self.image_list.setWordWrap(True)
            self.image_list.setTextElideMode(Qt.ElideRight)
            self.image_list.setViewMode(QListWidget.IconMode)
            self.image_list.setResizeMode(QListView.Adjust)
            self.image_list.setFlow(QListView.LeftToRight)
            self.image_list.setSpacing(4)
            self.image_list.setUniformItemSizes(True)
            self.image_list.setGridSize(QSize(70, 120))
            self.image_list.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        else:
            # List mode settings
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

        # Find all image files in the folder
        image_files = []
        for file_path in self.current_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                image_files.append(file_path)

        # Sort files by name
        image_files.sort(key=lambda x: x.name.lower())

        # Add to list widget
        for image_path in image_files:
            item = QListWidgetItem(image_path.name)
            item.setData(Qt.UserRole, str(image_path))
            if self.thumbnail_mode:
                try:
                    pixmap = QPixmap(str(image_path)).scaled(THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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

        # Select the first image if any images were found
        if image_files:
            self.image_list.setCurrentRow(0)
            first_item = self.image_list.item(0)
            if first_item:
                self.on_image_selected(first_item)
        else:
            self.filename_label.setText("No image files found")
            self.statusBar().showMessage("No image files found in the selected folder")

        if image_files:
            QTimer.singleShot(3000, lambda: self.statusBar().showMessage(
                f"Found {len(image_files)} image files - Use ↑/↓ keys to navigate, Ctrl+wheel to zoom"
            ))
        
    def refresh_image_list(self):
        """Refresh the image list and try to preserve current selection if image still exists."""
        current_path = self.current_image_path
        self.load_images()

        if current_path and current_path.exists():
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if Path(item.data(Qt.UserRole)) == current_path:
                    self.image_list.setCurrentRow(i)
                    self.on_image_selected(item)
                    return

        # Select first image if available
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
            self.on_image_selected(self.image_list.item(0))
        else:
            # No images left
            self.image_canvas._clear_image_data()
            self.filename_label.setText("No image selected")
            self.image_canvas.update()
            self.tags_edit.clear()
            self._disable_zoom_controls()
            self.tags_edit.setEnabled(False)
            self.save_changes_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.duplicate_btn.setEnabled(False)
            self.current_image_path = None
            self.statusBar().showMessage("No more images in folder")

    def on_image_selected(self, item):
        image_path = Path(item.data(Qt.UserRole))
        self.current_image_path = image_path
        self.filename_label.setText(image_path.name)

        # Unload old tags first
        self._unload_tags()

        # Display image in canvas
        success = self.image_canvas.set_image(image_path)
        self.image_canvas.fit_to_window()  # Ensure fit after layout
        if not success:
            self.filename_label.setText(f"Failed to load: {image_path.name}")
            self.statusBar().showMessage(f"Failed to load image: {image_path.name}")
            self._disable_zoom_controls()
            return

        # Enable zoom controls
        self._enable_zoom_controls()

        # Reset crop mode when switching images
        if self.crop_btn.isChecked():
            self.crop_btn.setChecked(False)
            self.toggle_crop_mode(False)

        # Load and display tags
        self.load_tags(image_path)

        # Enable tags editing
        self.tags_edit.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.duplicate_btn.setEnabled(True)

        # Update status bar with position and zoom info
        current_row = self.image_list.currentRow()
        total_count = self.image_list.count()
        zoom_percentage = self.image_canvas.get_zoom_percentage()
        self.statusBar().showMessage(
            f"Image {current_row + 1} of {total_count}: {image_path.name} - Zoom: {zoom_percentage}%"
        )
    
    def _enable_zoom_controls(self):
        """Enable zoom controls when image is loaded"""
        self.zoom_in_btn.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self.reset_zoom_btn.setEnabled(True)
        self.fit_window_btn.setEnabled(True)
    
    def _disable_zoom_controls(self):
        """Disable zoom controls when no image is loaded"""
        self.zoom_in_btn.setEnabled(False)
        self.zoom_out_btn.setEnabled(False)
        self.reset_zoom_btn.setEnabled(False)
        self.fit_window_btn.setEnabled(False)
        self.zoom_label.setText("—")
            
    def _unload_tags(self):
        """Unload current tags before loading new ones."""
        self.tags_edit.blockSignals(True)
        plain_format = QTextCharFormat()
        plain_format.setBackground(QColor())  # Default/transparent
        cursor = QTextCursor(self.tags_edit.document())
        cursor.select(QTextCursor.Document)
        cursor.mergeCharFormat(plain_format)
        self.tags_edit.clear()
        self.tags_edit.blockSignals(False)
        self.original_tags_content = ""

    def auto_resize_text_edit(self):
        """Auto-resize the text edit based on content"""
        doc_height = self.tags_edit.document().size().height()
        padding = 20
        new_height = int(doc_height + padding)
        new_height = max(TAGS_EDITOR_MIN_HEIGHT, min(TAGS_EDITOR_MAX_HEIGHT, new_height))
        self.tags_edit.setFixedHeight(new_height)

    def load_tags(self, image_path):
        tags_file = image_path.with_suffix('.txt')

        try:
            # Block signals while loading tags to avoid recursive highlighting
            self.tags_edit.blockSignals(True)
            if tags_file.exists():
                with open(tags_file, 'r', encoding='utf-8') as f:
                    tags_content = f.read().strip()
                    self.tags_edit.setPlainText(tags_content)
            else:
                self.tags_edit.setPlainText("")

            self.original_tags_content = self.tags_edit.toPlainText().strip()
            self.auto_resize_text_edit()
            self.update_unsaved_status()
            self.highlight_trigger_words()
            self.tags_edit.blockSignals(False)
        except Exception as e:
            QMessageBox.warning(
                self, "Error",
                f"Failed to load tags file: {str(e)}"
            )
            self.tags_edit.setPlainText("")
            self.original_tags_content = ""
            self.update_unsaved_status()
            
    def save_changes(self):
        """Save changes to tags and image."""
        if not self.current_image_path:
            return

        # Save tags
        tags_content = self.tags_edit.toPlainText().strip()
        tags_file = self.current_image_path.with_suffix('.txt')
        tags_msg = self._save_tags_file(tags_content, tags_file)
        self.original_tags_content = tags_content

        # Save image if needed
        image_msg = None
        if hasattr(self.image_canvas, 'needs_image_save') and self.image_canvas.needs_image_save():
            pixmap = self.image_canvas.get_combined_image() if self.image_canvas.has_painted else self.image_canvas.image_pixmap
            if self._save_image_pixmap(pixmap):
                self.image_canvas.set_image(self.current_image_path)
                image_msg = "Image saved"
            else:
                image_msg = "Failed to save image"

        # Status message
        if image_msg:
            self.statusBar().showMessage(f"{tags_msg}, {image_msg}")
        else:
            self.statusBar().showMessage(tags_msg)

        self.update_unsaved_status()

    def has_unsaved_changes(self):
        """Check if there are unsaved changes to tags or image."""
        if not self.current_image_path:
            return False
        current_tags = self.tags_edit.toPlainText().strip()
        image_needs_save = getattr(self.image_canvas, 'needs_image_save', lambda: False)()
        return current_tags != self.original_tags_content or image_needs_save

    def update_unsaved_status(self):
        """Update save button enabled state based on unsaved changes."""
        self.save_changes_btn.setEnabled(self.has_unsaved_changes())

    def navigate_to_image(self, row):
        """Navigate to the image at the given row index."""
        self.image_list.setCurrentRow(row)
        item = self.image_list.item(row)
        if item:
            self.on_image_selected(item)

    def prompt_save_before_navigate(self, key):
        """Prompt user to save changes before navigating."""
        msg_box = QMessageBox(self)
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
            self.save_changes()
            return True
        elif clicked == discard_btn:
            self.discard_changes()
            return True
        else:
            return False

    def discard_changes(self):
        """Discard unsaved changes by reloading original state."""
        if self.current_image_path:
            self.load_tags(self.current_image_path)
            if hasattr(self.image_canvas, 'set_image'):
                self.image_canvas.set_image(self.current_image_path)
        self.update_unsaved_status()
    
    def _save_tags_file(self, tags_content, tags_file):
        """Save tags to file and return status message"""
        if tags_content:
            with open(tags_file, 'w', encoding='utf-8') as f:
                f.write(tags_content)
            return f"Tags saved to {tags_file.name}"
        else:
            if tags_file.exists():
                tags_file.unlink()
                return f"Tags file {tags_file.name} removed"
            else:
                return "No tags to save"
    

    # Trigger words methods
    def load_trigger_words(self):
        """Load trigger words from _triggerwords.txt in current folder"""
        if not self.current_folder:
            self.trigger_words = []
            return

        triggers_path = self.current_folder / "_triggerwords.txt"

        if not triggers_path.exists():
            self.trigger_words = []
            return

        try:
            with open(triggers_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse comma and newline separated values
            words = []
            for line in content.splitlines():
                for word in line.split(','):
                    word = word.strip()
                    if word:
                        words.append(word)

            self.trigger_words = words
        except Exception as e:
            self.statusBar().showMessage(f"Error loading trigger words: {e}")
            print(f"Error loading trigger words: {e}")
            import traceback
            traceback.print_exc()
            self.trigger_words = []

    def refresh_trigger_words_ui(self):
        """Refresh trigger word buttons"""
        if not self.trigger_words_container:
            return

        layout = self.trigger_words_container.layout()
        if not layout:
            return

        # Clear existing widgets
        button_height = TRIGGER_WORD_BUTTON_HEIGHT
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Build trigger word colors map (lowercase key for case-insensitive matching)
        self.trigger_word_colors = {}

        for i, word in enumerate(self.trigger_words):
            btn = QPushButton(word)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, w=word: self.on_trigger_word_clicked(w))

            # Base colors
            base_r, base_g, base_b = BASE_TRIGGER_COLOR
            button_height = TRIGGER_WORD_BUTTON_HEIGHT

            # Progressively darker shade: each word gets 10% darker
            shade_factor = max(0.40, 1.0 - (i * 0.10))  # Down to 40% brightness
            r = int(base_r * shade_factor)
            g = int(base_g * shade_factor)
            b = int(base_b * shade_factor)

            # Store color for this trigger word
            self.trigger_word_colors[word.strip().lower()] = QColor(r, g, b)

            # Disable automatic sizing - let stylesheet control
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

            # Style with minimal padding for tight fit
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({r}, {g}, {b});
                    border: 1px solid #333;
                    border-radius: {button_height // 2}px;
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 2px 4px;
                }}
                QPushButton:hover {{
                    background-color: rgb({min(255, r+30)}, {min(255, g+30)}, {min(255, b+30)});
                }}
            """)
            layout.addWidget(btn)

        # Add "Add" button
        add_btn = QPushButton("+ Add Trigger Word")
        add_btn.clicked.connect(self.add_trigger_word)
        add_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #444;
                border: 1px solid #333;
                border-radius: {button_height // 2}px;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)
        layout.addWidget(add_btn)

    def on_trigger_word_clicked(self, trigger_word):
        """Add trigger word to front of comma-separated tags"""
        try:
            current_tags = self.tags_edit.toPlainText().strip()
            if current_tags:
                # Add trigger word with comma separator at the front
                new_tags = f"{trigger_word}, {current_tags}"
            else:
                new_tags = trigger_word

            # Insert at beginning and keep cursor at start for easy editing
            self.tags_edit.setPlainText(new_tags)
            self.tags_edit.moveCursor(QTextCursor.Start)
            self.update_unsaved_status()
        except Exception as e:
            import traceback
            print(f"Error adding trigger word '{trigger_word}': {e}")
            traceback.print_exc()

    def add_trigger_word(self):
        """Prompt user to add a new trigger word"""
        word, ok = QInputDialog.getText(
            self,
            "Add Trigger Word",
            "Enter new trigger word:"
        )

        if ok and word.strip():
            self.trigger_words.append(word.strip())

            # Save to _triggerwords.txt
            triggers_path = self.current_folder / "_triggerwords.txt"
            try:
                with open(triggers_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.trigger_words))

                # Refresh UI
                self.refresh_trigger_words_ui()
                self.statusBar().showMessage(f"Added trigger word: {word.strip()}")
            except Exception as e:
                self.statusBar().showMessage(f"Failed to save trigger words: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.warning(self, "Error", f"Failed to save trigger words: {e}")

    #

    def _save_image_if_modified(self):
        """Save image if it has been modified and return status message"""
        # Save edited image if there are paint marks
        if self.image_canvas.has_paint_marks():
            combined_image = self.image_canvas.get_combined_image()
            if combined_image and self._save_image_pixmap(combined_image):
                # Clear the paint overlay since it's now part of the base image
                self.image_canvas.has_painted = False
                self.image_canvas.paint_overlay.fill(Qt.transparent)
                self.image_canvas.update()
                # Reload the image to show the saved version
                self.image_canvas.set_image(self.current_image_path)
                return "Image with modifications saved"
            else:
                return "Failed to save edited image"
        else:
            # Check if we need to save a cropped image (cropping modifies the base pixmap)
            current_pixmap = self.image_canvas.image_pixmap
            if current_pixmap and self._save_image_pixmap(current_pixmap):
                return "Image saved"
            else:
                return "Failed to save image" if current_pixmap else None
    
    def _save_image_pixmap(self, pixmap):
        """Save a pixmap to the current image path"""
        if not self.current_image_path or not pixmap:
            return False
        
        file_ext = self.current_image_path.suffix.lower()
        image_format = None
        if file_ext in ['.jpg', '.jpeg']:
            image_format = 'JPEG'
        elif file_ext == '.png':
            image_format = 'PNG'
        elif file_ext == '.bmp':
            image_format = 'BMP'
        
        if image_format:
            return pixmap.save(str(self.current_image_path), image_format, JPEG_QUALITY)
        else:
            return pixmap.save(str(self.current_image_path))
            
    def duplicate_image(self):
        """Duplicate the current image and its associated tag file"""
        if not self.current_image_path or not self.current_image_path.exists():
            QMessageBox.warning(self, "No Image", "No image selected or image file not found.")
            return

        try:
            # Create duplicate image filename (e.g., "image.jpg" -> "image_copy.jpg")
            image_stem = self.current_image_path.stem
            image_suffix = self.current_image_path.suffix
            duplicate_image_path = self.current_image_path.parent / f"{image_stem}_copy{image_suffix}"

            # Handle case where duplicate already exists
            counter = 1
            while duplicate_image_path.exists():
                duplicate_image_path = self.current_image_path.parent / f"{image_stem}_copy{counter}{image_suffix}"
                counter += 1

            # Copy image file
            shutil.copy(str(self.current_image_path), str(duplicate_image_path))

            # Copy tag file if it exists
            tag_file = self.current_image_path.with_suffix('.txt')
            duplicate_tag_path = duplicate_image_path.with_suffix('.txt')
            if tag_file.exists():
                shutil.copy(str(tag_file), str(duplicate_tag_path))

            # Update UI to show the new image
            self.statusBar().showMessage(f"Duplicated '{self.current_image_path.name}' -> '{duplicate_image_path.name}'")

            # Reload the images list and select the duplicate
            self.refresh_image_list()

        except Exception as e:
            QMessageBox.critical(
                self, "Duplicate Failed",
                f"Failed to duplicate image: {str(e)}"
            )

    def delete_image(self):
        """Move current image and associated tag file to .deleted subfolder"""
        if not self.current_image_path or not self.current_image_path.exists():
            QMessageBox.warning(self, "No Image", "No image selected or image file not found.")
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Move '{self.current_image_path.name}' and its associated tag file to .deleted folder?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Create .deleted subfolder if it doesn't exist
                deleted_folder = self.current_image_path.parent / ".deleted"
                deleted_folder.mkdir(exist_ok=True)

                # Move image file
                deleted_image_path = deleted_folder / self.current_image_path.name
                shutil.move(str(self.current_image_path), str(deleted_image_path))

                # Move tag file if it exists
                tag_file = self.current_image_path.with_suffix('.txt')
                if tag_file.exists():
                    deleted_tag_path = deleted_folder / tag_file.name
                    shutil.move(str(tag_file), str(deleted_tag_path))

                # Update UI
                self.refresh_image_list()
                self.statusBar().showMessage(f"Moved '{self.current_image_path.name}' to .deleted folder")

            except Exception as e:
                QMessageBox.critical(
                    self, "Delete Failed",
                    f"Failed to delete image: {str(e)}"
                )

    def _handle_image_deleted(self):
        """Handle UI updates after image deletion"""
        current_row = self.image_list.currentRow()

        # Remove item from list
        self.image_list.takeItem(current_row)

        # Select next image if available, otherwise previous
        new_count = self.image_list.count()
        if new_count > 0:
            if current_row < new_count:
                # Next image available at same index
                new_row = current_row
            else:
                # No next image, go to previous (last item)
                new_row = new_count - 1

            self.image_list.setCurrentRow(new_row)
            new_item = self.image_list.item(new_row)
            if new_item:
                self.on_image_selected(new_item)
        else:
            # No images left
            self.image_canvas._clear_image_data()
            self.filename_label.setText("No image selected")
            self.image_canvas.update()
            self.tags_edit.clear()
            self._disable_zoom_controls()
            self.tags_edit.setEnabled(False)
            self.save_changes_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.duplicate_btn.setEnabled(False)
            self.current_image_path = None
            self.statusBar().showMessage("No more images in folder")

    def on_tags_changed(self):
        self.update_unsaved_status()
        self.highlight_trigger_words()

    def _clear_all_formats(self):
        """Clear all formatting and reset to widget's background color."""
        cursor = QTextCursor(self.tags_edit.document())
        plain_format = QTextCharFormat()
        plain_format.setBackground(self.tags_edit.palette().color(self.tags_edit.backgroundRole()))
        cursor.select(QTextCursor.Document)
        cursor.mergeCharFormat(plain_format)

    def highlight_trigger_words(self):
        """Highlight trigger words in the tags field by applying per-tag colors."""
        # Re-entrancy guard
        if self._highlighting:
            return
        self._highlighting = True

        try:
            # Block textChanged signal to prevent infinite loop
            self.tags_edit.blockSignals(True)

            if not self.trigger_words:
                self._clear_all_formats()
                return

            text = self.tags_edit.document().toPlainText()
            if not text.strip():
                self._clear_all_formats()
                return

            # Clear existing formats first
            cursor = QTextCursor(self.tags_edit.document())
            plain_format = QTextCharFormat()
            plain_format.setBackground(QColor())  # Default/transparent  # Reset to white background
            cursor.select(QTextCursor.Document)
            cursor.mergeCharFormat(plain_format)

            # Apply highlight colors based on trigger_word_colors map
            # Process tags in text order - only split on commas, preserving spaces within tags
            cursor.setPosition(0)
            i = 0
            while i < len(text):
                # Skip leading commas and whitespace
                while i < len(text) and text[i] in ', \t\n\r':
                    i += 1

                if i >= len(text):
                    break

                tag_start = i
                # Only stop at commas; preserve spaces within tags (e.g., "red dress")
                while i < len(text) and text[i] != ',':
                    i += 1

                tag_end = i
                tag = text[tag_start:tag_end]
                tag_lower = tag.strip().lower()

                # Check if this tag matches a trigger word
                if tag_lower in self.trigger_word_colors:
                    color = self.trigger_word_colors[tag_lower]
                    cursor.setPosition(tag_start)
                    cursor.setPosition(tag_end, QTextCursor.KeepAnchor)
                    highlight_format = QTextCharFormat()
                    highlight_format.setBackground(color)
                    cursor.setCharFormat(highlight_format)

        except Exception as e:
            self.statusBar().showMessage(f"Error highlighting trigger words: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._highlighting = False
            self.tags_edit.blockSignals(False)

    def eventFilter(self, obj, event):
        """Global event filter to capture arrow keys for navigation and delete key"""
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                if self.current_image_path and self.delete_btn.isEnabled():
                    self.delete_image()
                return True
            elif event.key() == Qt.Key_Up or event.key() == Qt.Key_Down:
                if self.image_list.count() > 0:
                    if self.has_unsaved_changes():
                        if not self.prompt_save_before_navigate(event.key()):
                            return True  # Cancel navigation
                    
                    current_row = self.image_list.currentRow()
                    
                    if current_row == -1:
                        new_row = 0
                    elif event.key() == Qt.Key_Up:
                        new_row = max(0, current_row - 1)
                    else:  # Qt.Key_Down
                        new_row = min(self.image_list.count() - 1, current_row + 1)
                    
                    if new_row != current_row:
                        self.navigate_to_image(new_row)
                
                return True
        
        return super().eventFilter(obj, event)