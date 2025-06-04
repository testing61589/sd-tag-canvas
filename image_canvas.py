"""
Custom image canvas widget for displaying, painting, and cropping images with zoom support.
"""

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QSize, QPoint, QRect, Signal, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QCursor

from config import (CROP_HANDLE_SIZE, MIN_CROP_SIZE, DEFAULT_ZOOM_FACTOR, 
                   MIN_ZOOM_FACTOR, MAX_ZOOM_FACTOR, MOUSE_WHEEL_ZOOM_STEP)


class ImageCanvas(QWidget):
    """Custom widget for displaying and painting on images with zoom support"""
    
    # Signal emitted when zoom level changes
    zoom_changed = Signal(float)
    
    def __init__(self):
        super().__init__()
        self.image_pixmap = None
        self.paint_pixmap = None  # Overlay for paint marks
        self.scaled_pixmap = None
        self.paint_overlay = None
        
        # Paint settings
        self.paint_color = QColor(255, 0, 0)  # Default red
        self.brush_size = 15
        self.is_painting = False
        self.is_dropper_active = False
        self.last_point = QPoint()
        self.has_painted = False  # Track if any painting has been done
        
        # Crop settings
        self.is_crop_mode = False
        self.is_dragging_handle = False
        self.active_handle = None  # Which handle is being dragged
        self.crop_rect = QRect()
        self.has_crop_selection = False
        self.handle_size = CROP_HANDLE_SIZE
        self.handle_hover = None  # Which handle is being hovered
        
        # Image positioning and zoom
        self.image_rect = None
        self.fit_scale_factor = 1.0  # Scale to fit widget
        self.zoom_factor = DEFAULT_ZOOM_FACTOR  # User zoom level
        self.scale_factor = 1.0  # Combined scale factor
        
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QWidget {
                border: 2px dashed #555;
                background-color: #2b2b2b;
            }
        """)
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
    def set_image(self, image_path):
        """Load and display an image"""
        try:
            self.image_pixmap = QPixmap(str(image_path))
            if not self.image_pixmap.isNull():
                # Reset zoom when loading new image
                self.zoom_factor = DEFAULT_ZOOM_FACTOR
                self.update_display()
                # Create a new paint overlay for this image
                self.paint_overlay = QPixmap(self.image_pixmap.size())
                self.paint_overlay.fill(Qt.transparent)
                self.has_painted = False  # Reset paint flag for new image
                self.clear_crop_selection()  # Clear any crop selection
                self.zoom_changed.emit(self.zoom_factor)
                return True
            else:
                self._clear_image_data()
                return False
        except Exception as e:
            print(f"Error loading image: {e}")
            self._clear_image_data()
            return False
    
    def _clear_image_data(self):
        """Clear all image-related data"""
        self.image_pixmap = None
        self.paint_overlay = None
        self.has_painted = False
        self.zoom_factor = DEFAULT_ZOOM_FACTOR
        self.clear_crop_selection()
        self.update()
    
    def zoom_in(self):
        """Zoom in by one step"""
        if not self.image_pixmap:
            return
        
        # Use smaller increments for more controlled zooming
        if self.zoom_factor < 1.0:
            # Below 100%, use 0.1 increments
            new_zoom = min(1.0, self.zoom_factor + 0.1)
        else:
            # Above 100%, use 0.25 increments
            new_zoom = min(MAX_ZOOM_FACTOR, self.zoom_factor + 0.25)
        
        self.set_zoom(new_zoom)
    
    def zoom_out(self):
        """Zoom out by one step"""
        if not self.image_pixmap:
            return
        
        # Use smaller increments for more controlled zooming
        if self.zoom_factor <= 1.0:
            # At or below 100%, use 0.1 increments
            new_zoom = max(MIN_ZOOM_FACTOR, self.zoom_factor - 0.1)
        else:
            # Above 100%, use 0.25 increments
            new_zoom = max(1.0, self.zoom_factor - 0.25)
        
        self.set_zoom(new_zoom)
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        if not self.image_pixmap:
            return
        self.set_zoom(DEFAULT_ZOOM_FACTOR)
    
    def fit_to_window(self):
        """Set zoom to fit image in window"""
        if not self.image_pixmap:
            return
        
        # Get the scroll area size if we're in one
        parent_scroll = self.parent()
        if hasattr(parent_scroll, 'viewport'):
            viewport_size = parent_scroll.viewport().size()
            available_width = max(viewport_size.width() - 40, 100)
            available_height = max(viewport_size.height() - 40, 100)
        else:
            widget_size = self.size()
            available_width = max(widget_size.width() - 40, 100)
            available_height = max(widget_size.height() - 40, 100)
        
        # Calculate what zoom level would fit the image perfectly
        fit_zoom = min(
            available_width / self.image_pixmap.width(),
            available_height / self.image_pixmap.height()
        )
        
        # Set this as the zoom level
        self.set_zoom(fit_zoom)
    
    def set_zoom(self, zoom_factor):
        """Set zoom factor and update display"""
        if not self.image_pixmap:
            return
            
        # Clamp zoom factor to safe bounds
        self.zoom_factor = max(MIN_ZOOM_FACTOR, min(MAX_ZOOM_FACTOR, zoom_factor))
        
        # Only update display if zoom actually changed
        self.update_display()
        self.zoom_changed.emit(self.zoom_factor)
    
    def get_zoom_percentage(self):
        """Get current zoom as percentage"""
        return int(self.zoom_factor * 100)
    
    def update_display(self):
        """Update the scaled display of the image"""
        if not self.image_pixmap:
            return
            
        # Get the scroll area size if we're in one
        parent_scroll = self.parent()
        if hasattr(parent_scroll, 'viewport'):
            viewport_size = parent_scroll.viewport().size()
            available_width = max(viewport_size.width() - 40, 100)
            available_height = max(viewport_size.height() - 40, 100)
        else:
            widget_size = self.size()
            available_width = max(widget_size.width() - 40, 100)
            available_height = max(widget_size.height() - 40, 100)
        
        # Calculate what scale would fit the image in the available space
        self.fit_scale_factor = min(
            available_width / self.image_pixmap.width(),
            available_height / self.image_pixmap.height()
        )
        
        # The actual scale factor depends on zoom mode:
        # - zoom_factor = 1.0 means actual size (100%)
        # - zoom_factor < 1.0 means smaller than actual size
        # - zoom_factor > 1.0 means larger than actual size
        
        if self.zoom_factor <= 1.0:
            # For zoom <= 100%, use fit scale as maximum, zoom as multiplier
            self.scale_factor = min(self.fit_scale_factor, self.zoom_factor)
        else:
            # For zoom > 100%, use zoom directly (ignoring fit scale)
            self.scale_factor = self.zoom_factor
        
        # Calculate final image size
        final_width = int(self.image_pixmap.width() * self.scale_factor)
        final_height = int(self.image_pixmap.height() * self.scale_factor)
        
        # Set minimum size for scroll area to work properly
        min_width = max(final_width + 40, 400)
        min_height = max(final_height + 40, 300)
        self.setMinimumSize(min_width, min_height)
        
        # Calculate image position (centered)
        widget_size = self.size()
        x = max((widget_size.width() - final_width) // 2, 20)
        y = max((widget_size.height() - final_height) // 2, 20)
            
        self.image_rect = (x, y, final_width, final_height)
        
        self.update()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if not self.image_pixmap:
            super().wheelEvent(event)
            return
            
        # Check if Ctrl is pressed for zoom
        if event.modifiers() & Qt.ControlModifier:
            try:
                # Zoom in/out based on wheel direction
                angle_delta = event.angleDelta().y()
                
                # Use smaller increments for mouse wheel
                zoom_step = 0.05  # 5% increments for smooth mouse wheel zooming
                
                if angle_delta > 0:
                    # Zoom in
                    new_zoom = min(MAX_ZOOM_FACTOR, self.zoom_factor + zoom_step)
                else:
                    # Zoom out
                    new_zoom = max(MIN_ZOOM_FACTOR, self.zoom_factor - zoom_step)
                
                self.set_zoom(new_zoom)
                event.accept()
            except Exception as e:
                print(f"Zoom error: {e}")
                super().wheelEvent(event)
        else:
            # Pass to parent for scrolling
            super().wheelEvent(event)
    
    def paintEvent(self, event):
        """Custom paint event to draw image and paint overlay"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(43, 43, 43))  # Dark background
        
        if self.image_pixmap and self.image_rect:
            # Draw the base image
            scaled_image = self.image_pixmap.scaled(
                QSize(int(self.image_pixmap.width() * self.scale_factor),
                     int(self.image_pixmap.height() * self.scale_factor)),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            x, y, w, h = self.image_rect
            painter.drawPixmap(x, y, scaled_image)
            
            # Draw paint overlay if it exists
            if self.paint_overlay:
                scaled_overlay = self.paint_overlay.scaled(
                    scaled_image.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                painter.drawPixmap(x, y, scaled_overlay)
            
            # Draw crop selection rectangle
            if self.is_crop_mode and self.has_crop_selection:
                self.draw_crop_rectangle(painter)
        else:
            # No image loaded
            painter.setPen(QColor(170, 170, 170))  # Light gray text for dark background
            painter.drawText(self.rect(), Qt.AlignCenter, "No image selected")
    
    def draw_crop_rectangle(self, painter):
        """Draw the crop selection rectangle with edge handles"""
        if not self.has_crop_selection or not self.image_rect:
            return
        
        # Convert image coordinates to screen coordinates for display
        screen_rect = self.image_to_screen_rect(self.crop_rect)
        if not screen_rect.isValid():
            return
        
        # Draw crop rectangle
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.SolidLine))  # Yellow solid line
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(screen_rect)
        
        # Draw dimmed overlay outside crop area
        self._draw_crop_overlay(painter, screen_rect)
        
        # Draw handles
        self.draw_crop_handles(painter, screen_rect)
    
    def _draw_crop_overlay(self, painter, screen_rect):
        """Draw dimmed overlay outside crop area"""
        painter.setBrush(QColor(0, 0, 0, 100))  # Semi-transparent black
        x, y, w, h = self.image_rect
        image_screen_rect = QRect(x, y, w, h)
        
        # Top rectangle
        if screen_rect.top() > image_screen_rect.top():
            painter.drawRect(image_screen_rect.left(), image_screen_rect.top(), 
                           image_screen_rect.width(), screen_rect.top() - image_screen_rect.top())
        
        # Bottom rectangle  
        if screen_rect.bottom() < image_screen_rect.bottom():
            painter.drawRect(image_screen_rect.left(), screen_rect.bottom(),
                           image_screen_rect.width(), image_screen_rect.bottom() - screen_rect.bottom())
        
        # Left rectangle
        if screen_rect.left() > image_screen_rect.left():
            painter.drawRect(image_screen_rect.left(), screen_rect.top(),
                           screen_rect.left() - image_screen_rect.left(), screen_rect.height())
        
        # Right rectangle
        if screen_rect.right() < image_screen_rect.right():
            painter.drawRect(screen_rect.right(), screen_rect.top(),
                           image_screen_rect.right() - screen_rect.right(), screen_rect.height())
    
    def draw_crop_handles(self, painter, screen_rect):
        """Draw crop handles on edges and corners"""
        # Handle colors
        normal_color = QColor(255, 255, 0)  # Yellow
        hover_color = QColor(255, 255, 255)  # White for hover
        
        handles = self.get_handle_rects(screen_rect)
        
        for handle_name, handle_rect in handles.items():
            # Choose color based on hover state
            if self.handle_hover == handle_name:
                painter.setBrush(hover_color)
                painter.setPen(QPen(QColor(0, 0, 0), 2))  # Black border for contrast
            else:
                painter.setBrush(normal_color)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
            
            painter.drawRect(handle_rect)
    
    def get_handle_rects(self, screen_rect):
        """Get rectangles for all crop handles"""
        handle_size = self.handle_size
        handles = {}
        
        # Corner handles
        handles['top-left'] = QRect(screen_rect.left() - handle_size//2, 
                                   screen_rect.top() - handle_size//2, 
                                   handle_size, handle_size)
        
        handles['top-right'] = QRect(screen_rect.right() - handle_size//2, 
                                    screen_rect.top() - handle_size//2, 
                                    handle_size, handle_size)
        
        handles['bottom-left'] = QRect(screen_rect.left() - handle_size//2, 
                                      screen_rect.bottom() - handle_size//2, 
                                      handle_size, handle_size)
        
        handles['bottom-right'] = QRect(screen_rect.right() - handle_size//2, 
                                       screen_rect.bottom() - handle_size//2, 
                                       handle_size, handle_size)
        
        # Edge handles (center of each edge)
        handles['top'] = QRect(screen_rect.center().x() - handle_size//2, 
                              screen_rect.top() - handle_size//2, 
                              handle_size, handle_size)
        
        handles['bottom'] = QRect(screen_rect.center().x() - handle_size//2, 
                                 screen_rect.bottom() - handle_size//2, 
                                 handle_size, handle_size)
        
        handles['left'] = QRect(screen_rect.left() - handle_size//2, 
                               screen_rect.center().y() - handle_size//2, 
                               handle_size, handle_size)
        
        handles['right'] = QRect(screen_rect.right() - handle_size//2, 
                                screen_rect.center().y() - handle_size//2, 
                                handle_size, handle_size)
        
        return handles
    
    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        # Only update display if we have an image and the resize was significant
        if self.image_pixmap and event.size() != event.oldSize():
            # Use a small delay to prevent rapid resize events from causing issues
            if not hasattr(self, '_resize_timer'):
                self._resize_timer = QTimer()
                self._resize_timer.setSingleShot(True)
                self._resize_timer.timeout.connect(self.update_display)
            
            self._resize_timer.stop()
            self._resize_timer.start(50)  # 50ms delay
    
    def mousePressEvent(self, event):
        """Handle mouse press for painting, cropping, or color picking"""
        if not self.image_pixmap or not self.image_rect:
            return
            
        # Check if click is within image bounds
        click_pos = event.position().toPoint()
        x, y, w, h = self.image_rect
        
        if self.is_crop_mode and self.has_crop_selection:
            # Check if clicking on a crop handle
            handle = self.get_handle_under_point(click_pos)
            if handle:
                self.is_dragging_handle = True
                self.active_handle = handle
                return
        
        if (x <= click_pos.x() <= x + w and y <= click_pos.y() <= y + h):
            if self.is_crop_mode:
                # In crop mode but not clicking a handle - do nothing
                pass
            elif self.is_dropper_active:
                self.pick_color(click_pos)
            else:
                self.is_painting = True
                self.last_point = click_pos
                self.draw_point(click_pos)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for painting, cropping, or hover effects"""
        if not self.image_pixmap or not self.image_rect:
            return
            
        current_point = event.position().toPoint()
        x, y, w, h = self.image_rect
        
        # Handle crop handle dragging
        if self.is_crop_mode and self.is_dragging_handle and self.active_handle:
            self.update_crop_from_handle(current_point)
            return
        
        # Handle crop handle hover effects
        if self.is_crop_mode and self.has_crop_selection:
            old_hover = self.handle_hover
            self.handle_hover = self.get_handle_under_point(current_point)
            if old_hover != self.handle_hover:
                self.update_cursor_for_handle(self.handle_hover)
                self.update()
            return
        
        # Handle painting
        if (x <= current_point.x() <= x + w and y <= current_point.y() <= y + h):
            if self.is_painting:
                self.draw_line(self.last_point, current_point)
                self.last_point = current_point
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if self.is_dragging_handle:
            self.is_dragging_handle = False
            self.active_handle = None
        self.is_painting = False
    
    def get_handle_under_point(self, point):
        """Get which handle (if any) is under the given point"""
        if not self.has_crop_selection or not self.image_rect:
            return None
        
        screen_rect = self.image_to_screen_rect(self.crop_rect)
        if not screen_rect.isValid():
            return None
        
        handles = self.get_handle_rects(screen_rect)
        
        for handle_name, handle_rect in handles.items():
            if handle_rect.contains(point):
                return handle_name
        
        return None
    
    def update_cursor_for_handle(self, handle_name):
        """Update cursor based on which handle is being hovered"""
        if not handle_name:
            if self.is_crop_mode:
                self.setCursor(Qt.ArrowCursor)
            return
        
        # Set appropriate resize cursors for different handles
        cursor_map = {
            'top-left': Qt.SizeFDiagCursor,
            'top-right': Qt.SizeBDiagCursor,
            'bottom-left': Qt.SizeBDiagCursor,
            'bottom-right': Qt.SizeFDiagCursor,
            'top': Qt.SizeVerCursor,
            'bottom': Qt.SizeVerCursor,
            'left': Qt.SizeHorCursor,
            'right': Qt.SizeHorCursor,
        }
        
        cursor = cursor_map.get(handle_name, Qt.ArrowCursor)
        self.setCursor(cursor)
    
    def update_crop_from_handle(self, mouse_pos):
        """Update crop rectangle based on handle being dragged"""
        if not self.active_handle or not self.image_rect:
            return
        
        # Convert mouse position to image coordinates
        image_point = self.screen_to_image_coords(mouse_pos)
        if not image_point:
            return
        
        # Get current crop rectangle
        left = self.crop_rect.left()
        top = self.crop_rect.top()
        right = self.crop_rect.right()
        bottom = self.crop_rect.bottom()
        
        # Update rectangle based on which handle is being dragged
        if self.active_handle == 'top-left':
            left = min(image_point.x(), right - MIN_CROP_SIZE)
            top = min(image_point.y(), bottom - MIN_CROP_SIZE)
        elif self.active_handle == 'top-right':
            right = max(image_point.x(), left + MIN_CROP_SIZE)
            top = min(image_point.y(), bottom - MIN_CROP_SIZE)
        elif self.active_handle == 'bottom-left':
            left = min(image_point.x(), right - MIN_CROP_SIZE)
            bottom = max(image_point.y(), top + MIN_CROP_SIZE)
        elif self.active_handle == 'bottom-right':
            right = max(image_point.x(), left + MIN_CROP_SIZE)
            bottom = max(image_point.y(), top + MIN_CROP_SIZE)
        elif self.active_handle == 'top':
            top = min(image_point.y(), bottom - MIN_CROP_SIZE)
        elif self.active_handle == 'bottom':
            bottom = max(image_point.y(), top + MIN_CROP_SIZE)
        elif self.active_handle == 'left':
            left = min(image_point.x(), right - MIN_CROP_SIZE)
        elif self.active_handle == 'right':
            right = max(image_point.x(), left + MIN_CROP_SIZE)
        
        # Constrain to image bounds
        left = max(0, left)
        top = max(0, top)
        right = min(self.image_pixmap.width(), right)
        bottom = min(self.image_pixmap.height(), bottom)
        
        # Update crop rectangle
        self.crop_rect = QRect(left, top, right - left, bottom - top)
        self.update()
        
        # Notify parent window to update button states
        self._notify_parent_crop_update()
    
    def _notify_parent_crop_update(self):
        """Notify parent window about crop updates"""
        parent = self.parent()
        while parent and not hasattr(parent, 'update_crop_buttons'):
            parent = parent.parent()
        if parent and hasattr(parent, 'update_crop_buttons'):
            parent.update_crop_buttons()
    
    def clear_crop_selection(self):
        """Clear crop selection"""
        self.has_crop_selection = False
        self.is_dragging_handle = False
        self.active_handle = None
        self.handle_hover = None
        self.crop_rect = QRect()
        self.update()
    
    def initialize_crop_selection(self):
        """Initialize crop selection to cover the entire image"""
        if not self.image_pixmap:
            return
        
        # Set crop rectangle to cover the entire image
        self.crop_rect = QRect(0, 0, self.image_pixmap.width(), self.image_pixmap.height())
        self.has_crop_selection = True
        self.update()
    
    def perform_crop(self):
        """Perform the actual crop operation"""
        if not self.has_crop_selection or not self.image_pixmap:
            return False
        
        # Ensure crop rectangle is within image bounds
        image_bounds = QRect(0, 0, self.image_pixmap.width(), self.image_pixmap.height())
        crop_rect = self.crop_rect.intersected(image_bounds)
        
        if crop_rect.isEmpty():
            return False
        
        # Crop the original image
        cropped_image = self.image_pixmap.copy(crop_rect)
        
        # If there are paint marks, crop the overlay too and apply it
        if self.has_painted and self.paint_overlay:
            cropped_overlay = self.paint_overlay.copy(crop_rect)
            
            # Apply the cropped overlay to the cropped image
            painter = QPainter(cropped_image)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawPixmap(0, 0, cropped_overlay)
            painter.end()
        
        # Update the image pixmap with the cropped version
        self.image_pixmap = cropped_image
        
        # Create new paint overlay for the cropped image
        self.paint_overlay = QPixmap(cropped_image.size())
        self.paint_overlay.fill(Qt.transparent)
        self.has_painted = False
        
        # Clear crop selection and update display
        self.clear_crop_selection()
        self.update_display()
        
        return True
    
    def draw_point(self, point):
        """Draw a single point"""
        if not self.paint_overlay or self.is_crop_mode:
            return
            
        # Convert screen coordinates to image coordinates
        image_point = self.screen_to_image_coords(point)
        if not image_point:
            return
            
        painter = QPainter(self.paint_overlay)
        painter.setPen(QPen(self.paint_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))
        painter.drawPoint(image_point)
        painter.end()
        
        self.has_painted = True  # Mark that painting has occurred
        self.update()
    
    def draw_line(self, start_point, end_point):
        """Draw a line between two points"""
        if not self.paint_overlay or self.is_crop_mode:
            return
            
        # Convert screen coordinates to image coordinates
        image_start = self.screen_to_image_coords(start_point)
        image_end = self.screen_to_image_coords(end_point)
        
        if not image_start or not image_end:
            return
            
        painter = QPainter(self.paint_overlay)
        painter.setPen(QPen(self.paint_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(image_start, image_end)
        painter.end()
        
        self.has_painted = True  # Mark that painting has occurred
        self.update()
    
    def screen_to_image_coords(self, screen_point):
        """Convert screen coordinates to image coordinates"""
        if not self.image_rect or not self.image_pixmap or self.scale_factor <= 0:
            return None
            
        x, y, w, h = self.image_rect
        
        # Check if point is within the displayed image bounds
        if not (x <= screen_point.x() <= x + w and y <= screen_point.y() <= y + h):
            return None
        
        # Calculate relative position within the displayed image
        rel_x = screen_point.x() - x
        rel_y = screen_point.y() - y
        
        # Convert to original image coordinates with bounds checking
        try:
            orig_x = int(rel_x / self.scale_factor)
            orig_y = int(rel_y / self.scale_factor)
            
            # Ensure coordinates are within image bounds
            orig_x = max(0, min(orig_x, self.image_pixmap.width() - 1))
            orig_y = max(0, min(orig_y, self.image_pixmap.height() - 1))
            
            return QPoint(orig_x, orig_y)
        except (ZeroDivisionError, OverflowError):
            return None
    
    def image_to_screen_rect(self, image_rect):
        """Convert image rectangle to screen coordinates"""
        if not self.image_rect or not image_rect.isValid() or self.scale_factor <= 0:
            return QRect()
        
        x, y, w, h = self.image_rect
        
        try:
            # Convert image coordinates to screen coordinates
            screen_x = x + int(image_rect.x() * self.scale_factor)
            screen_y = y + int(image_rect.y() * self.scale_factor)
            screen_w = int(image_rect.width() * self.scale_factor)
            screen_h = int(image_rect.height() * self.scale_factor)
            
            return QRect(screen_x, screen_y, screen_w, screen_h)
        except (OverflowError, ValueError):
            return QRect()
    
    def pick_color(self, point):
        """Pick color from the image at the given point"""
        if not self.image_pixmap:
            return
            
        image_point = self.screen_to_image_coords(point)
        if not image_point:
            return
            
        # Get color from the original image
        color = self.image_pixmap.toImage().pixelColor(image_point.x(), image_point.y())
        if color.isValid():
            self.paint_color = color
            # Emit signal or call parent method to update color display
            parent = self.parent()
            while parent and not hasattr(parent, 'update_color_display'):
                parent = parent.parent()
            if parent and hasattr(parent, 'update_color_display'):
                parent.update_color_display(color)
    
    def clear_paint(self):
        """Clear all paint marks"""
        if self.paint_overlay:
            self.paint_overlay.fill(Qt.transparent)
            self.has_painted = False  # Reset paint flag
            self.update()
    
    def get_combined_image(self):
        """Get the original image combined with paint overlay"""
        if not self.image_pixmap or not self.paint_overlay:
            return self.image_pixmap
        
        # Create a copy of the original image
        combined = QPixmap(self.image_pixmap)
        
        # Paint the overlay on top
        painter = QPainter(combined)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.drawPixmap(0, 0, self.paint_overlay)
        painter.end()
        
        return combined
    
    def has_paint_marks(self):
        """Check if there are any paint marks on the overlay"""
        return self.has_painted
    
    def set_paint_color(self, color):
        """Set the paint color"""
        self.paint_color = color
    
    def set_brush_size(self, size):
        """Set the brush size"""
        self.brush_size = size
    
    def set_dropper_mode(self, active):
        """Set dropper tool active/inactive"""
        self.is_dropper_active = active
        if active:
            self.setCursor(Qt.CrossCursor)
            self.is_crop_mode = False  # Disable crop mode
        else:
            self.setCursor(Qt.ArrowCursor)
    
    def set_crop_mode(self, active):
        """Set crop mode active/inactive"""
        self.is_crop_mode = active
        if active:
            self.setCursor(Qt.ArrowCursor)
            self.is_dropper_active = False  # Disable dropper mode
            self.initialize_crop_selection()  # Start with full image selected
        else:
            self.setCursor(Qt.ArrowCursor)
            self.clear_crop_selection()
    
    def has_crop_selection_ready(self):
        """Check if there's a valid crop selection"""
        if not self.has_crop_selection or not self.crop_rect.isValid():
            return False
        
        # Check if crop selection is different from the full image
        if not self.image_pixmap:
            return False
        
        full_rect = QRect(0, 0, self.image_pixmap.width(), self.image_pixmap.height())
        return self.crop_rect != full_rect