"""
Custom image canvas widget using QGraphicsView for displaying, painting, and cropping images with zoom support.
"""
import sys
print = lambda *args: sys.stderr.write(' '.join(map(str, args)) + '\\n') if args else None

from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem,
                               QSizePolicy, QFrame)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QRect, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QTransform, QFont

from config import (CROP_HANDLE_SIZE, MIN_CROP_SIZE, DEFAULT_ZOOM_FACTOR,
                   MIN_ZOOM_FACTOR, MAX_ZOOM_FACTOR, IMAGE_PADDING, DEFAULT_PAINT_COLOR)


class CropRectItem(QGraphicsRectItem):
    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.setPen(QPen(QColor(255, 255, 0), 3, Qt.SolidLine))  # Thicker for visibility
        self.setBrush(QBrush(Qt.NoBrush))
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self._drag_handle = None
        self._hover_handle = None
        self._whole_drag_start = None

    def _get_handle_at(self, pos):
        zoom = self.canvas.zoom_factor
        hs = max(20, CROP_HANDLE_SIZE / zoom)  # Adaptive size
        hs2 = hs / 2
        r = self.rect()
        print(f"Hit test: pos={pos}, hs={hs}, zoom={zoom}, rect={r}")

        handles = {
            'top-left': QRectF(r.left() - hs2, r.top() - hs2, hs, hs),
            'top-right': QRectF(r.right() - hs2, r.top() - hs2, hs, hs),
            'bottom-left': QRectF(r.left() - hs2, r.bottom() - hs2, hs, hs),
            'bottom-right': QRectF(r.right() - hs2, r.bottom() - hs2, hs, hs),
            'top': QRectF(r.center().x() - hs2, r.top() - hs2, hs, hs),
            'bottom': QRectF(r.center().x() - hs2, r.bottom() - hs2, hs, hs),
            'left': QRectF(r.left() - hs2, r.center().y() - hs2, hs, hs),
            'right': QRectF(r.right() - hs2, r.center().y() - hs2, hs, hs),
        }
        for name, hrect in handles.items():
            if hrect.contains(pos):
                print(f"HIT {name} {hrect}")
                return name
        print("No hit")
        return None

    def _get_cursor_for_handle(self, handle):
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
        return cursor_map.get(handle, Qt.ArrowCursor)

    def hoverMoveEvent(self, event):
        print("Crop hover")
        pos = event.pos()
        self._hover_handle = self._get_handle_at(pos)
        is_inside = self.rect().contains(pos)
        if self._hover_handle:
            cursor = self._get_cursor_for_handle(self._hover_handle)
        elif is_inside:
            cursor = Qt.SizeAllCursor
        else:
            cursor = Qt.ArrowCursor
        self.setCursor(cursor)
        self.update()
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        print("Crop press")
        if event.button() == Qt.LeftButton:
            local_pos = event.pos()
            handle = self._get_handle_at(local_pos)
            if handle:
                self._drag_handle = handle
                self._whole_drag_start = None
                print(f"Handle drag start: {handle}")
            elif self.rect().contains(local_pos):
                self._drag_handle = None
                self._whole_drag_start = self.mapToScene(local_pos)
                print("Whole drag start")
            else:
                super().mousePressEvent(event)
                return
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if self._drag_handle:
            self.canvas._resize_crop(scene_pos, self._drag_handle)
            self.canvas._notify_parent_crop_update()
        elif self._whole_drag_start is not None:
            delta = scene_pos - self._whole_drag_start
            self.canvas._move_crop_whole(delta)
            self._whole_drag_start = scene_pos
            self.canvas._notify_parent_crop_update()
        else:
            super().mouseMoveEvent(event)
            return
        event.accept()

    def mouseReleaseEvent(self, event):
        print("Crop release")
        self._drag_handle = None
        self._whole_drag_start = None
        event.accept()

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        zoom = self.canvas.zoom_factor
        hs = max(20, CROP_HANDLE_SIZE / zoom)
        hs2 = hs / 2
        r = self.rect()
        normal_color = QColor(255, 255, 0)
        hover_color = QColor(255, 255, 255)
        handle_pen = QPen(QColor(0, 0, 0), 2)

        handle_rects = [
            ('top-left', r.left() - hs2, r.top() - hs2),
            ('top-right', r.right() - hs2, r.top() - hs2),
            ('bottom-left', r.left() - hs2, r.bottom() - hs2),
            ('bottom-right', r.right() - hs2, r.bottom() - hs2),
            ('top', r.center().x() - hs2, r.top() - hs2),
            ('bottom', r.center().x() - hs2, r.bottom() - hs2),
            ('left', r.left() - hs2, r.center().y() - hs2),
            ('right', r.right() - hs2, r.center().y() - hs2),
        ]
        for name, hx, hy in handle_rects:
            hrect = QRectF(hx, hy, hs, hs)
            color = hover_color if name == self._hover_handle else normal_color
            painter.setBrush(color)
            painter.setPen(handle_pen if name == self._hover_handle else QPen(Qt.NoPen))
            painter.drawRect(hrect)


class ImageCanvas(QGraphicsView):
    zoom_changed = Signal(float)
    image_needs_save_changed = Signal(bool)

    def __init__(self):
        print("ImageCanvas __init__")
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(43, 43, 43))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("QGraphicsView { border: none; }")
        self.image_item = None
        self.paint_item = None
        self.image_pixmap = None
        self.paint_pixmap = None
        self.paint_color = QColor(*DEFAULT_PAINT_COLOR)
        self.brush_size = 15
        self.is_painting = False
        self.dropper_active = False
        self.crop_active = False
        self.original_image_size = None
        self._needs_image_save = False
        self.has_painted = False
        self.last_point = QPointF()
        self.crop_rect_item = None
        self.top_overlay = None
        self.bottom_overlay = None
        self.left_overlay = None
        self.right_overlay = None
        self.crop_rect = QRectF()
        self.zoom_factor = DEFAULT_ZOOM_FACTOR
        self.fit_zoom_factor = 1.0
        self.no_image_text = self.scene.addText("No image selected", QFont("Arial", 14))
        self.no_image_text.setDefaultTextColor(QColor(170, 170, 170))
        self.no_image_text.setPos(-200, -20)
        print("ImageCanvas init complete")

    def set_image(self, image_path):
        print(f"set_image({image_path})")
        try:
            self.image_pixmap = QPixmap(str(image_path))
            print(f"pixmap: {self.image_pixmap.size()}, valid: {not self.image_pixmap.isNull()}")
            if self.image_pixmap.isNull():
                self._clear_image_data()
                return False

            if self.image_item:
                self.scene.removeItem(self.image_item)
            if self.paint_item:
                self.scene.removeItem(self.paint_item)

            self.image_item = self.scene.addPixmap(self.image_pixmap)
            self.paint_pixmap = QPixmap(self.image_pixmap.size())
            self.paint_pixmap.fill(Qt.transparent)
            self.paint_item = self.scene.addPixmap(self.paint_pixmap)

            self.original_image_size = self.image_pixmap.size()
            self.has_painted = False
            self._update_needs_image_save()
            self.clear_crop_selection()
            self.no_image_text.setVisible(False)

            rect = QRectF(self.image_pixmap.rect()).adjusted(-IMAGE_PADDING, -IMAGE_PADDING, IMAGE_PADDING, IMAGE_PADDING)
            self.scene.setSceneRect(rect)

            self.zoom_factor = DEFAULT_ZOOM_FACTOR
            QTimer.singleShot(0, self._deferred_fit)
            return True
        except Exception as e:
            print(f"set_image error: {e}")
            self._clear_image_data()
            return False

    def _deferred_fit(self):
        print(f"_deferred_fit, canvas size: {self.size()}, viewport: {self.viewport().size()}")
        if self.image_item:
            rect = self._get_image_rect()
            self.fitInView(rect.adjusted(-IMAGE_PADDING, -IMAGE_PADDING, IMAGE_PADDING, IMAGE_PADDING), Qt.KeepAspectRatio)
            self.zoom_factor = min(self.transform().m11(), self.transform().m22())
            self.fit_zoom_factor = self.zoom_factor
            self.centerOn(self.image_item)
            self.zoom_changed.emit(self.zoom_factor)
            print(f"Fit complete, zoom: {self.zoom_factor}, scene rect: {self.scene.sceneRect()}")

    def _get_image_rect(self):
        return self.image_item.sceneBoundingRect() if self.image_item else QRectF()

    def _clear_image_data(self):
        print("_clear_image_data")
        self.clear_crop_selection()
        if self.image_item:
            self.scene.removeItem(self.image_item)
            self.image_item = None
        if self.paint_item:
            self.scene.removeItem(self.paint_item)
            self.paint_item = None
        self.image_pixmap = None
        self.paint_pixmap = None
        self.has_painted = False
        self._update_needs_image_save()
        self.zoom_factor = DEFAULT_ZOOM_FACTOR
        self.no_image_text.setVisible(True)
        self.fit_zoom_factor = 1.0
        self.scene.setSceneRect(QRectF())

    def zoom_in(self):
        if not self.image_item:
            return
        step = 0.25 if self.zoom_factor >= 1.0 else 0.1
        new_zoom = min(MAX_ZOOM_FACTOR, self.zoom_factor + step)
        self.set_zoom(new_zoom)

    def zoom_out(self):
        if not self.image_item:
            return
        step = 0.25 if self.zoom_factor > 1.0 else 0.1
        new_zoom = max(MIN_ZOOM_FACTOR, self.zoom_factor - step)
        self.set_zoom(new_zoom)

    def reset_zoom(self):
        if not self.image_item:
            return
        self.set_zoom(DEFAULT_ZOOM_FACTOR)

    def fit_to_window(self):
        print("fit_to_window called")
        if not self.image_item:
            return
        rect = self._get_image_rect()
        self.fitInView(rect.adjusted(-IMAGE_PADDING, -IMAGE_PADDING, IMAGE_PADDING, IMAGE_PADDING), Qt.KeepAspectRatio)
        self.zoom_factor = min(self.transform().m11(), self.transform().m22())
        self.fit_zoom_factor = self.zoom_factor
        self.zoom_changed.emit(self.zoom_factor)
        self.centerOn(self.image_item)

    def set_zoom(self, zoom_factor):
        if not self.image_item:
            return
        self.zoom_factor = max(MIN_ZOOM_FACTOR, min(MAX_ZOOM_FACTOR, zoom_factor))
        self.resetTransform()
        self.scale(self.zoom_factor, self.zoom_factor)
        self.zoom_changed.emit(self.zoom_factor)

    def get_zoom_percentage(self):
        return int(self.zoom_factor * 100)

    def resizeEvent(self, event):
        print(f"resizeEvent old: {event.oldSize()}, new: {event.size()}")
        super().resizeEvent(event)
        if self.image_item and abs(self.zoom_factor - self.fit_zoom_factor) < 0.01:
            self.fit_to_window()

    def wheelEvent(self, event):
        if self.image_item and event.modifiers() & Qt.ControlModifier:
            factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
            new_zoom = max(MIN_ZOOM_FACTOR, min(MAX_ZOOM_FACTOR, self.zoom_factor * factor))
            self.set_zoom(new_zoom)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if not self.image_item:
            super().mousePressEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        image_rect = self._get_image_rect()
        if not image_rect.contains(scene_pos):
            super().mousePressEvent(event)
            return
        if self.crop_active:
            super().mousePressEvent(event)
        elif self.dropper_active:
            self.pick_color(scene_pos)
            event.accept()
        else:
            if event.button() == Qt.LeftButton:
                self.is_painting = True
                self.last_point = scene_pos
                self.draw_point(scene_pos)
                event.accept()
            else:
                super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.image_item:
            super().mouseMoveEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        image_rect = self._get_image_rect()
        if not image_rect.contains(scene_pos):
            super().mouseMoveEvent(event)
            return
        if self.is_painting:
            self.draw_line(self.last_point, scene_pos)
            self.last_point = scene_pos
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.is_painting = False
        super().mouseReleaseEvent(event)

    def _resize_crop(self, scene_pos, handle):
        left, top, right, bottom = self.crop_rect.getCoords()
        img_w = self.image_item.pixmap().width()
        img_h = self.image_item.pixmap().height()
        if handle in ('top', 'bottom'):
            scene_pos.setX(max(left, min(scene_pos.x(), right)))
        elif handle in ('left', 'right'):
            scene_pos.setY(max(top, min(scene_pos.y(), bottom)))
        if handle == 'top-left':
            left = min(scene_pos.x(), right - MIN_CROP_SIZE)
            top = min(scene_pos.y(), bottom - MIN_CROP_SIZE)
        elif handle == 'top-right':
            right = max(scene_pos.x(), left + MIN_CROP_SIZE)
            top = min(scene_pos.y(), bottom - MIN_CROP_SIZE)
        elif handle == 'bottom-left':
            left = min(scene_pos.x(), right - MIN_CROP_SIZE)
            bottom = max(scene_pos.y(), top + MIN_CROP_SIZE)
        elif handle == 'bottom-right':
            right = max(scene_pos.x(), left + MIN_CROP_SIZE)
            bottom = max(scene_pos.y(), top + MIN_CROP_SIZE)
        elif handle == 'top':
            top = min(scene_pos.y(), bottom - MIN_CROP_SIZE)
        elif handle == 'bottom':
            bottom = max(scene_pos.y(), top + MIN_CROP_SIZE)
        elif handle == 'left':
            left = min(scene_pos.x(), right - MIN_CROP_SIZE)
        elif handle == 'right':
            right = max(scene_pos.x(), left + MIN_CROP_SIZE)
        left = max(0.0, left)
        top = max(0.0, top)
        right = min(float(img_w), right)
        bottom = min(float(img_h), bottom)
        if right - left < MIN_CROP_SIZE:
            center_x = (left + right) / 2
            left = max(0.0, center_x - MIN_CROP_SIZE / 2)
            right = min(float(img_w), left + MIN_CROP_SIZE)
        if bottom - top < MIN_CROP_SIZE:
            center_y = (top + bottom) / 2
            top = max(0.0, center_y - MIN_CROP_SIZE / 2)
            bottom = min(float(img_h), top + MIN_CROP_SIZE)
        self.crop_rect = QRectF(left, top, right - left, bottom - top)
        self.crop_rect_item.setRect(self.crop_rect)
        self._update_crop_overlays()

    def _move_crop_whole(self, delta):
        if not self.crop_rect_item:
            return
        img_w = float(self.image_item.pixmap().width())
        img_h = float(self.image_item.pixmap().height())
        new_left = self.crop_rect.left() + delta.x()
        new_top = self.crop_rect.top() + delta.y()
        w = self.crop_rect.width()
        h = self.crop_rect.height()
        new_left = max(0.0, min(new_left, img_w - w))
        new_top = max(0.0, min(new_top, img_h - h))
        self.crop_rect = QRectF(new_left, new_top, w, h)
        self.crop_rect_item.setRect(self.crop_rect)
        self._update_crop_overlays()

    def _notify_parent_crop_update(self):
        parent = self.parent()
        while parent and not hasattr(parent, 'update_crop_buttons'):
            parent = parent.parent()
        if parent:
            parent.update_crop_buttons()

    def clear_crop_selection(self):
        self.crop_active = False
        if self.crop_rect_item:
            self.scene.removeItem(self.crop_rect_item)
            self.crop_rect_item = None
        if self.top_overlay:
            self.scene.removeItem(self.top_overlay)
            self.top_overlay = None
        if self.bottom_overlay:
            self.scene.removeItem(self.bottom_overlay)
            self.bottom_overlay = None
        if self.left_overlay:
            self.scene.removeItem(self.left_overlay)
            self.left_overlay = None
        if self.right_overlay:
            self.scene.removeItem(self.right_overlay)
            self.right_overlay = None
        self.crop_rect = QRectF()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def initialize_crop_selection(self):
        if not self.image_item:
            return
        img_w = float(self.image_item.pixmap().width())
        img_h = float(self.image_item.pixmap().height())
        self.crop_rect = QRectF(0, 0, img_w, img_h)
        self.crop_rect_item = CropRectItem(self)
        self.scene.addItem(self.crop_rect_item)
        self.crop_rect_item.setRect(self.crop_rect)
        self.crop_rect_item.setZValue(2)
        self._create_crop_overlays()
        self._update_crop_overlays()

    def set_crop_mode(self, active):
        self.crop_active = active
        if active:
            self.dropper_active = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.initialize_crop_selection()
        else:
            self.clear_crop_selection()

    def has_crop_selection_ready(self):
        if not self.crop_active or not self.crop_rect_item:
            return False
        full_rect = QRectF(0, 0, self.image_item.pixmap().width(), self.image_item.pixmap().height())
        return self.crop_rect != full_rect

    def perform_crop(self):
        if not self.has_crop_selection_ready() or not self.image_item:
            return False
        crop_r = self.crop_rect.toRect()
        img_r = self._get_image_rect().toRect()
        crop_r = crop_r.intersected(img_r)
        if crop_r.isEmpty():
            return False
        cropped_image = self.image_pixmap.copy(crop_r)
        if self.has_painted:
            cropped_paint = self.paint_pixmap.copy(crop_r)
            painter = QPainter(cropped_image)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawPixmap(0, 0, cropped_paint)
            painter.end()
        self.image_pixmap = cropped_image
        self.image_item.setPixmap(self.image_pixmap)
        self.paint_pixmap = QPixmap(cropped_image.size())
        self.paint_pixmap.fill(Qt.transparent)
        self.paint_item.setPixmap(self.paint_pixmap)
        self.has_painted = False
        self._update_needs_image_save()
        if self.crop_active:
            self.crop_rect = QRectF(0, 0, cropped_image.width(), cropped_image.height())
            if self.crop_rect_item:
                self.crop_rect_item.setRect(self.crop_rect)
                self._update_crop_overlays()
        return True

    def pick_color(self, scene_pos):
        if not self.image_item:
            return
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        color = self.image_pixmap.toImage().pixelColor(x, y)
        if color.isValid():
            self.paint_color = color
            parent = self.parent()
            while parent and not hasattr(parent, 'update_color_display'):
                parent = parent.parent()
            if parent:
                parent.update_color_display(color)

    def draw_point(self, point):
        if not self.paint_pixmap:
            return
        painter = QPainter(self.paint_pixmap)
        painter.setPen(QPen(self.paint_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))
        painter.drawPoint(point)
        painter.end()
        self.paint_item.setPixmap(self.paint_pixmap)
        self.has_painted = True
        self._update_needs_image_save()

    def draw_line(self, start, end):
        if not self.paint_pixmap:
            return
        painter = QPainter(self.paint_pixmap)
        painter.setPen(QPen(self.paint_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(start, end)
        painter.end()
        self.paint_item.setPixmap(self.paint_pixmap)
        self.has_painted = True
        self._update_needs_image_save()

    def clear_paint(self):
        if self.paint_pixmap:
            self.paint_pixmap.fill(Qt.transparent)
            self.paint_item.setPixmap(self.paint_pixmap)
        self.has_painted = False
        self._update_needs_image_save()

    def needs_image_save(self):
        """Returns True if image has unsaved modifications (paint or crop)"""
        return self._needs_image_save

    def _update_needs_image_save(self):
        if not self.image_pixmap:
            self._needs_image_save = False
        else:
            orig_size = self.original_image_size
            curr_size = self.image_pixmap.size()
            self._needs_image_save = self.has_painted or (orig_size is not None and orig_size != curr_size)
        self.image_needs_save_changed.emit(self._needs_image_save)

    def has_paint_marks(self):
        return self.has_painted

    def get_combined_image(self):
        if not self.image_pixmap or not self.paint_pixmap:
            return self.image_pixmap
        combined = self.image_pixmap.copy()
        painter = QPainter(combined)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.drawPixmap(0, 0, self.paint_pixmap)
        painter.end()
        return combined

    def set_paint_color(self, color):
        self.paint_color = color

    def set_brush_size(self, size):
        self.brush_size = max(1, size)

    def _create_crop_overlays(self):
        gray_color = QColor(128, 128, 128, 128)
        no_pen = QPen(Qt.NoPen)
        brush = QBrush(gray_color)

        def make_overlay():
            overlay = QGraphicsRectItem()
            overlay.setZValue(1)
            overlay.setBrush(brush)
            overlay.setPen(no_pen)
            overlay.setFlag(QGraphicsItem.ItemIsMovable, False)
            overlay.setFlag(QGraphicsItem.ItemIsSelectable, False)
            overlay.setAcceptHoverEvents(False)
            self.scene.addItem(overlay)
            return overlay

        self.top_overlay = make_overlay()
        self.bottom_overlay = make_overlay()
        self.left_overlay = make_overlay()
        self.right_overlay = make_overlay()

    def _update_crop_overlays(self):
        if not self.image_item or not self.crop_rect_item:
            return
        img_w = float(self.image_item.pixmap().width())
        img_h = float(self.image_item.pixmap().height())
        crop = self.crop_rect

        # Top overlay
        self.top_overlay.setRect(QRectF(0, 0, img_w, crop.top()))

        # Bottom overlay
        self.bottom_overlay.setRect(QRectF(0, crop.bottom(), img_w, img_h - crop.bottom()))

        # Left overlay
        self.left_overlay.setRect(QRectF(0, crop.top(), crop.left(), crop.height()))

        # Right overlay
        self.right_overlay.setRect(QRectF(crop.right(), crop.top(), img_w - crop.right(), crop.height()))

    def set_dropper_mode(self, active):
        self.dropper_active = active
        if active:
            self.crop_active = False
            self.clear_crop_selection()
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            if not self.crop_active:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
