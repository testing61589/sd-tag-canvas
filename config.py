"""
Configuration and constants for the Image Tag Editor application.
"""

# Supported image file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

# Default paint settings
DEFAULT_PAINT_COLOR = (255, 0, 0)  # Red
DEFAULT_BRUSH_SIZE = 15
MIN_BRUSH_SIZE = 1
MAX_BRUSH_SIZE = 100

# Crop handle settings
CROP_HANDLE_SIZE = 12
MIN_CROP_SIZE = 10  # Minimum crop dimension in pixels

# Zoom settings
DEFAULT_ZOOM_FACTOR = 1.0
MIN_ZOOM_FACTOR = 0.1
MAX_ZOOM_FACTOR = 5.0
ZOOM_STEP = 0.25
MOUSE_WHEEL_ZOOM_STEP = 0.1

# UI Constants
LEFT_PANEL_MIN_WIDTH = 250
LEFT_PANEL_DEFAULT_WIDTH = 300
RIGHT_PANEL_DEFAULT_WIDTH = 900
TAGS_EDITOR_MIN_HEIGHT = 60
TAGS_EDITOR_MAX_HEIGHT = 200
PAINT_TOOLS_MAX_HEIGHT = 150  # Increased for zoom controls

# File settings
JPEG_QUALITY = 95

# Application info
APP_NAME = "Image Tag Editor with Paint and Crop Tools"
APP_VERSION = "1.3"