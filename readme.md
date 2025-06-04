# Image Tag Editor

A PySide6-based application for editing images and managing associated tags with painting, cropping, and zoom capabilities.

## File Structure

The application has been modularized into the following files:

### Core Files
- **main.py** - Application entry point with command line argument support
- **config.py** - Configuration constants and settings
- **main_window.py** - Main application window (ImageTagEditor class)
- **image_canvas.py** - Custom image display widget with painting, cropping, and zoom (ImageCanvas class)

### File Organization
```
image_tag_editor/
├── main.py              # Application entry point
├── config.py            # Constants and configuration
├── main_window.py       # Main window class
├── image_canvas.py      # Custom image widget
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Features

- **Image Management**: Browse and navigate through image folders
- **Command Line Support**: Automatically open folders passed as arguments
- **Tag Editing**: Add and edit tags for each image (saved as .txt files)
- **Paint Tools**: Draw on images with customizable brush size and color
- **Color Picker**: Pick colors from the image using the dropper tool
- **Crop Tool**: Select and crop image areas with resize handles
- **Zoom Functionality**: Zoom in/out with multiple zoom options
- **Keyboard Navigation**: Use arrow keys to navigate between images

## Requirements

- Python 3.6+
- PySide6

Install dependencies:

```bash
pip install PySide6
```

Or install all dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the application without arguments:

```bash
python main.py
```

### Command Line Arguments

Open the application with a specific folder:

```bash
python main.py /path/to/your/images
```

Examples:

```bash
# Linux/macOS
python main.py ~/Pictures/vacation_photos
python main.py "/home/user/My Images"

# Windows
python main.py "C:\Users\Username\Pictures"
python main.py D:\Photos

# Get help
python main.py --help

# Show version
python main.py --version
```

Features of command line support:

- Validates that the folder exists and is accessible
- Shows error dialogs if the folder cannot be opened
- Gracefully continues without a folder if there are issues
- Supports paths with spaces (use quotes)
- Cross-platform path support

## Controls

### Keyboard Shortcuts
- **Ctrl+O**: Open folder
- **Ctrl+R**: Apply crop
- **Ctrl+Q**: Exit application
- **Ctrl++**: Zoom in
- **Ctrl+-**: Zoom out
- **Ctrl+0**: Reset zoom to 100%
- **Ctrl+F**: Fit image to window
- **↑/↓ Arrow Keys**: Navigate between images

### Mouse Controls
- **Ctrl+Mouse Wheel**: Zoom in/out on image
- **Click and Drag**: Paint on image (when paint tool is active)
- **Click**: Pick color from image (when dropper tool is active)
- **Drag Handles**: Resize crop selection (when crop tool is active)

## Tools

### Paint Tool (Default)
Click and drag to paint on the image with the selected brush size and color.

### Color Picker
Click the dropper button (🎨), then click on the image to pick colors.

### Crop Tool
Click the crop button (✂️) to enable crop mode. Drag the handles around the yellow selection rectangle to adjust the crop area, then click "Apply Crop" to crop the image.

### Zoom Tools
- **Zoom In (+)**: Increase zoom level
- **Zoom Out (−)**: Decrease zoom level  
- **Reset (1:1)**: Reset zoom to 100%
- **Fit**: Fit image to window size
- **Zoom Display**: Shows current zoom percentage
- **Mouse Wheel**: Hold Ctrl and scroll to zoom in/out

## Module Descriptions

### config.py
Contains all application constants and configuration settings:

- Supported image file extensions
- Default paint settings (color, brush size)
- Zoom settings (min/max zoom, zoom steps)
- UI dimensions and limits
- File format settings

### image_canvas.py
The ImageCanvas class provides:

- Image display with scaling, centering, and zoom support
- Paint overlay system for non-destructive editing
- Crop selection with draggable handles
- Zoom functionality with mouse wheel support
- Mouse interaction handling for painting, cropping, and zooming
- Coordinate conversion between screen and image space
- Scroll area support for zoomed images

### main_window.py
The ImageTagEditor class provides:

- Main application window and UI layout
- File browser and image navigation
- Paint tool controls (color picker, brush size, etc.)
- Zoom controls (zoom in/out, reset, fit to window)
- Tags editing interface
- Menu system and keyboard shortcuts
- Image and tag saving functionality
- Support for initial folder loading
- Scroll area for image display

### main.py
Application entry point that:

- Parses command line arguments using argparse
- Validates folder paths and handles errors gracefully
- Creates the QApplication instance
- Sets application metadata
- Creates and shows the main window with optional initial folder
- Starts the event loop

## Benefits of This Structure

- **Separation of Concerns**: Each file has a specific responsibility
- **Maintainability**: Easier to find and modify specific functionality
- **Reusability**: Components can be imported and used independently
- **Testability**: Individual modules can be tested in isolation
- **Configuration Management**: All settings centralized in config.py
- **Scalability**: Easy to add new features without cluttering existing files
- **Command Line Integration**: Supports automation and batch workflows
- **User Experience**: Smooth zoom and navigation controls

## Error Handling

The application includes robust error handling for:

- Invalid or non-existent folder paths
- Permission denied errors
- Malformed file paths
- Missing dependencies
- Image loading failures
- Zoom operations on invalid images

When errors occur with command line arguments, the application will:

- Display a clear error message
- Continue running without the problematic folder
- Allow the user to open a folder manually

## Future Enhancements

This modular structure makes it easy to add new features:

- Additional image editing tools (blur, sharpen, filters, etc.)
- Plugin system for custom tools
- Different export formats
- Undo/redo functionality
- Batch processing capabilities
- Additional command line options (e.g., --start-with-file)
- Configuration file support
- Advanced zoom features (zoom to selection, zoom to fit width/height)
- Pan functionality for zoomed images
- Thumbnail navigation panel