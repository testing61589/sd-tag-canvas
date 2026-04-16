# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
# Start the GUI application
python main.py

# Start with a specific folder
python main.py /path/to/image/folder

# Show help and version
python main.py --help
python main.py --version
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
```

### Utility Scripts
```bash
# Convert tags to Danbooru format
python scripts/convert_to_danboory.py

# Remove duplicate images
python scripts/dedupe_images.py
```

## Architecture Overview

This is a PySide6-based image tag editor with painting, cropping, and zoom capabilities. The application follows a modular architecture:

### Core Application Structure
- **main.py**: Entry point with command-line argument parsing and application initialization
- **main_window.py**: Main GUI window (`ImageTagEditor` class) - handles UI layout, file browser, tool controls, and menu system
- **image_canvas.py**: Custom image display widget (`ImageCanvas` class) - handles image rendering, painting overlay, crop selection, zoom, and mouse interactions
- **config.py**: Centralized configuration constants for UI dimensions, zoom settings, paint settings, and supported file formats

### Key Components
- **Image Management**: Browse folders, navigate with arrow keys, automatic .txt tag file association
- **Paint System**: Non-destructive overlay painting with customizable brush size and color picker
- **Crop Tool**: Interactive crop selection with draggable handles and resize functionality
- **Zoom System**: Mouse wheel zoom, zoom controls (in/out/reset/fit), coordinate conversion between screen and image space
- **Tag Editor**: Text area for editing tags that are saved as .txt files alongside images

### Module System
- **modules/module/BaseImageCaptionModel.py**: Abstract base class for image captioning models with batch processing capabilities
- **modules/util/path_util.py**: Utility functions for path handling and supported image extension validation
- **scripts/**: Utility scripts for tag conversion and image deduplication using perceptual hashing

### Data Flow
1. Images loaded through file browser or command line
2. Image displayed in `ImageCanvas` with zoom and scroll support
3. Paint operations create overlay without modifying original image
4. Tags loaded from/saved to corresponding .txt files
5. Crop operations modify the actual image file
6. All modifications saved when navigating between images

### UI Layout
- Left panel: File browser and navigation
- Center: Scrollable image canvas with zoom support
- Right panel: Tag editor and tool controls (paint color, brush size, zoom controls)
- Menu bar: File operations and keyboard shortcuts (Ctrl+O, Ctrl+R, Ctrl+Q, arrow keys)

The modular design enables easy extension with new tools, image processing capabilities, and export formats while maintaining separation of concerns between UI, image handling, and data management.