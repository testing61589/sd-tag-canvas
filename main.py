"""
Main entry point for the Image Tag Editor application.
"""

import sys
import argparse
import traceback
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMessageBox

from main_window import ImageTagEditor
from config import APP_NAME, APP_VERSION


def excepthook(exctype, value, tb):
    """Custom exception hook to print full stack traces"""
    print(f"\n{'='*60}")
    print(f"Unhandled exception occurred!")
    print(f"{'='*60}")
    traceback.print_exception(exctype, value, tb)
    print(f"{'='*60}\n")


def qt_error_handler(exctype, value, tb):
    """Handle uncaught Qt exceptions - calls sys.excepthook"""
    excepthook(exctype, value, tb)
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Image Tag Editor with Paint and Crop Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Open application without folder
  python main.py /path/to/images    # Open application with folder
  python main.py "C:\\My Images"    # Windows path with spaces
        """.strip()
    )
    
    parser.add_argument(
        'folder',
        nargs='?',  # Optional positional argument
        help='Path to the folder containing images to open automatically'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'{APP_NAME} {APP_VERSION}'
    )
    
    return parser.parse_args()


def validate_folder_path(folder_path):
    """Validate that the provided folder path exists and is accessible"""
    if not folder_path:
        return None, None
    
    try:
        path = Path(folder_path).resolve()
        
        if not path.exists():
            return None, f"Folder does not exist: {folder_path}"
        
        if not path.is_dir():
            return None, f"Path is not a directory: {folder_path}"
        
        # Try to list the directory to check permissions
        try:
            list(path.iterdir())
        except PermissionError:
            return None, f"Permission denied accessing folder: {folder_path}"
        
        return path, None
        
    except Exception as e:
        return None, f"Invalid folder path: {folder_path}\nError: {str(e)}"


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    # Enable unhandled exception logging inside Qt
    sys.excepthook = qt_error_handler
    
    # Parse command line arguments
    args = parse_arguments()

    # Validate folder path if provided
    initial_folder = None
    if args.folder:
        folder_path, error_msg = validate_folder_path(args.folder)
        if error_msg:
            # Show error dialog and continue with no initial folder
            QMessageBox.warning(
                None,
                "Folder Error",
                f"Could not open the specified folder:\n\n{error_msg}\n\n"
                "The application will start without opening a folder."
            )
        else:
            initial_folder = folder_path

    # Create and show main window
    try:
        window = ImageTagEditor(initial_folder=initial_folder)
        window.show()
    except Exception as e:
        print(f"FATAL: Failed to initialize UI: {e}")
        traceback.print_exc()
        sys.exit(1)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()