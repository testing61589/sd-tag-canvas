"""Tags editing and management for the Image Tag Editor."""

from pathlib import Path
from PySide6.QtWidgets import QLabel, QTextEdit, QPushButton
from PySide6.QtGui import QFont, QTextCharFormat, QTextCursor

from config import TAGS_EDITOR_MIN_HEIGHT, TAGS_EDITOR_MAX_HEIGHT
from PySide6.QtGui import QTextOption


class TagsManager:
    """Manages tags editing, loading, saving, and highlighting."""

    def __init__(self, parent):
        self.parent = parent
        self.original_tags_content = ""

    def create_tags_section(self, layout):
        """Create tags section and add to layout."""
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
        self.save_changes_btn.clicked.connect(self.parent.save_changes)
        layout.addWidget(self.save_changes_btn)

    def load_tags(self, image_path: Path):
        """Load tags from file and display."""
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
            self.parent.highlight_trigger_words()
            self.tags_edit.blockSignals(False)
        except Exception as e:
            self.parent.statusBar().showMessage(f"Error loading tags: {e}")
            self.tags_edit.setPlainText("")
            self.original_tags_content = ""
            self.update_unsaved_status()

    def save_changes(self):
        """Save changes to tags and trigger status update."""
        if not self.parent.current_image_path:
            return

        # Save tags
        tags_content = self.tags_edit.toPlainText().strip()
        tags_file = self.parent.current_image_path.with_suffix('.txt')
        self._save_tags_file(tags_content, tags_file)
        self.original_tags_content = tags_content

        # Trigger save if image needs it
        if hasattr(self.parent.image_canvas, 'needs_image_save') and self.parent.image_canvas.needs_image_save():
            pixmap = self.parent.image_canvas.get_combined_image() if self.parent.image_canvas.has_painted else self.parent.image_canvas.image_pixmap
            if self.parent._save_image_pixmap(pixmap):
                self.parent.image_canvas.set_image(self.parent.current_image_path)

        self.update_unsaved_status()

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes to tags."""
        if not self.parent.current_image_path:
            return False
        current_tags = self.tags_edit.toPlainText().strip()
        image_needs_save = getattr(self.parent.image_canvas, 'needs_image_save', lambda: False)()
        return current_tags != self.original_tags_content or image_needs_save

    def update_unsaved_status(self):
        """Update save button enabled state based on unsaved changes."""
        self.save_changes_btn.setEnabled(self.has_unsaved_changes())

    def auto_resize_text_edit(self):
        """Auto-resize the text edit based on content."""
        doc_height = self.tags_edit.document().size().height()
        padding = 20
        new_height = int(doc_height + padding)
        new_height = max(TAGS_EDITOR_MIN_HEIGHT, min(TAGS_EDITOR_MAX_HEIGHT, new_height))
        self.tags_edit.setFixedHeight(new_height)

    def _save_tags_file(self, tags_content: str, tags_file: Path):
        """Save tags to file."""
        if tags_content:
            with open(tags_file, 'w', encoding='utf-8') as f:
                f.write(tags_content)
        else:
            if tags_file.exists():
                tags_file.unlink()

    def update_tags_text(self, text: str):
        """Update tags text content."""
        self.tags_edit.setPlainText(text)
        self.original_tags_content = text
        self.update_unsaved_status()

    def on_tags_changed(self):
        """Handle tags text change event."""
        self.parent.update_unsaved_status()
        self.parent.highlight_trigger_words()

    def trigger_tags_updated(self):
        """Signal that tags have been updated externally."""
        self.update_unsaved_status()
        self.parent.highlight_trigger_words()
