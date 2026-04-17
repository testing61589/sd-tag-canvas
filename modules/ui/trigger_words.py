"""Trigger words management for the Image Tag Editor."""

from pathlib import Path
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QInputDialog, QMessageBox
from PySide6.QtGui import QColor, Qt, QFont
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtCore import QSize

from config import TRIGGER_WORD_BUTTON_HEIGHT, BASE_TRIGGER_COLOR
from PySide6.QtGui import QTextCursor


class TriggerWordsManager:
    """Manages trigger word buttons and highlighting."""

    def __init__(self, parent):
        self.parent = parent
        self.trigger_words = []
        self.trigger_words_container = None
        self.trigger_word_colors = {}
        self._highlighting = False

    def create_trigger_words_section(self, layout):
        """Create trigger words section and add to layout."""
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
            if self.parent.current_folder:
                self.load_trigger_words()
                self.refresh_trigger_words_ui()
        except Exception as e:
            self.parent.statusBar().showMessage(f"Error creating trigger words section: {e}")
            import traceback
            traceback.print_exc()

    def load_trigger_words(self):
        """Load trigger words from _triggerwords.txt in current folder."""
        if not self.parent.current_folder:
            self.trigger_words = []
            return

        triggers_path = self.parent.current_folder / "_triggerwords.txt"

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
            self.parent.statusBar().showMessage(f"Error loading trigger words: {e}")
            self.trigger_words = []

    def refresh_trigger_words_ui(self):
        """Refresh trigger word buttons in UI."""
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

        # Build trigger word colors map
        self.trigger_word_colors = {}

        for i, word in enumerate(self.trigger_words):
            btn = QPushButton(word)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, w=word: self.on_trigger_word_clicked(w))

            # Base colors with progressive shading
            base_r, base_g, base_b = BASE_TRIGGER_COLOR
            shade_factor = max(0.40, 1.0 - (i * 0.10))
            r = int(base_r * shade_factor)
            g = int(base_g * shade_factor)
            b = int(base_b * shade_factor)

            # Store color for this trigger word
            self.trigger_word_colors[word.strip().lower()] = QColor(r, g, b)

            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

            # Style with minimal padding
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

    def on_trigger_word_clicked(self, trigger_word: str):
        """Add trigger word to front of comma-separated tags."""
        try:
            current_tags = self.parent.tags_manager.tags_edit.toPlainText().strip()
            if current_tags:
                new_tags = f"{trigger_word}, {current_tags}"
            else:
                new_tags = trigger_word

            self.parent.tags_manager.tags_edit.setPlainText(new_tags)
            self.parent.tags_manager.tags_edit.moveCursor(QTextCursor.Start)
            self.parent.update_unsaved_status()
        except Exception as e:
            import traceback
            print(f"Error adding trigger word '{trigger_word}': {e}")
            traceback.print_exc()

    def add_trigger_word(self):
        """Prompt user to add a new trigger word."""
        word, ok = QInputDialog.getText(
            self.parent,
            "Add Trigger Word",
            "Enter new trigger word:"
        )

        if ok and word.strip():
            self.trigger_words.append(word.strip())

            # Save to _triggerwords.txt
            triggers_path = self.parent.current_folder / "_triggerwords.txt"
            try:
                with open(triggers_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.trigger_words))

                # Refresh UI
                self.refresh_trigger_words_ui()
                self.parent.statusBar().showMessage(f"Added trigger word: {word.strip()}")
            except Exception as e:
                self.parent.statusBar().showMessage(f"Failed to save trigger words: {e}")
                QMessageBox.warning(self.parent, "Error", f"Failed to save trigger words: {e}")

    def highlight_trigger_words(self):
        """Highlight trigger words in the tags field by applying per-tag colors."""
        # Re-entrancy guard
        if self._highlighting:
            return
        self._highlighting = True

        try:
            # Block textChanged signal
            self.parent.tags_manager.tags_edit.blockSignals(True)

            if not self.trigger_words:
                self._clear_all_formats()
                return

            text = self.parent.tags_manager.tags_edit.document().toPlainText()
            if not text.strip():
                self._clear_all_formats()
                return

            # Clear existing formats first
            self._clear_all_formats()

            # Apply highlight colors
            from PySide6.QtGui import QTextCursor, QTextCharFormat
            cursor = QTextCursor(self.parent.tags_manager.tags_edit.document())
            cursor.setPosition(0)
            i = 0
            while i < len(text):
                # Skip leading commas and whitespace
                while i < len(text) and text[i] in ', \t\n\r':
                    i += 1

                if i >= len(text):
                    break

                tag_start = i
                # Only stop at commas; preserve spaces within tags
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
            self.parent.statusBar().showMessage(f"Error highlighting trigger words: {e}")
        finally:
            self._highlighting = False
            self.parent.tags_manager.tags_edit.blockSignals(False)

    def _clear_all_formats(self):
        """Clear all formatting and reset to widget's background color."""
        from PySide6.QtGui import QTextCursor, QTextCharFormat
        cursor = QTextCursor(self.parent.tags_manager.tags_edit.document())
        plain_format = QTextCharFormat()
        plain_format.setBackground(self.parent.tags_manager.tags_edit.palette().color(self.parent.tags_manager.tags_edit.backgroundRole()))
        cursor.select(QTextCursor.Document)
        cursor.mergeCharFormat(plain_format)
