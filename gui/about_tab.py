from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt


class AboutTabMixin:
    """Mixin that creates the About tab (Tab 5)."""

    def _create_about_tab(self):
        """Tab 6: About / Info."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignCenter)

        # --- Original project credit ---
        orig_title = QLabel("TuNet")
        orig_title.setStyleSheet("font-size: 22pt; font-weight: bold; color: #7E3AF2;")
        orig_title.setAlignment(Qt.AlignCenter)
        orig_author = QLabel("Created by tpo.comp")
        orig_author.setStyleSheet("font-size: 12pt; color: #6b7280;")
        orig_author.setAlignment(Qt.AlignCenter)
        orig_desc = QLabel(
            "A direct, pixel-level mapping from source to destination images\n"
            "via an encoder-decoder network.")
        orig_desc.setAlignment(Qt.AlignCenter)
        orig_desc.setWordWrap(True)
        orig_link = QLabel('<a href="https://github.com/tpc2233/tunet">github.com/tpc2233/tunet</a>')
        orig_link.setOpenExternalLinks(True)
        orig_link.setAlignment(Qt.AlignCenter)

        # --- Separator ---
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        # --- Fork credit ---
        fork_title = QLabel("VFX Tools Fork")
        fork_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #16A34A;")
        fork_title.setAlignment(Qt.AlignCenter)
        fork_author = QLabel("Maintained by emlcpfx")
        fork_author.setStyleSheet("font-size: 11pt; color: #6b7280;")
        fork_author.setAlignment(Qt.AlignCenter)
        fork_changes = QLabel(
            "Changes since fork:\n"
            "\u2022 Unified PyQt GUI for training, inference, and export\n"
            "\u2022 Render queue with progress monitoring\n"
            "\u2022 Live preview with pan/zoom\n"
            "\u2022 Auto-matte / auto-mask generation\n"
            "\u2022 MSRN architecture option and BigCat features\n"
            "\u2022 Checkpoint resume, validation, and improved naming\n"
            "\u2022 Export to Flame (.gizmo) and Nuke (.nk) formats\n"
            "\u2022 Multi-OS support")
        fork_changes.setAlignment(Qt.AlignCenter)
        fork_changes.setWordWrap(True)
        fork_link = QLabel('<a href="https://github.com/emlcpfx/tunet">github.com/emlcpfx/tunet</a>')
        fork_link.setOpenExternalLinks(True)
        fork_link.setAlignment(Qt.AlignCenter)

        # --- Layout ---
        layout.addStretch()
        layout.addWidget(orig_title)
        layout.addWidget(orig_author)
        layout.addSpacing(8)
        layout.addWidget(orig_desc)
        layout.addSpacing(4)
        layout.addWidget(orig_link)
        layout.addSpacing(20)
        layout.addWidget(separator)
        layout.addSpacing(20)
        layout.addWidget(fork_title)
        layout.addWidget(fork_author)
        layout.addSpacing(8)
        layout.addWidget(fork_changes)
        layout.addSpacing(4)
        layout.addWidget(fork_link)
        layout.addStretch()
        self.tabs.addTab(tab, "About")
