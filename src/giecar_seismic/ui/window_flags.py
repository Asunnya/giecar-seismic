"""Window-manager hints shared by the app's large, resizable dialogs."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog


def make_resizable_dialog(dialog: QDialog) -> None:
    """Give a QDialog minimize/maximize title-bar buttons.

    `Qt.Dialog` windows only get title/system-menu/close/"?" hints by default,
    so window managers draw no maximize button. Call before `show()`.
    """
    flags = dialog.windowFlags()
    flags &= ~Qt.WindowFlags(Qt.WindowType.WindowContextHelpButtonHint)
    flags |= Qt.WindowType.WindowMinimizeButtonHint
    flags |= Qt.WindowType.WindowMaximizeButtonHint
    dialog.setWindowFlags(flags)
