import os

# Must be set before PyQt5 creates any QApplication. Respects an existing
# value (e.g. a developer running the suite with a real display), so this
# only supplies the headless default CI/sandboxed environments need.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication instance shared by every UI test in the
    session -- PyQt5 only allows one per process, and tests must not try
    to create their own.
    """
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def wait_for_signal(qapp):
    """Deterministically block the calling (GUI) thread until a Qt signal
    fires, using a local QEventLoop -- not sleep(), not
    QApplication.processEvents() in a polling loop. The timeout is a
    safety net against a genuine hang/deadlock, not a timing assumption:
    the loop always exits the instant the signal arrives, however long
    that takes.
    """

    def _wait(signal, timeout_ms: int = 5000) -> None:
        loop = QEventLoop()
        signal.connect(loop.quit)
        QTimer.singleShot(timeout_ms, loop.quit)
        loop.exec_()

    return _wait
