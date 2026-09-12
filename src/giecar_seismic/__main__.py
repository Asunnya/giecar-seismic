import sys

from PyQt5.QtWidgets import QApplication

from giecar_seismic.ui.main_window import MainWindow


def main() -> int:
    # No `service` is composed here: the SQLAlchemy-backed repositories
    # and SEG-Y/HDF5 factories FilterJobService needs for a real
    # end-to-end run don't exist yet. Wiring that composition here just to
    # make the entrypoint "work" would hide that gap behind a fake demo;
    # instead the window opens with `service=None`, visibly incomplete
    # (Run Filter stays disabled) until that composition is built.
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
