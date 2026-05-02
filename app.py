import sys
from PySide6 import QtWidgets
from main_window import NoiseStudio

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = NoiseStudio()
    window.show()
    sys.exit(app.exec())