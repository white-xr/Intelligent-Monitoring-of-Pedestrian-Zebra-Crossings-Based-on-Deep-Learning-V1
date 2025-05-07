import ultralytics
print(ultralytics.__version__)
import sys
sys.path.append('C:\\Users\\RUI\\Desktop\\PedestrianDetection-master\\PedestrianDetection-master\\src\\gui')
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from src.gui.mainwindow import PDMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = PDMainWindow()
    win.show()
    sys.exit(app.exec_())
