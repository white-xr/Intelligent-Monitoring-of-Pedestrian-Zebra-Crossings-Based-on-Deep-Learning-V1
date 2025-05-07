import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QToolButton, QFileDialog, QMessageBox, QPushButton
from PyQt5.QtGui import QIcon, QCursor, QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow
import cv2
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import pyqtSlot
from src.detect.peopledetect import PeopleDetect
from src.gui import ui_mainwindow
from src.gui.constant import SYS_NAME, SYS_VERSION
from src.gui.selectpath import SelectPathDialog
from src.threads.videoreadthread import VideoReadThread
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time


class PDMainWindow(QMainWindow, ui_mainwindow.Ui_MainWindow):
    def __init__(self):
        super(PDMainWindow, self).__init__()
        self.setupUi(self)
        self.initWidget()
        self.initConnect()

        toolButton = self.findChild(QToolButton, 'toolButton')
        if toolButton:
            toolButton.clicked.connect(self.slotOpenVideo)

        self.people_detect = PeopleDetect()
        self.videoReadThread = VideoReadThread()
        self.videoReadThread.signalFrame.connect(self.slotUpdateResult)
        self.cameraReadThread = VideoReadThread()
        self.cameraReadThread.signalFrame.connect(self.slotUpdateResult)

        self.beginRecoding = False
        self.beginRecodingTime = ""
        self.video_out = None
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        self.pauseButton = self.findChild(QPushButton, 'pauseButton')
        if self.pauseButton:
            self.pauseButton.clicked.connect(self.slotPauseVideo)

    def output_text(self, text):
        current_text = self.textEdit.toPlainText()
        new_text = current_text + text + "\n"
        self.textEdit.setPlainText(new_text)

    def slotUpdateResult(self, original_frame, detected_frame, count):
        self.showImgToLabel(original_frame, 2)
        self.showImgToLabel(detected_frame, 2)
        self.lcdNumber.display(count)

        # 提取检测结果信息
        detection_summary = f"检测到人数: {count} 行人"
        self.outputToTextEdit(detection_summary)

        if self.beginRecoding:
            if self.video_out:
                self.video_out.write(detected_frame)
    def initWidget(self):
        try:
            self.setWindowModality(Qt.ApplicationModal)
            self.setWindowTitle(SYS_NAME)
            self.setWindowIcon(QIcon("./icons/icon.jpg"))
            self.statusbar.setStyleSheet("color:green")
        except Exception as e:
            print(f"初始化界面组件出错: {e}")

    def initConnect(self):
        self.actionopenImage.triggered.connect(self.slotOpenImage)
        self.actionopenVideo.triggered.connect(self.slotOpenVideo)
        self.actioncloseVideo.triggered.connect(self.slotCloseVideo)
        self.actionopenCam.triggered.connect(self.slotOpenCamera)
        self.actioncloseCam.triggered.connect(self.slotCloseCamera)
        self.actionbeginRecoding.triggered.connect(self.slotBeginRecoding)
        self.actionendRecoding.triggered.connect(self.slotEndRecoding)
        self.actionabout.triggered.connect(self.slotShowAbout)
        self.actioninstructions.triggered.connect(self.slotOpenInstructions)

    def slotOpenImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", "图片文件 (*.jpg *.png *.bmp)")
        if not filePath:
            print("未选择图片文件")
            return

        if not os.path.exists(filePath):
            print(f"图片路径无效: {filePath}")
            return

        try:
            frame = cv2.imread(filePath)
            if frame is None:
                print("无法读取图片")
                return

            # 调用检测方法
            orig, detected_frame, person_count = self.people_detect.detectVideo(frame)

            # 更新界面
            self.showImgToLabel(orig, 1)
            self.showImgToLabel(detected_frame, 2)
            self.lcdNumber.display(person_count)
            self.output_text(f"检测到人数: {person_count}")
        except Exception as e:
            print(f"图片处理失败: {e}")

    def slotOpenVideo(self):
        if self.cameraReadThread.isRunning():
            QMessageBox.information(self, '提示', '摄像头已经打开，请先关闭摄像头读取！', QMessageBox.Yes)
            return
        if self.videoReadThread.isRunning():
            QMessageBox.information(self, '提示', '视频已经打开，请先关闭视频读取！', QMessageBox.Yes)
            return

        spdia = SelectPathDialog()
        spdia.setType("video")
        spdia.exec()
        if spdia.getIsSure():
            filePath = spdia.getPath()
            if not os.path.exists(filePath):
                QMessageBox.information(self, '提示', '视频文件路径不存在！', QMessageBox.Yes)
                return
            self.videoReadThread.threadStart(filePath)

    def slotCloseVideo(self):
        self.videoReadThread.threadStop()
        QMessageBox.information(self, '提示', '已关闭！', QMessageBox.Yes)

    def slotOpenCamera(self):
        if self.videoReadThread.isRunning():
            QMessageBox.information(self, '提示', '请先关闭视频读取', QMessageBox.Yes)
            return
        if self.cameraReadThread.isRunning():
            QMessageBox.information(self, '提示', '摄像头已经打开，请先关闭摄像头读取！', QMessageBox.Yes)
            return
        self.cameraReadThread.threadStart(0)

    def slotCloseCamera(self):
        self.cameraReadThread.threadStop()
        QMessageBox.information(self, '提示', '已关闭！', QMessageBox.Yes)

    def slotBeginRecoding(self):
        if self.videoReadThread.isRunning() or self.cameraReadThread.isRunning():
            self.beginRecoding = True
            self.beginRecodingTime = time.strftime('%Y-%m-%d_%H:%M:%S')
            video_filename = '{}.avi'.format(self.beginRecodingTime)
            if self.videoReadThread.isRunning():
                first_frame = self.videoReadThread.capture.read()[1]
            elif self.cameraReadThread.isRunning():
                first_frame = self.cameraReadThread.capture.read()[1]
            else:
                QMessageBox.information(self, '提示', '视频或摄像头未开启，无法录制！', QMessageBox.Yes)
                return

            if first_frame is not None:
                height, width, _ = first_frame.shape
                self.video_out = cv2.VideoWriter(video_filename, self.fourcc, 30.0, (width, height))
                info = f'视频录制将保存于本地文件{video_filename}'
                self.statusbar.showMessage(info)
                QMessageBox.information(self, '提示', info, QMessageBox.Yes)
        else:
            QMessageBox.information(self, '提示', '视频或者摄像头未打开无法开启录制！', QMessageBox.Yes)

    def slotEndRecoding(self):
        if self.beginRecoding:
            if self.video_out:
                self.video_out.release()
            self.beginRecoding = False
            QMessageBox.information(self, '提示', '已结束录制！', QMessageBox.Yes)
        else:
            QMessageBox.information(self, '提示', '无法结束录制，请先开启录制！', QMessageBox.Yes)

    def showImgToLabel(self, frame, id=1):
        try:
            size = self.geometry()
            max_width = int(size.width() / 2 - 50)
            self.label_img2.setMaximumWidth(max_width)
            result_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qtImg = QImage(result_frame.data,
                           result_frame.shape[1],
                           result_frame.shape[0],
                           QImage.Format_RGB888)
            self.label_img2.setScaledContents(True)
            self.label_img2.setPixmap(QPixmap.fromImage(qtImg))
        except Exception as e:
            print(f"显示图片到label出错: {e}")

    def outputToTextEdit(self, text):
        current_text = self.textEdit.toPlainText()
        new_text = current_text + text + "\n"
        self.textEdit.setPlainText(new_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    def drawText(self, text, img):
        try:
            cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(pilimg)
            font = ImageFont.truetype("SimHei.ttf", 20, encoding="utf-8")
            draw.text((100, 50), text, (255, 0, 0), font=font)
            cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            return cv2charimg
        except Exception as e:
            print(f"绘制文字出错: {e}")
            return img

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '本程序',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.slotKillAllThread()
            event.accept()
        else:
            event.ignore()

    def slotKillAllThread(self):
        if self.videoReadThread.isRunning():
            self.videoReadThread.threadStop()
        if self.cameraReadThread.isRunning():
            self.cameraReadThread.threadStop()

    def slotShowAbout(self):
        info = "系统名称：{} \n 版本：{} ".format(SYS_NAME, SYS_VERSION)
        QMessageBox.information(self, '关于', info, QMessageBox.Yes)

    def slotOpenInstructions(self):
        QMessageBox.information(self, '使用说明', "请查看程序安装目录中”操作文档“", QMessageBox.Yes)

    def slotPauseVideo(self):
        if self.videoReadThread.isRunning():
            self.videoReadThread.pause = not self.videoReadThread.pause
            if self.videoReadThread.pause:
                QMessageBox.information(self, '提示', '视频已暂停！', QMessageBox.Yes)
            else:
                QMessageBox.information(self, '提示', '视频已继续播放！', QMessageBox.Yes)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    win = PDMainWindow()
    win.show()
    sys.exit(app.exec_())
