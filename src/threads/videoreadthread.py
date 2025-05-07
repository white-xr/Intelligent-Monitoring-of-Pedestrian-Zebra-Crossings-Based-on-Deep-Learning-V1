import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import cv2
from src.detect.peopledetect import PeopleDetect


class VideoReadThread(QThread):
    signalFrame = pyqtSignal(object, object, int)
    signalFailed = pyqtSignal(str)

    def __init__(self, parent=None):
        super(VideoReadThread, self).__init__(parent)
        self.work = False
        self.pause = False
        self.video_path = ""
        self.people_detect = PeopleDetect()

    def threadStart(self, path):
        self.video_path = path
        self.work = True
        self.pause = False
        self.start()

    def threadStop(self):
        self.work = False
        self.quit()
        self.wait()

    def run(self):
        try:
            self.capture = cv2.VideoCapture(self.video_path)

            # 设置分辨率
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置宽度
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置高度

            if not self.capture.isOpened():
                self.signalFailed.emit("无法打开视频文件或摄像头")
                return

            fps = self.capture.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
            frame_interval = 1 / fps if fps > 0 else 0.033  # 每帧间隔时间，默认30FPS

            while self.work:
                start_time = time.time()

                if self.pause:
                    self.msleep(100)  # 暂停时休眠，减少资源占用
                    continue

                success, frame = self.capture.read()
                if not success:
                    break

                original_frame = frame.copy()
                _, detected_frame, count = self.people_detect.detectVideo(frame)

                self.signalFrame.emit(original_frame, detected_frame, count)

                # 控制帧率
                elapsed_time = time.time() - start_time
                if elapsed_time < frame_interval:
                    time.sleep(frame_interval - elapsed_time)

            self.capture.release()
        except Exception as err:
            self.signalFailed.emit(str(err))