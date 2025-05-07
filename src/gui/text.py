import tkinter as tk
from tkinter import messagebox
import cv2
from ultralytics import YOLO
import threading

# 行人检测类
class PeopleDetect:
    def __init__(self):
        # 加载 YOLOv8 预训练模型（coco 数据集）
        self.model = YOLO("yolov8n.pt")

    def detectVideo(self, frame):
        # YOLOv8 进行视频帧行人检测
        results = self.model(frame)
        # 复制原图，用于显示未处理的画面
        orig = frame.copy()
        person_count = 0
        # 解析检测结果
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框坐标
                conf = float(box.conf[0])  # 置信度
                cls = int(box.cls[0])  # 类别索引

                # 只检测行人
                if cls == 0 and conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    person_count += 1
        return orig, frame, person_count


# 视频读取线程类
class VideoReadThread(threading.Thread):
    def __init__(self, text_box):
        threading.Thread.__init__(self)
        self.text_box = text_box
        self.capture = None
        self.is_running = False

    def threadStart(self):
        self.is_running = True
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            self.is_running = False
            return
        self.start()

    def run(self):
        people_detect = PeopleDetect()
        while self.is_running:
            ret, frame = self.capture.read()
            if not ret:
                break
            orig, detected_frame, person_count = people_detect.detectVideo(frame)
            self.update_text(person_count)
            cv2.imshow('People Detection', detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        cv2.destroyAllWindows()

    def update_text(self, person_count):
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, f"检测到人数: {person_count}")

    def threadStop(self):
        self.is_running = False


# 创建 Tkinter 主窗口
root = tk.Tk()
root.title("行人检测系统")

# 创建 Text 组件
text_box = tk.Text(root, height=2, width=30)
text_box.pack(pady=10)

# 创建开始和停止按钮
video_thread = VideoReadThread(text_box)


def start_detection():
    video_thread.threadStart()


def stop_detection():
    video_thread.threadStop()


start_button = tk.Button(root, text="开始检测", command=start_detection)
start_button.pack(pady=5)

stop_button = tk.Button(root, text="停止检测", command=stop_detection)
stop_button.pack(pady=5)

root.mainloop()