from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class PeopleDetect:
    def __init__(self):
        # 加载 YOLOv8 预训练模型
        self.model = YOLO("yolov8n.pt")
        # 加载中文字体
        self.font = ImageFont.truetype("SimHei.ttf", 20)

    def detectVideo(self, frame):
        # YOLOv8 进行视频帧检测
        results = self.model(frame)

        # 复制原图
        orig = frame.copy()

        # 转换为 PIL 图像以支持中文绘制
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_frame)

        person_count = 0  # 初始化行人计数

        # 解析检测结果
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 检测框坐标
                conf = float(box.conf[0])  # 置信度
                cls = int(box.cls[0])  # 类别索引

                # 仅统计行人数量
                if cls == 0 and conf > 0.5:  # 类别为行人且置信度大于 0.5
                    person_count += 1
                    # 绘制检测框和标签
                    draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
                    draw.text((x1, y1 - 20), f"行人 {conf:.2f}", font=self.font, fill="green")

        # 转换回 OpenCV 图像
        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

        return orig, frame, person_count