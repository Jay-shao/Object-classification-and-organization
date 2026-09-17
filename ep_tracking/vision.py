# -*- coding: utf-8 -*-
"""
YOLOv11 目标检测封装（bottle / box）。

框架无关：只负责“给一张 BGR 图像 -> 返回检测列表”，
仿真（ROS2 节点）和真机（robomaster SDK）都复用这一层，不重复造轮子。
"""
import os

from ultralytics import YOLO

# 训练好的最佳权重。优先用训练目录里这份，找不到再退回 zip 里那份。
DEFAULT_WEIGHT = (
    r"C:\Users\Administrator\yolo_lab3\runs\detect\runs\bottle_box\weights\best.pt"
)


class Detection:
    """一条检测结果：类别 + 置信度 + 归一化前的像素 bbox 中心与尺寸。"""

    __slots__ = ("class_id", "name", "confidence", "cx", "cy",
                 "x1", "y1", "x2", "y2", "w", "h")

    def __init__(self, class_id, name, confidence, x1, y1, x2, y2):
        self.class_id = int(class_id)
        self.name = name
        self.confidence = float(confidence)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0

    def __repr__(self):
        return (
            f"Detection({self.name}, {self.confidence:.2f}, "
            f"cx={self.cx:.0f}, cy={self.cy:.0f}, w={self.w:.0f}, h={self.h:.0f})"
        )


class YoloDetector:
    def __init__(self, weight=DEFAULT_WEIGHT, conf=0.25, imgsz=640, device=None):
        if not os.path.exists(weight):
            raise FileNotFoundError(f"找不到权重文件: {weight}")
        self.model = YOLO(weight)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        # ultralytics 的类别名表，形如 {0: 'bottle', 1: 'box'}
        self.names = self.model.names

    def detect(self, frame_bgr):
        """对一帧 BGR 图像做推理，返回 List[Detection]，按置信度从高到低排序。"""
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names
        detections = []
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(class_id, names[class_id], confidence, x1, y1, x2, y2)
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def find(self, frame_bgr, target_class, min_conf=0.5):
        """找指定类别（如 'bottle'）中置信度最高且 >= min_conf 的目标，找不到返回 None。"""
        best = None
        for detection in self.detect(frame_bgr):
            if detection.name == target_class and detection.confidence >= min_conf:
                if best is None or detection.confidence > best.confidence:
                    best = detection
        return best
