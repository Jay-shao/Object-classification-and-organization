#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用训练好的 YOLOv11(bottle/box) 替换 grid_detector.py 的 ROS2 仿真检测节点。

接口保持不变：
    订阅 /sorting_camera/image      sensor_msgs/Image
    发布 /sorting/detections        vision_msgs/Detection2DArray

与 grid_detector 的差异：真正对相机画面跑 YOLO 推理，用 bbox 中心 -> 网格号。
任务控制器 sorting_task.py 无需改动即可继续使用。

部署到仿真包（在 Ubuntu 那台机器上）：
    1) 把本文件与 vision.py 一起放到 robomaster_ep_sorting_sim/scripts/
    2) 在 CMakeLists.txt 的 install(PROGRAMS ...) 里加入
           scripts/vision.py
           scripts/yolo_detector.py
    3) 把 launch 里 detector 的 executable 从 grid_detector.py 改成 yolo_detector.py
    4) 需要安装 cv_bridge：sudo apt install ros-humble-cv-bridge

注意：
    1) 仿真里物体是红色方块/蓝色圆柱，与真实 bottle/box 图像差异大，
       YOLO 未必能稳定识别，主要用于验证“接口打通”；真机才是该模型的主战场。
    2) bbox->grid 沿用 grid_detector 的映射约定；实拍/真机需重新标定。
    3) 类别映射：YOLO 的 box->class_1、bottle->class_2，按你们实验定义可改。
"""
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from cv_bridge import CvBridge

from vision import YoloDetector, DEFAULT_WEIGHT

# 与 grid_detector 保持一致的网格映射约定
GRID_IMAGE_X = {i: 100.0 + (i - 1) * 120.0 for i in range(1, 10)}

# YOLO 类别名 -> 任务控制器使用的 class_1 / class_2
CLASS_MAP = {"box": "class_1", "bottle": "class_2"}
MIN_CONFIDENCE = 0.50


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        weight = os.environ.get('YOLO_WEIGHT', DEFAULT_WEIGHT)
        self.detector = YoloDetector(weight=weight, conf=MIN_CONFIDENCE)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Detection2DArray, '/sorting/detections', 10
        )
        self.subscription = self.create_subscription(
            Image, '/sorting_camera/image', self.image_callback, 10
        )
        self.get_logger().info(f'YOLO detector ready, weight={weight}')

    def image_callback(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except Exception as error:
            self.get_logger().warn(f'imgmsg_to_cv2 failed: {error}')
            return

        detections = self.detector.detect(frame)
        output = Detection2DArray()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = 'sorting_camera'

        for detection in detections:
            if detection.confidence < MIN_CONFIDENCE:
                continue
            mapped = CLASS_MAP.get(detection.name)
            if mapped is None:
                continue

            grid = int(round((detection.cx - 100.0) / 120.0)) + 1
            if grid < 1 or grid > 9:
                continue

            message_det = Detection2D()
            message_det.header = output.header
            if hasattr(message_det, 'id'):
                message_det.id = f'grid_{grid}'

            center = message_det.bbox.center
            if hasattr(center, 'position'):
                center.position.x = float(detection.cx)
                center.position.y = float(detection.cy)
            else:
                center.x = float(detection.cx)
                center.y = float(detection.cy)

            message_det.bbox.size_x = float(detection.w)
            message_det.bbox.size_y = float(detection.h)

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = mapped
            hypothesis.hypothesis.score = float(detection.confidence)
            message_det.results.append(hypothesis)
            output.detections.append(message_det)

        self.publisher.publish(output)


def main():
    rclpy.init()
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
