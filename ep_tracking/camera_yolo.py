# -*- coding: utf-8 -*-
"""
EP 摄像头 + YOLO(best.pt) 实时画置信框 —— 跑在 Python 3.12。

为什么是两个进程：robomaster 只在 3.8，YOLO 只在 3.12，
所以 bridge(3.8) 负责取 EP 摄像头画面，本脚本(3.12) 负责跑 YOLO 画框。

用法（两个终端）：
    终端 A（3.8）：  python robot_bridge.py
    终端 B（3.12）： D:\\Anaconda\\file\\python.exe camera_yolo.py

按键： q 退出
"""
import argparse
import base64
import json
import socket
import time

import cv2
import numpy as np

from vision import YoloDetector, DEFAULT_WEIGHT

HOST = "127.0.0.1"
PORT = 8723
COLORS = {"bottle": (0, 255, 0), "box": (255, 0, 0)}


class CameraClient:
    """从 robot_bridge 取 EP 摄像头画面。"""

    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.create_connection((host, port), timeout=30)
        self.reader = self.sock.makefile("r", encoding="utf-8")

    def frame(self):
        self.sock.sendall((json.dumps({"cmd": "get_frame"}) + "\n").encode())
        line = self.reader.readline()
        if not line:
            return None
        resp = json.loads(line)
        if resp.get("type") != "frame":
            return None
        buf = base64.b64decode(resp["jpeg_b64"])
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def main():
    parser = argparse.ArgumentParser(description="EP 摄像头 + YOLO 实时识别")
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT, help="best.pt 路径")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    detector = YoloDetector(weight=args.weight, conf=args.conf)
    cam = CameraClient(args.host, args.port)
    print("EP 摄像头 + YOLO 实时识别已启动，q 退出")

    while True:
        img = cam.frame()
        if img is None:
            time.sleep(0.05)
            continue
        for d in detector.detect(img):
            color = COLORS.get(d.name, (0, 255, 255))
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
            cv2.putText(img, f"{d.name} {d.confidence:.2f}",
                        (int(d.x1), max(14, int(d.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("EP + YOLO", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
