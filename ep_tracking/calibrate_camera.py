# -*- coding: utf-8 -*-
"""
摄像头正前方校准 —— 跑在 Python 3.12（连 bridge）。

目的：摄像头可能有偏角，「正前方」不在画面中心(640)。本脚本帮你测出
「正前方」到底对应哪个像素 cx，之后把 tracking_brain.py 里的 CAM_CENTER_X
改成这个值即可。

用法（先启动 robot_bridge.py 3.8）：
    1) 把一个物体【正对车头】放好（放在车正前方，别偏）。
    2) 跑本脚本，画面中央有条绿线。
    3) 按 c → 检测并打印这个物体的 cx。
    4) 把打印的 cx 填到 tracking_brain.py 的 CAM_CENTER_X。

按键： c=检测正前方物体并打印cx  q=退出
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


class Client:
    def __init__(self, host, port):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    det = YoloDetector(weight=DEFAULT_WEIGHT, conf=args.conf)
    c = Client(HOST, PORT)
    print("把一个物体【正对车头】放好，按 c 检测（q 退出）")

    while True:
        img = c.frame()
        if img is None:
            time.sleep(0.05)
            continue
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.putText(img, f"center={w//2}", (w // 2 + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("calibrate", img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            dets = det.detect(img)
            if dets:
                d = max(dets, key=lambda x: x.confidence)
                print(f"检测到 {d.name}(conf={d.confidence:.2f})，cx={d.cx:.0f}")
                print(f"  → 把 tracking_brain.py 顶部 CAM_CENTER_X 改成 {int(d.cx)}")
            else:
                print("没检测到物体，换个位置/光照，或把物体再放正一点")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
