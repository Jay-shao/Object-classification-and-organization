# -*- coding: utf-8 -*-
"""
摄像头 + YOLO 联调脚本 —— 跑在 Python 3.12（torch/ultralytics/cv2）。

用途：验证 EP 自带摄像头能不能出图，以及 YOLOv11(bottle/box) 能不能在真机画面上跑出框。
**不会做任何底盘/机械臂动作**，纯粹取图 -> 检测 -> 显示，安全。

先启动 robot_bridge.py（Python 3.8），再运行本脚本：

    python camera_check.py                 # 摄像头 + YOLO 叠加显示
    python camera_check.py --no-yolo       # 只看原始摄像头画面（不跑模型）
    python camera_check.py --save out.mp4  # 顺便把画面存成 mp4

窗口里按 q 退出。
"""
import argparse
import base64
import json
import socket
import time

import cv2
import numpy as np

from vision import YoloDetector, DEFAULT_WEIGHT


class FrameClient:
    """与 robot_bridge.py 之间最小化的取图客户端（TCP/JSON，只取图不动车）。"""

    def __init__(self, host="127.0.0.1", port=8723):
        self.sock = socket.create_connection((host, port), timeout=30)
        self.sock.settimeout(5)
        self.reader = self.sock.makefile("r", encoding="utf-8")

    def get_frame(self):
        self.sock.sendall((json.dumps({"cmd": "get_frame"}) + "\n").encode())
        line = self.reader.readline()
        if not line:
            raise ConnectionError("robot_bridge 断开")
        resp = json.loads(line)
        if resp.get("type") != "frame":
            return None
        buf = base64.b64decode(resp["jpeg_b64"])
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def draw(frame, dets, fps):
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (120, 120, 120), 1)
    colors = {"bottle": (0, 255, 0), "box": (0, 165, 255)}
    for d in dets:
        c = colors.get(d.name, (0, 255, 255))
        cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), c, 2)
        cv2.putText(frame, f"{d.name} {d.confidence:.2f}",
                    (int(d.x1), max(16, int(d.y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    label = f"FPS {fps:.1f} | {w}x{h} | 目标 {len(dets)} 个"
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser(description="EP 摄像头 + YOLO 联调(3.12)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8723)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--weight", default=DEFAULT_WEIGHT)
    ap.add_argument("--no-yolo", action="store_true", help="只看原始画面，不跑模型")
    ap.add_argument("--save", default=None, help="存成视频文件，如 out.mp4")
    args = ap.parse_args()

    print(f"正在连接 robot_bridge {args.host}:{args.port} ……（若机器人还没连上，这里会反复重试，Ctrl+C 退出）")
    client = None
    deadline = time.time() + 120
    while client is None:
        try:
            client = FrameClient(args.host, args.port)
        except (ConnectionRefusedError, OSError, socket.timeout):
            if time.time() > deadline:
                raise SystemExit("连不上 robot_bridge，请确认 3.8 端已启动并连上真机。")
            time.sleep(1.0)
    detector = None if args.no_yolo else YoloDetector(weight=args.weight, conf=args.conf)
    if not args.no_yolo:
        print(f"YOLO 就绪，权重: {args.weight}，置信度阈值 {args.conf}")

    writer = None
    fps, frames, t0 = 0.0, 0, time.time()

    print("等待第一帧画面……（若一直没画面，检查：1) 机器人是否开机并连上 2) robot_bridge 是否打印『真机已连接』）")
    while True:
        frame = client.get_frame()
        if frame is None:
            print("  拿到空帧，重试……")
            time.sleep(0.05)
            continue
        break

    print(f"第一帧到达，分辨率 {frame.shape[1]}x{frame.shape[0]}。窗口按 q 退出。")

    try:
        while True:
            frame = client.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            dets = [] if detector is None else detector.detect(frame)
            frames += 1
            if frames % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - t0)
                t0 = now
                top = ", ".join(f"{d.name} {d.confidence:.2f}" for d in dets[:5])
                print(f"FPS {fps:.1f} | 检测到 {len(dets)} 个" + (f": {top}" if top else ""))

            draw(frame, dets, fps)
            cv2.imshow("EP camera + YOLO", frame)

            if writer is None and args.save:
                writer = cv2.VideoWriter(
                    args.save, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                    (frame.shape[1], frame.shape[0]))
            if writer is not None:
                writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("中断。")
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        try:
            client.sock.close()
        except Exception:
            pass
        print("已退出。")


if __name__ == "__main__":
    main()
