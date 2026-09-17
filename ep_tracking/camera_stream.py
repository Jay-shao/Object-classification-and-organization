# -*- coding: utf-8 -*-
"""
只取 RoboMaster EP 摄像头画面，通过本机 TCP 推给 camera_yolo.py（3.12）。

跑在 Python 3.8（已装 robomaster）。本脚本不做任何运动、不碰底盘/机械臂，只推画面。

为什么两个进程：robomaster 只在 3.8，YOLO 只在 3.12，没法合进同一个 Python。

用法：
    python camera_stream.py               # AP 直连
    python camera_stream.py --sn 序列号    # STA
"""
import argparse
import base64
import json
import socket

import cv2
from robomaster import robot

HOST = "0.0.0.0"
PORT = 8723


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sn", default=None, help="真机序列号（STA 模式需要）")
    args = parser.parse_args()

    ep = robot.Robot()
    if args.sn:
        ep.initialize(conn_type="sta", sn=args.sn)
    else:
        ep.initialize(conn_type="ap")

    camera = ep.camera
    camera.start_video_stream(display=False)
    print("[camera] 摄像头已开启")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[camera] 监听 {HOST}:{PORT}，等待 YOLO 端连接……")

    conn, addr = server.accept()
    print(f"[camera] 已连接 {addr}")
    reader = conn.makefile("r", encoding="utf-8")
    try:
        while True:
            line = reader.readline()
            if not line:
                break
            try:
                req = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if req.get("cmd") == "get_frame":
                img = camera.read_cv2_image(timeout=3)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf.tobytes()).decode()
                h, w = img.shape[:2]
                resp = {"type": "frame", "w": w, "h": h, "jpeg_b64": b64}
                conn.sendall((json.dumps(resp) + "\n").encode())
    finally:
        conn.close()
        server.close()
        ep.close()
        print("[camera] 已退出")


if __name__ == "__main__":
    main()
