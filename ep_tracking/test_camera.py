# -*- coding: utf-8 -*-
"""
真机摄像头单独测试 —— 跑在 Python 3.8（已装 robomaster）。

只看摄像头画面，不做任何运动/检测，用来确认：
    1) 能不能取到画面、分辨率多少、帧率多少；
    2) 画面清不清楚、有没有对准桌子/物体、是否镜像/倒置；
    3) 按 s 存一帧，之后用 3.12 的 test_yolo_snapshot.py 单独看 YOLO 识别效果。

用法：
    python test_camera.py               # AP 直连
    python test_camera.py --sn 序列号    # STA

按键： s = 保存当前帧到 snapshots/ ； q = 退出
"""
import argparse
import os
import time

import cv2
from robomaster import robot


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

    os.makedirs("snapshots", exist_ok=True)
    count = 0
    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    print("摄像头测试中：s=存图  q=退出")
    try:
        while True:
            img = camera.read_cv2_image(timeout=3)
            if img is None:
                print("没取到画面")
                time.sleep(0.1)
                continue

            fps_n += 1
            if time.time() - fps_t0 >= 2.0:
                fps = fps_n / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_n = 0

            h, w = img.shape[:2]
            show = img.copy()
            cv2.putText(show, f"{w}x{h}  fps={fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("camera", show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                count += 1
                path = os.path.join("snapshots", f"frame_{count:03d}.jpg")
                cv2.imwrite(path, img)
                print(f"已保存 {path}  尺寸 {w}x{h}")
    finally:
        cv2.destroyAllWindows()
        ep.close()
        print("已退出")


if __name__ == "__main__":
    main()
