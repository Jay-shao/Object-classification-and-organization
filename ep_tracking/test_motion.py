# -*- coding: utf-8 -*-
"""
底盘运动测试 —— 跑在 Python 3.8（已装 robomaster）。

一边看摄像头画面，一边手动让车动，确认每个方向到底动不动：
    w = 前进 10cm
    s = 后退 10cm
    a = 左移 10cm
    d = 右移 10cm
    q = 退出

重点：按 a/d 时，看摄像头画面里的物体有没有「横着」移动。
如果按 a/d 车不动（或变成原地转），说明横移接口/固件有问题，我们再换 drive_wheels 方案。

用法：
    python test_motion.py               # AP 直连
    python test_motion.py --sn 序列号    # STA
"""
import argparse
import time

import cv2
from robomaster import robot

SPD = 0.15       # m/s
STEP = 0.10      # 每次 10cm


def move(chassis, x, y, z):
    dur = STEP / SPD
    chassis.drive_speed(x=x, y=y, z=z, timeout=dur)
    time.sleep(dur + 0.05)
    chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sn", default=None)
    args = parser.parse_args()

    ep = robot.Robot()
    if args.sn:
        ep.initialize(conn_type="sta", sn=args.sn)
    else:
        ep.initialize(conn_type="ap")
    ep.camera.start_video_stream(display=False)
    chassis = ep.chassis

    print("w前进 s后退 a左移 d右移 q退出（每次10cm）")
    try:
        while True:
            img = ep.camera.read_cv2_image(timeout=3)
            if img is not None:
                cv2.imshow("camera", img)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                move(chassis, +SPD, 0, 0)
            elif key == ord('s'):
                move(chassis, -SPD, 0, 0)
            elif key == ord('a'):
                move(chassis, 0, +SPD, 0)
            elif key == ord('d'):
                move(chassis, 0, -SPD, 0)
    finally:
        cv2.destroyAllWindows()
        ep.close()


if __name__ == "__main__":
    main()
