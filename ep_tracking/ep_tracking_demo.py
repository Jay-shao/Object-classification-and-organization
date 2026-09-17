# -*- coding: utf-8 -*-
"""
RoboMaster EP 真机：抓取前目标丢失 -> 视觉追踪 -> 跟随 -> 重新抓取 演示。

复用 vision.py 的 YOLO 检测 与 tracker.py 的追踪状态机；
运动层用 DJI robomaster SDK（纯 Python，不走 ROS2）。

用法：
    # 1) 先用电脑摄像头做纯视觉/逻辑联调（不碰真机，最推荐先跑这个）
    python ep_tracking_demo.py --class bottle --webcam

    # 2) 真机（AP 模式默认；STA 直连需提供序列号）
    python ep_tracking_demo.py --class bottle --sn 3JKDH2T00XXXXX

    # 3) 抓盒子
    python ep_tracking_demo.py --class box --webcam
"""
import argparse
import time

import cv2

from vision import YoloDetector, DEFAULT_WEIGHT
from tracker import TargetTracker, TrackerConfig


# ---------- 运动控制参数（真机需按场地标定） ----------
ROT_SPEED_DEG = 20.0      # 原地旋转角速度 deg/s
FWD_SPEED_MS = 0.15       # 前进速度 m/s
ROT_STEP_S = 0.30         # 每帧旋转步长（秒）
FWD_STEP_S = 0.30         # 每帧前进步长（秒）
GRASP_CLOSE_WAIT_S = 1.0  # 夹爪闭合后等待时间


class DryRunMotion:
    """纯逻辑联调用：只打印动作，不发任何真机指令。"""

    def __init__(self, log=print):
        self.log = log

    def rotate(self, deg):
        self.log(f"  [底盘] 原地旋转 {deg:+.1f}°")

    def approach(self, distance):
        self.log(f"  [底盘] 前进 {distance:.2f} m")

    def stop(self):
        self.log("  [底盘] 停车")

    def grasp(self, target):
        self.log(
            f"  [抓取] {target.name} conf={target.confidence:.2f} "
            f"-> 下降-夹取-抬升"
        )

    def release(self):
        self.log("  [抓取] 松爪")


class EPMotion:
    """真机运动层：封装 robomaster SDK 的底盘 / 夹爪 / 机械臂。"""

    def __init__(self, ep_robot):
        self.robot = ep_robot
        self.chassis = ep_robot.chassis
        self.gripper = ep_robot.gripper
        self.arm = ep_robot.robotic_arm

    def rotate(self, deg):
        # drive_speed 在 timeout 窗口内按给定角速度转；z>0 逆时针。
        sign = 1.0 if deg >= 0 else -1.0
        duration = abs(deg) / ROT_SPEED_DEG
        self.chassis.drive_speed(x=0, y=0, z=sign * ROT_SPEED_DEG, timeout=duration)
        time.sleep(duration + 0.05)

    def approach(self, distance):
        duration = distance / FWD_SPEED_MS
        self.chassis.drive_speed(x=FWD_SPEED_MS, y=0, z=0, timeout=duration)
        time.sleep(duration + 0.05)

    def stop(self):
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)

    def grasp(self, target):
        # 机械臂坐标 (x 前进 mm, y 高度 mm) 需按实际桌面标定，下面为占位示例。
        self.arm.moveto(x=120, y=30).wait_for_completed()   # 伸到抓取点（下探）
        self.gripper.close(power=60)
        time.sleep(GRASP_CLOSE_WAIT_S)
        self.arm.moveto(x=60, y=120).wait_for_completed()   # 抬升
        # 提示：你们“闭合耗时法”抓取检测已在 robomaster_ep_pick_place.py 里，
        # 可在这里复用它判断是否真夹到（closed≈1.0s 夹到 / ≈1.9s 空抓，阈值 1.4s）。

    def release(self):
        self.gripper.open(power=60)
        time.sleep(0.5)


def make_webcam_source(index=0):
    cap = cv2.VideoCapture(index)

    def get_frame():
        ok, frame = cap.read()
        return frame if ok else None

    return get_frame


def make_robot_source(ep_robot):
    camera = ep_robot.camera
    camera.start_video_stream(display=False)

    def get_frame():
        try:
            return camera.read_cv2_image(timeout=3)
        except Exception:
            return None

    return get_frame


def run(get_frame, motion, detector, target_class, conf):
    # 用第一帧尺寸初始化追踪参数（EP 相机 1280x720，摄像头通常 640x480）
    frame = None
    for _ in range(50):
        frame = get_frame()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        raise RuntimeError("拿不到相机画面")

    height, width = frame.shape[:2]
    config = TrackerConfig(image_width=width, image_height=height, min_conf=conf)
    tracker = TargetTracker(target_class, config)
    print(f"图像 {width}x{height}，目标类别={target_class}，开始追踪……")
    print("（把目标拿走再放到别处，观察它重新搜索并跟过去）")

    try:
        while True:
            frame = get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            detections = detector.detect(frame)
            action, target = tracker.update(detections)
            label = target.name if target else "-"
            print(f"[{tracker.state}] 动作={action:<15} 目标={label}")

            if action == TargetTracker.ROTATE_RIGHT:
                motion.rotate(+ROT_STEP_S * ROT_SPEED_DEG)
            elif action == TargetTracker.ROTATE_LEFT:
                motion.rotate(-ROT_STEP_S * ROT_SPEED_DEG)
            elif action == TargetTracker.SCAN_RIGHT:
                motion.rotate(+ROT_STEP_S * ROT_SPEED_DEG * 1.5)
            elif action == TargetTracker.SCAN_LEFT:
                motion.rotate(-ROT_STEP_S * ROT_SPEED_DEG * 1.5)
            elif action == TargetTracker.APPROACH:
                motion.approach(FWD_STEP_S * FWD_SPEED_MS)
            elif action == TargetTracker.HOLD:
                motion.stop()
            elif action == TargetTracker.READY_TO_GRASP:
                motion.stop()
                motion.grasp(target)
                motion.release()
                tracker.reset()
    except KeyboardInterrupt:
        print("停止。")
    finally:
        motion.stop()


def main():
    parser = argparse.ArgumentParser(description="EP 抓取前目标丢失视觉追踪演示")
    parser.add_argument("--class", dest="target_class", default="bottle",
                        help="目标类别：bottle / box")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--webcam", action="store_true",
                        help="用电脑摄像头做纯视觉联调，不连真机")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT, help="best.pt 路径")
    parser.add_argument("--sn", default=None,
                        help="真机序列号（STA 模式需要；AP 模式可省略）")
    args = parser.parse_args()

    detector = YoloDetector(weight=args.weight, conf=args.conf)

    if args.webcam:
        motion = DryRunMotion()
        get_frame = make_webcam_source(0)
    else:
        from robomaster import robot
        ep_robot = robot.Robot()
        if args.sn:
            ep_robot.initialize(conn_type="sta", sn=args.sn)
        else:
            ep_robot.initialize(conn_type="ap")
        motion = EPMotion(ep_robot)
        get_frame = make_robot_source(ep_robot)

    run(get_frame, motion, detector, args.target_class, args.conf)


if __name__ == "__main__":
    main()
