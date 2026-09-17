# -*- coding: utf-8 -*-
"""
真机桥接进程 —— 跑在 Python 3.8（已装 robomaster）。

负责：连 EP 真机、读自带摄像头、执行底盘/机械臂/夹爪动作。
通过本机 TCP(JSON) 与 tracking_brain.py(3.12) 通信。

坐标约定（厘米，与你给的 (0,0)/(0,70)/(±100,50) 一致）：
    起点 (0,0)，车头朝 +Y。物体行在 y=70，沿 x 排开。放置位 (+100,50)/(-100,50)。

注意：底盘 move(x=前进, y=侧移) 与你的 (x=侧移, y=前进) 正好对调，
本文件统一换算。真机若发现方向反了，只改下面 LANE_DIR / FORWARD_DIR 的符号。

用法：
    python robot_bridge.py                # AP 直连
    python robot_bridge.py --sn 序列号     # STA
"""
import argparse
import base64
import json
import socket
import time

import cv2
from robomaster import robot

HOST = "0.0.0.0"
PORT = 8723

# ---- 方向符号：真机若走反，只改这里 ----
LANE_DIR = 1        # +1: 往 +x（右侧放置位）是 move(y=+)；反了改 -1
FORWARD_DIR = 1     # +1: 往物体行（+y）是 move(x=+)；反了改 -1

# ---- 底盘速度（drive_speed 的 x/y 范围 [-3.5,3.5] m/s，z 范围 [-600,600] °/s）----
CHASSIS_SPD = 0.1   # 前进/横移速度 m/s（慢，避免移动时画面糊）
ROT_SPD = 30        # 旋转角速度 °/s
ARM_MOVE_S = 2.0    # 机械臂单次 moveto 大概耗时（秒），代替 wait_for_completed

# ---- 机械臂坐标（单位 mm；x:0~220, y:0~150）—— 需按实际标定 ----
ARM_GRAB = (140, 40)     # 下探抓取：x 前伸 140mm、y 高 40mm
ARM_LIFT = (80, 140)     # 抬升：收回来、抬高，避免拖地
ARM_RELEASE = (140, 60)  # 放下时的姿态

GRASP_CLOSE_WAIT_S = 1.0  # 夹爪闭合后等待时间


class RobotBridge:
    def __init__(self, conn_type="ap", sn=None):
        self.robot = robot.Robot()
        if sn:
            self.robot.initialize(conn_type="sta", sn=sn)
        else:
            self.robot.initialize(conn_type="ap")
        self.camera = self.robot.camera
        self.chassis = self.robot.chassis
        self.arm = self.robot.robotic_arm
        self.gripper = self.robot.gripper
        self.camera.start_video_stream(display=False)
        # 订阅底盘里程计，拿真实位置（回调收到 (x前进m, y侧移m, z朝向deg) 三元组）
        self.pos = (0.0, 0.0, 0.0)
        sub_ok = self.chassis.sub_position(cs=0, freq=10, callback=self._on_position)
        print(f"[bridge] 真机已连接，视频流已开启；位置订阅={sub_ok}")

    def _on_position(self, pos):
        self.pos = pos

    # ---------- 基础运动 ----------
    # 不用 chassis.move().wait_for_completed()：它在真机上等不到完成推送会卡死。
    # 改用 drive_speed（设速度 + 定时自动停，立即返回），自己 time.sleep 控制时长。
    def _drive(self, x, y, z, seconds):
        self.chassis.drive_speed(x=x, y=y, z=z, timeout=seconds)
        time.sleep(seconds + 0.05)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)

    def forward(self, meters):
        sign = 1.0 if meters >= 0 else -1.0
        self._drive(FORWARD_DIR * sign * CHASSIS_SPD, 0, 0, abs(meters) / CHASSIS_SPD)

    def strafe(self, meters):
        sign = 1.0 if meters >= 0 else -1.0
        self._drive(0, LANE_DIR * sign * CHASSIS_SPD, 0, abs(meters) / CHASSIS_SPD)

    def rotate(self, deg):
        sign = 1.0 if deg >= 0 else -1.0
        self._drive(0, 0, sign * ROT_SPD, abs(deg) / ROT_SPD)

    def stop(self):
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)

    # ---------- 抓取 / 放置 ----------
    # moveto 同样不用 wait_for_completed（也可能卡），发命令后用固定时间等它到位。
    def grasp(self):
        self.arm.moveto(*ARM_GRAB)          # 下探
        time.sleep(ARM_MOVE_S)
        self.gripper.close(power=60)
        time.sleep(GRASP_CLOSE_WAIT_S)
        self.arm.moveto(*ARM_LIFT)          # 抬升
        time.sleep(ARM_MOVE_S)

    def release(self):
        self.arm.moveto(*ARM_RELEASE)       # 下放到放置位
        time.sleep(ARM_MOVE_S)
        self.gripper.open(power=60)
        time.sleep(0.5)
        self.arm.moveto(*ARM_LIFT)          # 收回
        time.sleep(ARM_MOVE_S)

    # ---------- 取图 / 位置 ----------
    def get_frame(self):
        img = self.camera.read_cv2_image(timeout=3)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode(), img.shape[1], img.shape[0]

    def get_position(self):
        # pos = (x前进m, y侧移m, z朝向deg)
        return {"type": "pos", "fx": float(self.pos[0]),
                "ly": float(self.pos[1]), "z": float(self.pos[2])}

    # ---------- 指令分发 ----------
    def handle(self, req):
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"type": "ok"}
        if cmd == "get_frame":
            b64, w, h = self.get_frame()
            return {"type": "frame", "w": w, "h": h, "jpeg_b64": b64}
        if cmd == "get_position":
            return self.get_position()
        if cmd == "move":
            op = req.get("op")
            if op == "forward":
                self.forward(float(req.get("dist", 0)))
            elif op == "strafe":
                self.strafe(float(req.get("dist", 0)))
            elif op == "rotate":
                self.rotate(float(req.get("deg", 0)))
            elif op == "stop":
                self.stop()
            return {"type": "ok"}
        if cmd == "grasp":
            self.grasp()
            return {"type": "ok"}
        if cmd == "release":
            self.release()
            return {"type": "ok"}
        if cmd == "shutdown":
            return {"type": "bye"}
        return {"type": "error", "msg": "unknown cmd"}


def main():
    parser = argparse.ArgumentParser(description="EP 真机桥接进程(3.8)")
    parser.add_argument("--sn", default=None, help="真机序列号（STA 模式需要）")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    bridge = RobotBridge(sn=args.sn)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, args.port))
    server.listen(1)
    print(f"[bridge] 监听 {HOST}:{args.port}，等待 tracking_brain 连接……")

    conn, addr = server.accept()
    print(f"[bridge] 已连接 {addr}")
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
            try:
                resp = bridge.handle(req)
            except Exception as error:
                # 单条指令出错不能把整个 bridge 弄崩，打印真正的错误继续服务
                print(f"[bridge] 指令执行出错: {error}")
                resp = {"type": "error", "msg": str(error)}
            try:
                conn.sendall((json.dumps(resp) + "\n").encode())
            except OSError:
                break
            if resp.get("type") == "bye":
                break
    finally:
        bridge.stop()
        bridge.robot.close()
        server.close()
        print("[bridge] 已退出")


if __name__ == "__main__":
    main()
