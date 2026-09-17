# -*- coding: utf-8 -*-
"""
底盘诊断 —— 跑在 Python 3.8（已装 robomaster）。

目的：确认底盘到底能不能动，并打印机器人当前模式。
会依次：
    1) 打印版本 + 当前模式
    2) 强制设成 free（自由）模式
    3) 直接用麦轮驱动前进 1 秒（drive_wheels，最底层的轮子控制）
    4) 再试 drive_speed 前进 0.5 秒

用法：
    python test_chassis.py            # WiFi 直连(ap)
    python test_chassis.py rndis      # USB 直连（最稳）
    python test_chassis.py sta --sn 序列号
"""
import argparse
import time

from robomaster import robot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("conn", nargs="?", default="ap", choices=["ap", "sta", "rndis"])
    parser.add_argument("--sn", default=None)
    args = parser.parse_args()

    r = robot.Robot()
    kwargs = {"sn": args.sn} if args.sn else {}
    r.initialize(conn_type=args.conn, **kwargs)
    print("连接成功，版本:", r.get_version())

    print("当前模式:", r.get_robot_mode())
    print("设置 free 模式:", r.set_robot_mode("free"))
    print("设置后模式:", r.get_robot_mode())

    print("\n[1] drive_wheels 前进 1 秒（直接麦轮）……")
    r.chassis.drive_wheels(w1=100, w2=100, w3=100, w4=100, timeout=1)
    time.sleep(1.2)
    r.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)

    print("[2] drive_speed 前进 0.5 秒……")
    r.chassis.drive_speed(x=0.2, y=0, z=0, timeout=0.5)
    time.sleep(0.6)
    r.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)

    print("\n诊断结束。请观察：车有没有往前动？")
    r.close()


if __name__ == "__main__":
    main()
