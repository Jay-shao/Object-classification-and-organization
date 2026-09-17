# -*- coding: utf-8 -*-
"""
横移 + 里程计测试 —— 跑在 Python 3.8（已装 robomaster）。

会依次：
    1) 订阅底盘里程计(sub_position)，打印初始位置
    2) 前进 20cm，打印位置（看里程计 x=前进 有没有变）
    3) 左移 20cm，打印位置（看里程计 y=横移 有没有变，同时看车有没有真的横着走）

用法：
    python test_strafe_odom.py            # WiFi 直连(ap)
    python test_strafe_odom.py rndis      # USB
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

    pos = {"x": 0.0, "y": 0.0, "z": 0.0}

    def cb(p):
        pos["x"], pos["y"], pos["z"] = p[0], p[1], p[2]

    r.chassis.sub_position(cs=0, freq=10, callback=cb)
    time.sleep(0.6)

    def show(tag):
        print(f"[{tag}] 里程计 x(前)={pos['x']:.2f}m  y(侧)={pos['y']:.2f}m  z(向)={pos['z']:.1f}°")

    show("初始")

    print("\n[1] 前进 20cm ……")
    r.chassis.drive_speed(x=0.15, y=0, z=0, timeout=1.3)
    time.sleep(1.5)
    r.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
    show("前进后")

    print("\n[2] 左移 20cm ……（重点看车有没有横着走、y 有没有变）")
    r.chassis.drive_speed(x=0, y=0.15, z=0, timeout=1.3)
    time.sleep(1.5)
    r.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
    show("左移后")

    print("\n测试结束。请对照：车横移了没？里程计 y 变了没？")
    r.close()


if __name__ == "__main__":
    main()
