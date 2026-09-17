# -*- coding: utf-8 -*-
"""
分拣决策进程 —— 跑在 Python 3.12（torch / ultralytics / cv2）。

流程（顺序处理，从格子1开始）：
    1) 前进到工作线。
    2) 假设车启动时正对「格子1」（最左）。格子间距 16cm，共 9 个。
    3) 从格子1开始，依次：识别这格是什么 → 抓到对应放置位 → 下一个格子。
        空格子直接跳过。

定位：用 EP 里程计(chassis.sub_position)拿真实位置做闭环，不再靠瞎推算。
防糊：走得很慢，且每次到位后停稳才识别。

关键：判定「正前方是啥」用「离画面中心最近的物体」，而不是「置信度最高的」。

用法（先启动 robot_bridge.py）：
    python tracking_brain.py
"""
import argparse
import base64
import json
import socket
import time

import cv2
import numpy as np

from vision import YoloDetector, DEFAULT_WEIGHT

# ---- 场地（厘米）—— 需按实际标定 ----
WORK_Y = 42          # 检测工作线（先停在这个深度看）
APPROACH_Y = 47      # 抓取深度（比 WORK_Y 更靠前一点，给夹爪预留位置）
PLACE_Y = 40         # 放置位在前方
PLACE_X = 100        # 两个放置位 |x| = 100cm
GRID_SPACING = 16    # 格子间距 cm
NUM_GRIDS = 9        # 格子总数
STOP_WAIT = 0.4      # 每次到位后停稳的时间（秒），避免画面糊

# 里程计方向符号：车若往反方向走，把对应的改成 -1
LAT_SIGN = 1
FWD_SIGN = 1

CENTER_BAND_PX = 200  # 判定「物体在正前方」的像素带
DEADBAND_PX = 40      # 抓取对中死区（像素）
CAM_CENTER_X = 701    # 「正前方」在画面里的像素 x（用 calibrate_camera.py 测得）
MIN_CONF = 0.5
ALIGN_MAX_STEPS = 60

# 哪类放哪边：+1=右侧(+100)，-1=左侧(-100)
CLASS_SIDE = {"bottle": +1, "box": -1}
CLASS_COLORS = {"bottle": (0, 255, 0), "box": (255, 0, 0)}


class RobotClient:
    def __init__(self, host="127.0.0.1", port=8723):
        self.sock = socket.create_connection((host, port), timeout=30)
        self.reader = self.sock.makefile("r", encoding="utf-8")

    def rpc(self, req):
        self.sock.sendall((json.dumps(req) + "\n").encode())
        line = self.reader.readline()
        if not line:
            raise ConnectionError("robot_bridge 断开")
        resp = json.loads(line)
        if resp.get("type") == "error":
            print(f"  [bridge 报错] {resp.get('msg')}")
        return resp

    def get_frame(self):
        resp = self.rpc({"cmd": "get_frame"})
        if resp.get("type") != "frame":
            return None
        buf = base64.b64decode(resp["jpeg_b64"])
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def get_position(self):
        return self.rpc({"cmd": "get_position"})

    def forward(self, m): self.rpc({"cmd": "move", "op": "forward", "dist": m})
    def strafe(self, m): self.rpc({"cmd": "move", "op": "strafe", "dist": m})
    def stop(self): self.rpc({"cmd": "move", "op": "stop"})
    def grasp(self): self.rpc({"cmd": "grasp"})
    def release(self): self.rpc({"cmd": "release"})


class SortTask:
    def __init__(self, client, detector, conf):
        self.client = client
        self.detector = detector
        self.conf = conf
        self.img_w = 1280
        self.img_h = 720
        self.cx_forward = CAM_CENTER_X

    # ---------- 定位（里程计闭环，慢速 + 停稳） ----------
    def read_pos(self):
        resp = self.client.get_position()
        if resp.get("type") != "pos":
            return None
        return LAT_SIGN * resp["ly"] * 100.0, FWD_SIGN * resp["fx"] * 100.0

    def goto_lateral(self, tx, tol=1.5):
        for _ in range(300):
            p = self.read_pos()
            if p is None:
                time.sleep(0.05)
                continue
            err = tx - p[0]
            if abs(err) <= tol:
                time.sleep(STOP_WAIT)
                return True
            step = max(-2.0, min(2.0, err))   # 每次最多 2cm，慢
            self.client.strafe(step * 0.01)
            time.sleep(0.1)
        return False

    def goto_forward(self, ty, tol=1.5):
        for _ in range(300):
            p = self.read_pos()
            if p is None:
                time.sleep(0.05)
                continue
            err = ty - p[1]
            if abs(err) <= tol:
                time.sleep(STOP_WAIT)
                return True
            step = max(-2.0, min(2.0, err))
            self.client.forward(step * 0.01)
            time.sleep(0.1)
        return False

    def wait_first_frame(self):
        frame = None
        for _ in range(100):
            frame = self.client.get_frame()
            if frame is not None:
                break
            time.sleep(0.05)
        if frame is None:
            raise RuntimeError("拿不到真机画面")
        self.img_h, self.img_w = frame.shape[:2]
        print(f"图像 {self.img_w}x{self.img_h}")

    def show(self, frame, detections, text=""):
        img = frame.copy()
        cx = self.img_w // 2
        cv2.line(img, (cx, 0), (cx, self.img_h), (120, 120, 120), 1)
        for d in detections:
            color = CLASS_COLORS.get(d.name, (0, 255, 255))
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
            cv2.putText(img, f"{d.name} {d.confidence:.2f}",
                        (int(d.x1), max(14, int(d.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("EP 分拣", img)
        cv2.waitKey(1)

    # ---------- 检测辅助 ----------
    def _nearest(self, dets, cls=None):
        center = self.cx_forward
        best, best_dist = None, None
        for d in dets:
            if d.confidence < self.conf:
                continue
            if cls is not None and d.name != cls:
                continue
            dist = abs(d.cx - center)
            if best_dist is None or dist < best_dist:
                best, best_dist = d, dist
        return best, (best_dist if best_dist is not None else 1e9)

    def detect_center_class(self):
        frame = self.client.get_frame()
        if frame is None:
            return None
        dets = self.detector.detect(frame)
        self.show(frame, dets, "识别中")
        center = self.cx_forward
        desc = ", ".join(f"{d.name}@cx={d.cx:.0f}(距中{d.cx - center:+.0f})" for d in dets)
        if desc:
            print(f"    [检测] {desc}")
        t, dist = self._nearest(dets)
        if t is not None and dist <= CENTER_BAND_PX:
            return t.name
        return None

    # ---------- 抓取 + 放置 ----------
    def align_and_grasp(self, target_class):
        for _ in range(ALIGN_MAX_STEPS):
            frame = self.client.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            dets = self.detector.detect(frame)
            self.show(frame, dets, f"对中 {target_class}")
            t, dist = self._nearest(dets, target_class)
            if t is None:
                print("  没找到目标（可能被拿走），跳过")
                return False
            if dist > DEADBAND_PX:
                self.client.strafe(0.02 if (t.cx - self.cx_forward) > 0 else -0.02)
                time.sleep(0.1)
                continue
            break
        else:
            print("  对中超时，跳过")
            return False

        self.goto_forward(APPROACH_Y)

        frame = self.client.get_frame()
        if frame is not None:
            dets = self.detector.detect(frame)
            self.show(frame, dets, f"确认 {target_class}")
            t, _ = self._nearest(dets, target_class)
            if t is None:
                print("  抓前目标丢失，跳过")
                return False

        self.client.grasp()
        return True

    def place(self, side):
        self.goto_lateral(side * PLACE_X)
        self.goto_forward(PLACE_Y)
        self.client.release()

    def run(self):
        self.wait_first_frame()
        print("前进到工作线……")
        self.goto_forward(WORK_Y)
        print("从格子1开始，依次识别并分类摆放……\n")
        for n in range(1, NUM_GRIDS + 1):
            self.goto_forward(WORK_Y)
            self.goto_lateral((n - 1) * GRID_SPACING)
            cls = self.detect_center_class()
            if cls is None:
                print(f"格子{n}: 空")
                continue
            print(f"格子{n}: {cls} → 抓取")
            if self.align_and_grasp(cls):
                self.place(CLASS_SIDE.get(cls, +1))
            else:
                print(f"  格子{n} 抓取失败，跳过")
        print("\n全部处理完")


def main():
    parser = argparse.ArgumentParser(description="EP 分拣决策进程(3.12)")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8723)
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    args = parser.parse_args()

    detector = YoloDetector(weight=args.weight, conf=args.conf)
    client = RobotClient(args.host, args.port)
    try:
        SortTask(client, detector, args.conf).run()
    except KeyboardInterrupt:
        print("停止。")
    finally:
        try:
            client.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
