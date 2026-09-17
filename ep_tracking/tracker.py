# -*- coding: utf-8 -*-
"""
“抓取前目标丢失 -> 重新搜索 -> 跟随 -> 重新进入可抓取范围” 状态机。

纯逻辑，不含任何硬件 / ROS / SDK 调用；仿真与真机共用。

对应状态机：

    SEARCH ──找到目标──▶ LOCKED ──对中+靠近──▶ 连续确认 N 帧 ──▶ READY_TO_GRASP
       ▲                    │                                          │
       │                    └────────── 目标突然不见了 ────────────────┘
       └──────────── TRACK（扫描找目标）

关键点：
- “对中”用 bbox 中心相对画面中心的水平偏差 error_x 决定左转/右转；
- “靠近”用 bbox 高度占画面比例估算距离，越大越近；
- “抓前确认”需要连续多帧满足“居中且够近”才发 READY_TO_GRASP，
  这样你在它还没抓的瞬间把瓶子拿走，它不会空抓，而是回到搜索继续跟。
"""
import time


class TrackerConfig:
    def __init__(
        self,
        image_width=1280,
        image_height=720,
        deadband_px=60,       # 中心死区：|error_x| 小于它就不转，避免左右抖
        grasp_h_ratio=0.55,   # bbox 高度占画面比例 >= 此值认为“够近，可抓”
        min_conf=0.50,        # 低于此置信度视为没看到目标
        ready_frames=3,       # 抓前需连续确认的帧数（CHECK_TARGET）
        scan_flip_s=3.0,      # 连续丢失超过此秒数就反向扫描
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.center_x = image_width / 2.0
        self.deadband_px = deadband_px
        self.grasp_h_ratio = grasp_h_ratio
        self.grasp_h = image_height * grasp_h_ratio
        self.min_conf = min_conf
        self.ready_frames = ready_frames
        self.scan_flip_s = scan_flip_s


class TargetTracker:
    # 状态
    SEARCH = "SEARCH"   # 尚未找到目标
    LOCKED = "LOCKED"   # 已找到并锁定目标
    TRACK = "TRACK"     # 目标丢失，正在搜索/跟随

    # 输出动作（由驱动层翻译成具体运动指令）
    SCAN_RIGHT = "SCAN_RIGHT"
    SCAN_LEFT = "SCAN_LEFT"
    ROTATE_RIGHT = "ROTATE_RIGHT"
    ROTATE_LEFT = "ROTATE_LEFT"
    APPROACH = "APPROACH"          # 目标居中但还远 -> 前进
    HOLD = "HOLD"                  # 目标已居中且够近，再确认几帧
    READY_TO_GRASP = "READY_TO_GRASP"  # 确认无误，交给驱动层抓取

    def __init__(self, target_class, config=None):
        self.target_class = target_class
        self.cfg = config or TrackerConfig()
        self.state = self.SEARCH
        self.last_seen = None
        self.lost_since = None
        self.ready_streak = 0
        self.scan_dir = 1          # 1=右, -1=左
        self._last_flip = None

    def reset(self):
        """完成一次抓取后回到搜索，准备下一轮。"""
        self.state = self.SEARCH
        self.last_seen = None
        self.lost_since = None
        self.ready_streak = 0

    def _best(self, detections):
        best = None
        for detection in detections:
            if detection.name == self.target_class \
                    and detection.confidence >= self.cfg.min_conf:
                if best is None or detection.confidence > best.confidence:
                    best = detection
        return best

    def update(self, detections, now=None):
        """每帧调用。返回 (action, target_detection_or_None)。"""
        now = time.monotonic() if now is None else now
        target = self._best(detections)

        if target is not None:
            # 找到目标
            self.state = self.LOCKED
            self.last_seen = target
            self.lost_since = None
            error_x = target.cx - self.cfg.center_x

            # 1) 先对中：水平偏差大就转
            if abs(error_x) > self.cfg.deadband_px:
                self.ready_streak = 0
                action = self.ROTATE_RIGHT if error_x > 0 else self.ROTATE_LEFT
                return action, target

            # 2) 已居中但太远 -> 前进
            if target.h < self.cfg.grasp_h:
                self.ready_streak = 0
                return self.APPROACH, target

            # 3) 居中且够近：连续确认若干帧才真正抓，防“最后关头被拿走”
            self.ready_streak += 1
            if self.ready_streak >= self.cfg.ready_frames:
                return self.READY_TO_GRASP, target
            return self.HOLD, target

        # 没看到目标 -> 进入搜索/跟随
        if self.lost_since is None:
            self.lost_since = now
        self.state = self.TRACK
        self.ready_streak = 0

        # 连续丢一段时间就反向扫描，扩大搜索范围
        if self._last_flip is None or now - self._last_flip > self.cfg.scan_flip_s:
            self._last_flip = now
            self.scan_dir *= -1

        action = self.SCAN_RIGHT if self.scan_dir > 0 else self.SCAN_LEFT
        return action, None
