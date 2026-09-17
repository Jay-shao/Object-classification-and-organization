#!/usr/bin/env python3
"""RoboMaster EP nine-grid bottle/box sorting for Windows and Python 3.8.

The task follows the supplied Gazebo sorting state machine:

1. Visit nine pickup grids arranged in a straight line.
2. Use the RoboMaster camera and a YOLO model to identify bottle or box.
3. Skip an empty or low-confidence grid.
4. Approach, grasp, lift, and retreat.
5. Send bottles to the left bin and boxes to the right bin.
6. Return to the home line after every object and continue with the next grid.

OpenCV runs on the main thread so that the preview window remains responsive on
Windows. Robot control runs in a worker thread. Press Q or Esc in the preview,
or Ctrl+C in the console, to request an emergency stop.
"""

import argparse
import csv
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


Detection = Dict[str, Any]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a number".format(name))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


def validate_config(settings: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    connection = settings["connection"]
    vision = settings["vision"]
    layout = settings["layout"]
    motion = settings["motion"]
    arm = settings["arm"]

    if connection.get("type") not in ("rndis", "ap", "sta"):
        raise ValueError("connection.type must be rndis, ap, or sta")
    if connection.get("protocol", "tcp") not in ("tcp", "udp"):
        raise ValueError("connection.protocol must be tcp or udp")

    model_path = Path(str(vision.get("model", "best.pt")))
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    if not model_path.is_file():
        raise ValueError("YOLO model not found: {}".format(model_path))
    settings["_model_path"] = model_path.resolve()

    confidence = number(vision.get("confidence", 0.60), "vision.confidence")
    if not 0.05 <= confidence <= 0.99:
        raise ValueError("vision.confidence must be between 0.05 and 0.99")
    vision["confidence"] = confidence

    roi = vision.get("target_roi", [0.18, 0.12, 0.82, 0.95])
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("vision.target_roi must contain four normalized values")
    roi = [number(value, "vision.target_roi") for value in roi]
    if not (0.0 <= roi[0] < roi[2] <= 1.0 and 0.0 <= roi[1] < roi[3] <= 1.0):
        raise ValueError("vision.target_roi must be inside the image")
    vision["target_roi"] = roi

    grid_count = int(number(layout.get("grid_count", 9), "layout.grid_count"))
    center_index = int(number(layout.get("home_grid_index", 5), "layout.home_grid_index"))
    if grid_count != 9:
        raise ValueError("this task requires exactly nine grids")
    if center_index < 1 or center_index > grid_count:
        raise ValueError("layout.home_grid_index is outside the grid range")
    layout["grid_count"] = grid_count
    layout["home_grid_index"] = center_index
    layout["grid_spacing_m"] = number(layout.get("grid_spacing_m", 0.42), "layout.grid_spacing_m")
    layout["left_bin_y_m"] = number(layout.get("left_bin_y_m", 1.80), "layout.left_bin_y_m")
    layout["right_bin_y_m"] = number(layout.get("right_bin_y_m", -1.80), "layout.right_bin_y_m")
    if layout["grid_spacing_m"] <= 0.05:
        raise ValueError("layout.grid_spacing_m is too small")

    motion["grid_approach_m"] = number(motion.get("grid_approach_m", 0.72), "motion.grid_approach_m")
    motion["xy_speed_mps"] = number(motion.get("xy_speed_mps", 0.30), "motion.xy_speed_mps")
    motion["yaw_speed_dps"] = number(motion.get("yaw_speed_dps", 35.0), "motion.yaw_speed_dps")
    motion["bin_forward_m"] = number(motion.get("bin_forward_m", 0.45), "motion.bin_forward_m")
    if not 0.05 <= motion["xy_speed_mps"] <= 2.00:
        raise ValueError("motion.xy_speed_mps must be between 0.05 and 2.00")
    if not 5.0 <= motion["yaw_speed_dps"] <= 60.0:
        raise ValueError("motion.yaw_speed_dps must be between 5 and 60")
    if not 0.02 <= motion["grid_approach_m"] <= 2.00:
        raise ValueError("motion.grid_approach_m must be between 0.02 and 2.00")
    if not 0.05 <= motion["bin_forward_m"] <= 2.00:
        raise ValueError("motion.bin_forward_m must be between 0.05 and 2.00")

    for key in ("home_x_mm", "safe_y_mm", "place_x_mm", "place_y_mm"):
        arm[key] = number(arm[key], "arm." + key)
    arm["open_power"] = int(number(arm.get("open_power", 45), "arm.open_power"))
    arm["open_seconds"] = number(arm.get("open_seconds", 0.8), "arm.open_seconds")
    if not 1 <= arm["open_power"] <= 100:
        raise ValueError("arm.open_power must be between 1 and 100")
    if arm["open_seconds"] <= 0:
        raise ValueError("arm.open_seconds must be positive")
    for label in ("bottle", "box"):
        profile = arm["object_profiles"][label]
        profile["pick_x_mm"] = number(profile["pick_x_mm"], "arm.object_profiles.{}.pick_x_mm".format(label))
        profile["pick_y_mm"] = number(profile["pick_y_mm"], "arm.object_profiles.{}.pick_y_mm".format(label))
        profile["gripper_power"] = int(number(profile["gripper_power"], "arm.object_profiles.{}.gripper_power".format(label)))
        profile["close_seconds"] = number(profile["close_seconds"], "arm.object_profiles.{}.close_seconds".format(label))
        if not 1 <= profile["gripper_power"] <= 100:
            raise ValueError("gripper power must be between 1 and 100")

    routes = settings.get("class_routes", {})
    if routes.get("bottle") not in ("left", "right") or routes.get("box") not in ("left", "right"):
        raise ValueError("class_routes must map bottle and box to left/right")
    if routes["bottle"] == routes["box"]:
        raise ValueError("bottle and box must use different bins")

    settings["_config_path"] = config_path.resolve()
    return settings


def wait_action(action: Any, label: str, stop_event: threading.Event) -> None:
    if stop_event.is_set():
        raise InterruptedError("stop requested before {}".format(label))
    print("[ACTION] {}".format(label), flush=True)
    result = action.wait_for_completed()
    succeeded = bool(getattr(action, "has_succeeded", True))
    if not result or not succeeded:
        state = getattr(action, "state", "unknown")
        reason = getattr(action, "failure_reason", "")
        raise RuntimeError("{} failed: state={}, reason={}".format(label, state, reason))
    if stop_event.is_set():
        raise InterruptedError("stop requested after {}".format(label))


class VisionRuntime(object):
    """Reads the RoboMaster camera, performs YOLO inference, and shows OpenCV."""

    def __init__(
        self,
        camera_module: Any,
        model: Any,
        settings: Dict[str, Any],
        stop_event: threading.Event,
    ) -> None:
        self.camera = camera_module
        self.model = model
        self.settings = settings
        self.stop_event = stop_event
        self.condition = threading.Condition()
        self.frame_id = 0
        self.frame_size = (0, 0)
        self.latest_detections = []  # type: List[Detection]
        self.status = "Starting camera..."
        self.emergency_callback = None  # type: Optional[Callable[[], None]]
        self.window_name = "RoboMaster EP - YOLO Bottle/Box Sorting"

    def set_status(self, text: str) -> None:
        with self.condition:
            self.status = text

    def set_emergency_callback(self, callback: Callable[[], None]) -> None:
        self.emergency_callback = callback

    def request_stop(self, reason: str) -> None:
        if not self.stop_event.is_set():
            print("[EMERGENCY STOP] {}".format(reason), file=sys.stderr, flush=True)
        self.stop_event.set()
        with self.condition:
            self.status = "STOP REQUESTED: " + reason
            self.condition.notify_all()
        if self.emergency_callback is not None:
            try:
                self.emergency_callback()
            except Exception as error:
                print("[WARN] Emergency callback: {}".format(error), file=sys.stderr)

    def _infer(self, frame: Any) -> Tuple[Any, List[Detection]]:
        import cv2

        vision = self.settings["vision"]
        results = self.model.predict(
            source=frame,
            conf=vision["confidence"],
            iou=vision.get("iou", 0.45),
            imgsz=int(vision.get("image_size", 640)),
            device=str(vision.get("device", "cpu")),
            verbose=False,
        )
        result = results[0]

        height, width = frame.shape[:2]
        detections = []  # type: List[Detection]
        if result.boxes is not None:
            names = result.names
            xyxy_values = result.boxes.xyxy.detach().cpu().tolist()
            confidence_values = result.boxes.conf.detach().cpu().tolist()
            class_values = result.boxes.cls.detach().cpu().tolist()
            for xyxy, confidence, class_number in zip(
                xyxy_values, confidence_values, class_values
            ):
                class_index = int(class_number)
                label = str(names[class_index])
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                detection = {
                    "label": label.strip().lower(),
                    "confidence": float(confidence),
                    "bbox": (x1, y1, x2, y2),
                    "center_x": (x1 + x2) / 2.0,
                    "center_y": (y1 + y2) / 2.0,
                    "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                }
                if detection["label"] in ("bottle", "box") and self._plausible(
                    detection, (width, height)
                ):
                    detections.append(detection)

        # Draw only plausible bottle/box, plus the ROI and center line. Table legs
        # and people filtered out by _plausible are never shown as bottle/box.
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox"]]
            color = (255, 160, 0) if detection["label"] == "bottle" else (0, 200, 80)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                "{} {:.2f}".format(detection["label"], detection["confidence"]),
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        x1 = int(vision["target_roi"][0] * width)
        y1 = int(vision["target_roi"][1] * height)
        x2 = int(vision["target_roi"][2] * width)
        y2 = int(vision["target_roi"][3] * height)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.line(annotated, (width // 2, y1), (width // 2, y2), (255, 255, 0), 1)
        return annotated, detections

    def _plausible(self, detection: Detection, frame_size: Tuple[int, int]) -> bool:
        """Drop detections that are geometrically implausible for a bottle/box,
        such as razor-thin table legs or frame-filling people."""
        width, height = frame_size
        if width <= 0 or height <= 0:
            return False
        vision = self.settings["vision"]
        x1, y1, x2, y2 = detection["bbox"]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        aspect = box_h / box_w
        area_frac = (box_w * box_h) / float(width * height)
        width_frac = box_w / float(width)
        if not (number(vision.get("min_aspect_ratio", 0.30), "min_aspect_ratio")
                <= aspect <= number(vision.get("max_aspect_ratio", 5.00), "max_aspect_ratio")):
            return False
        if not (number(vision.get("min_area_frac", 0.004), "min_area_frac")
                <= area_frac <= number(vision.get("max_area_frac", 0.50), "max_area_frac")):
            return False
        if not (number(vision.get("min_width_frac", 0.02), "min_width_frac")
                <= width_frac <= number(vision.get("max_width_frac", 0.80), "max_width_frac")):
            return False
        return True

    @staticmethod
    def _overlay_status(frame: Any, status: str) -> Any:
        import cv2

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (15, 15, 15), -1)
        cv2.putText(
            frame,
            status[:110],
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Q/ESC: emergency stop",
            (12, frame.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def run_display_loop(self, done_event: threading.Event) -> None:
        import cv2

        finish_seen_at = None  # type: Optional[float]
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while not self.stop_event.is_set():
            frame = self.camera.read_cv2_image(strategy="newest", timeout=1.0)
            if frame is None:
                if done_event.is_set():
                    break
                continue
            try:
                annotated, detections = self._infer(frame)
            except Exception as error:
                self.request_stop("YOLO inference failed: {}".format(error))
                raise

            height, width = frame.shape[:2]
            with self.condition:
                self.frame_id += 1
                self.frame_size = (width, height)
                self.latest_detections = detections
                status = self.status
                self.condition.notify_all()

            annotated = self._overlay_status(annotated, status)
            cv2.imshow(self.window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self.request_stop("operator pressed Q/Esc")
                break

            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.request_stop("preview window was closed")
                    break
            except cv2.error:
                pass

            if done_event.is_set():
                if finish_seen_at is None:
                    finish_seen_at = time.monotonic()
                elif time.monotonic() - finish_seen_at >= 2.0:
                    break
        cv2.destroyAllWindows()

    def _best_target(self, detections: List[Detection], frame_size: Tuple[int, int]) -> Optional[Detection]:
        width, height = frame_size
        if width <= 0 or height <= 0:
            return None
        roi = self.settings["vision"]["target_roi"]
        candidates = []  # type: List[Detection]
        for detection in detections:
            if detection["label"] not in ("bottle", "box"):
                continue
            normalized_x = detection["center_x"] / width
            normalized_y = detection["center_y"] / height
            if not (roi[0] <= normalized_x <= roi[2] and roi[1] <= normalized_y <= roi[3]):
                continue
            candidates.append(detection)
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item["confidence"], item["area"]))

    def wait_for_first_frame(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.frame_id == 0 and not self.stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("RoboMaster camera frame timed out")
                self.condition.wait(timeout=min(0.5, remaining))
        if self.stop_event.is_set():
            raise InterruptedError("stop requested while waiting for camera")

    def wait_for_stable_target(self, timeout: float) -> Optional[Detection]:
        stable_frames = int(self.settings["vision"].get("stable_frames", 4))
        position_tol = number(
            self.settings["vision"].get("stable_position_frac", 0.12), "stable_position_frac"
        )
        deadline = time.monotonic() + timeout
        last_frame_id = -1
        stable_label = None  # type: Optional[str]
        stable_count = 0
        best = None  # type: Optional[Detection]
        last_center = None  # type: Optional[Tuple[float, float]]

        while not self.stop_event.is_set() and time.monotonic() < deadline:
            with self.condition:
                if self.frame_id == last_frame_id:
                    self.condition.wait(timeout=0.20)
                if self.frame_id == last_frame_id:
                    continue
                last_frame_id = self.frame_id
                detections = list(self.latest_detections)
                frame_size = self.frame_size

            candidate = self._best_target(detections, frame_size)
            if candidate is None:
                stable_label = None
                stable_count = 0
                best = None
                last_center = None
                continue

            center = (candidate["center_x"], candidate["center_y"])
            width, height = frame_size
            if last_center is not None and width > 0 and height > 0:
                dx = abs(center[0] - last_center[0]) / float(width)
                dy = abs(center[1] - last_center[1]) / float(height)
                if dx > position_tol or dy > position_tol:
                    # the "object" jumped between frames: treat as a moving false positive
                    stable_label = None
                    stable_count = 0
                    best = None
                    last_center = center
                    continue

            if candidate["label"] == stable_label:
                stable_count += 1
                if best is None or candidate["confidence"] > best["confidence"]:
                    best = candidate
            else:
                stable_label = candidate["label"]
                stable_count = 1
                best = candidate
            last_center = center
            if stable_count >= stable_frames:
                return dict(best) if best is not None else None

        if self.stop_event.is_set():
            raise InterruptedError("stop requested during visual detection")
        return None


class PhysicalSorter(object):
    """Executes the nine-grid state machine using the RoboMaster Python SDK."""

    def __init__(
        self,
        ep_robot: Any,
        vision: VisionRuntime,
        settings: Dict[str, Any],
        stop_event: threading.Event,
        done_event: threading.Event,
    ) -> None:
        self.robot = ep_robot
        self.chassis = ep_robot.chassis
        self.arm = ep_robot.robotic_arm
        self.gripper = ep_robot.gripper
        self.vision = vision
        self.settings = settings
        self.stop_event = stop_event
        self.done_event = done_event
        self.success = False
        self.error = None  # type: Optional[BaseException]
        self.current_world_y = 0.0
        self.current_yaw_deg = 0.0
        self.holding = False
        self.current_grid = 0

        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "sorting_{:%Y%m%d_%H%M%S}.csv".format(datetime.now())
        self.log_stream = self.log_path.open("w", newline="", encoding="utf-8-sig")
        self.log_writer = csv.writer(self.log_stream)
        self.log_writer.writerow(
            ["timestamp", "state", "grid", "label", "confidence", "result", "detail"]
        )
        self.log_stream.flush()

    def record(
        self,
        state: str,
        grid: Any = "",
        label: str = "",
        confidence: Any = "",
        result: str = "info",
        detail: str = "",
    ) -> None:
        message = "{} grid={} label={} confidence={} {}".format(
            state, grid, label, confidence, detail
        ).strip()
        print("[STATE] " + message, flush=True)
        self.vision.set_status(message)
        self.log_writer.writerow(
            [datetime.now().isoformat(), state, grid, label, confidence, result, detail]
        )
        self.log_stream.flush()

    def check_stop(self) -> None:
        if self.stop_event.is_set():
            raise InterruptedError("operator stop requested")

    def emergency_stop(self) -> None:
        try:
            self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
        except Exception:
            pass
        try:
            if hasattr(self.arm, "stop"):
                self.arm.stop()
        except Exception:
            pass
        try:
            self.gripper.pause()
        except Exception:
            pass

    def interruptible_sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check_stop()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def move_chassis(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, label: str = "chassis move") -> None:
        self.check_stop()
        motion = self.settings["motion"]
        wait_action(
            self.chassis.move(
                x=float(x),
                y=float(y),
                z=float(z),
                xy_speed=motion["xy_speed_mps"],
                z_speed=motion["yaw_speed_dps"],
            ),
            label,
            self.stop_event,
        )

    def move_arm(self, x_mm: float, y_mm: float, label: str) -> None:
        self.check_stop()
        wait_action(
            self.arm.moveto(x=int(round(x_mm)), y=int(round(y_mm))),
            label,
            self.stop_event,
        )

    def _gripper_command(self, command: Callable[[], Any], label: str) -> bool:
        attempts = int(self.settings["arm"].get("gripper_retries", 3))
        settle = number(
            self.settings["arm"].get("gripper_settle_seconds", 0.3), "gripper_settle_seconds"
        )
        # Let the arm finish settling before commanding the gripper; the SDK often
        # rejects a close/open that arrives while the arm is still busy.
        self.interruptible_sleep(settle)
        for attempt in range(1, attempts + 1):
            self.check_stop()
            try:
                accepted = command()
            except Exception as error:
                print(
                    "[WARN] {} error (attempt {}/{}): {}".format(label, attempt, attempts, error),
                    file=sys.stderr,
                    flush=True,
                )
                accepted = False
            if accepted is not False:
                return True
            print(
                "[WARN] {} rejected (attempt {}/{}); retrying...".format(label, attempt, attempts),
                file=sys.stderr,
                flush=True,
            )
            self.interruptible_sleep(0.4)
        return False

    def open_gripper(self) -> None:
        power = int(self.settings["arm"].get("open_power", 45))
        duration = number(self.settings["arm"].get("open_seconds", 0.8), "arm.open_seconds")
        self.check_stop()
        print("[ACTION] Open gripper", flush=True)
        accepted = self._gripper_command(lambda: self.gripper.open(power=power), "gripper open")
        if not accepted:
            raise RuntimeError("gripper open command was rejected")
        self.interruptible_sleep(duration)
        self.gripper.pause()
        self.holding = False

    def close_gripper(self, profile: Dict[str, Any]) -> None:
        self.check_stop()
        print("[ACTION] Close gripper", flush=True)
        accepted = self._gripper_command(
            lambda: self.gripper.close(power=int(profile["gripper_power"])), "gripper close"
        )
        if not accepted:
            raise RuntimeError("gripper close command was rejected")
        self.interruptible_sleep(float(profile["close_seconds"]))
        self.gripper.pause()
        self.holding = True

    def arm_home(self) -> None:
        wait_action(self.arm.recenter(), "Arm recenter", self.stop_event)
        arm_cfg = self.settings["arm"]
        self.move_arm(
            arm_cfg["home_x_mm"],
            arm_cfg["safe_y_mm"],
            "Move arm to safe home pose",
        )

    def arm_search_posture(self) -> None:
        """Lower the arm out of the camera's view while searching for targets on
        the ground, so the camera can actually see the objects."""
        arm_cfg = self.settings["arm"]
        self.move_arm(
            arm_cfg.get("search_x_mm", 180),
            arm_cfg.get("search_y_mm", 0),
            "Lower arm for searching",
        )

    def grid_world_y(self, grid: int) -> float:
        layout = self.settings["layout"]
        return (grid - layout["home_grid_index"]) * layout["grid_spacing_m"]

    def move_from_home_to_grid(self, grid: int) -> None:
        target_y = self.grid_world_y(grid)
        self.record("ALIGN_GRID", grid, detail="target_y={:.3f}m".format(target_y))
        self.move_chassis(y=target_y, label="Move laterally to grid {}".format(grid))
        self.current_world_y = target_y

    def return_scan_position_to_home(self, grid: int) -> None:
        # Empty grid: stay on the scan line and keep sweeping toward the next
        # grid instead of driving all the way back to the home line.
        pass

    def sweep_to_grid_and_detect(self, grid: int) -> Optional[Detection]:
        """Slide laterally toward the grid in small steps, watching for a target
        after each step. Stop as soon as a stable bottle/box is seen and return
        it, so we grasp where the object was actually found (not only at the
        nominal grid center)."""
        self.arm_search_posture()
        target_y = self.grid_world_y(grid)
        step = number(self.settings["vision"].get("sweep_step_m", 0.10), "sweep_step_m")
        check_timeout = float(self.settings["vision"].get("sweep_check_seconds", 0.8))
        remaining = target_y - self.current_world_y
        direction = 1.0 if remaining >= 0 else -1.0
        total = abs(remaining)
        self.record("ALIGN_GRID", grid, detail="target_y={:.3f}m sweep".format(target_y))
        travelled = 0.0
        while travelled < total - 1e-6:
            self.check_stop()
            move = min(step, total - travelled)
            self.move_chassis(y=direction * move, label="Sweep toward grid {}".format(grid))
            self.current_world_y += direction * move
            travelled += move
            target = self.vision.wait_for_stable_target(timeout=check_timeout)
            if target is not None:
                return target
        # Thorough final check at the target position. This also covers the home
        # grid (grid 5), where the sweep distance is zero.
        return self.vision.wait_for_stable_target(
            timeout=float(self.settings["vision"].get("scan_timeout_seconds", 3.0))
        )

    def _center_once(self, target: Detection) -> None:
        """Do a single lateral correction to put the target on the image center
        line before grasping. The sweep only gets the object roughly into the ROI."""
        width, _height = self.vision.frame_size
        if width <= 0:
            return
        vision_cfg = self.settings["vision"]
        deadband = number(vision_cfg.get("center_deadband", 0.06), "vision.center_deadband")
        gain = number(vision_cfg.get("center_gain_m", 0.35), "vision.center_gain_m")
        max_step = number(vision_cfg.get("max_center_step_m", 0.10), "vision.max_center_step_m")
        sign = number(vision_cfg.get("vision_to_chassis_y_sign", 1.0), "vision.vision_to_chassis_y_sign")
        center_x = number(vision_cfg.get("center_x_fraction", 0.5), "center_x_fraction") * width
        error = (target["center_x"] - center_x) / (width / 2.0)
        if abs(error) <= deadband:
            return
        correction = max(-max_step, min(max_step, sign * error * gain))
        self.move_chassis(y=correction, label="Center target before grasp")
        self.current_world_y += correction

    def align_with_detection(self, grid: int, target: Detection) -> Detection:
        vision_cfg = self.settings["vision"]
        deadband = number(vision_cfg.get("center_deadband", 0.06), "vision.center_deadband")
        gain = number(vision_cfg.get("center_gain_m", 0.35), "vision.center_gain_m")
        max_step = number(vision_cfg.get("max_center_step_m", 0.10), "vision.max_center_step_m")
        sign = number(vision_cfg.get("vision_to_chassis_y_sign", -1.0), "vision.vision_to_chassis_y_sign")
        iterations = int(vision_cfg.get("max_center_iterations", 3))

        for iteration in range(iterations):
            width, _height = self.vision.frame_size
            if width <= 0:
                raise RuntimeError("camera width is unavailable")
            center_x = number(vision_cfg.get("center_x_fraction", 0.5), "center_x_fraction") * width
            normalized_error = (target["center_x"] - center_x) / (width / 2.0)
            self.record(
                "VISION_ALIGN",
                grid,
                target["label"],
                "{:.2f}".format(target["confidence"]),
                detail="iteration={} horizontal_error={:.3f}".format(iteration + 1, normalized_error),
            )
            if abs(normalized_error) <= deadband:
                return target
            correction = max(-max_step, min(max_step, sign * normalized_error * gain))
            self.move_chassis(y=correction, label="Visual lateral correction")
            self.current_world_y += correction
            self.interruptible_sleep(0.35)
            updated = self.vision.wait_for_stable_target(
                timeout=float(vision_cfg.get("realign_timeout_seconds", 2.5))
            )
            if updated is None:
                raise RuntimeError("target was lost during visual alignment")
            if updated["label"] != target["label"]:
                raise RuntimeError("target class changed during visual alignment")
            target = updated
        return target

    def pick_object(self, grid: int, label: str) -> None:
        arm_cfg = self.settings["arm"]
        profile = arm_cfg["object_profiles"][label]
        self.record("APPROACH_GRID", grid, label)
        self.move_chassis(
            x=self.settings["motion"]["grid_approach_m"],
            label="Approach pickup grid {}".format(grid),
        )

        # Lower straight down from the search posture and grasp in one motion:
        # no raise-then-lower cycle, and no backward retreat afterward (delivery
        # turns and drives forward so we don't push already-placed objects).
        self.record("ARM_DESCEND", grid, label)
        self.move_arm(profile["pick_x_mm"], profile["pick_y_mm"], "Lower arm to {}".format(label))
        self.record("GRIP_CLOSE", grid, label)
        self.close_gripper(profile)
        self.record("ARM_LIFT", grid, label)
        self.move_arm(profile["pick_x_mm"], arm_cfg["safe_y_mm"], "Lift {}".format(label))

    def deliver_object(self, grid: int, label: str) -> None:
        route = self.settings["class_routes"][label]
        motion = self.settings["motion"]
        arm_cfg = self.settings["arm"]
        detect_y = self.current_world_y  # where the object was found; return here after
        bin_forward = motion["bin_forward_m"]

        # box -> left bin, bottle -> right bin: turn toward the bin and drive
        # FORWARD a fixed distance. We never back up before turning, so we don't
        # push objects already placed in front of the robot.
        if route == "left":
            turn_deg = 90.0
            turn_state = "TURN_LEFT"
        else:
            turn_deg = -90.0
            turn_state = "TURN_RIGHT"

        self.record(turn_state, grid, label)
        self.move_chassis(z=turn_deg, label="Turn toward {} bin".format(route))
        self.current_yaw_deg = turn_deg

        self.record("DRIVE_TO_BIN", grid, label, detail="forward={:.3f}m".format(bin_forward))
        self.move_chassis(x=bin_forward, label="Drive to {} bin".format(route))

        self.record("PLACE_DESCEND", grid, label)
        self.move_arm(arm_cfg["place_x_mm"], arm_cfg["safe_y_mm"], "Move above {} bin".format(route))
        self.move_arm(arm_cfg["place_x_mm"], arm_cfg["place_y_mm"], "Lower into {} bin".format(route))
        self.record("GRIP_OPEN", grid, label)
        self.open_gripper()
        self.record("PLACED", grid, label, result="success")

        self.move_arm(arm_cfg["place_x_mm"], arm_cfg["safe_y_mm"], "Lift after placement")

        # Return to the exact spot where the object was picked up: back out of
        # the bin, turn to face forward again, then undo the forward approach so
        # we stay on the scan line (parallel to the object line) instead of
        # creeping toward it a little more after every object.
        self.record("RETURN_SCAN", grid, label)
        self.move_chassis(x=-bin_forward, label="Back away from {} bin".format(route))
        self.move_chassis(z=-turn_deg, label="Restore forward heading")
        self.current_yaw_deg = 0.0
        self.move_chassis(
            x=-motion["grid_approach_m"],
            label="Return to scan line",
        )
        self.current_world_y = detect_y

    def process_grid(self, grid: int) -> bool:
        self.current_grid = grid
        target = self.sweep_to_grid_and_detect(grid)
        if target is None:
            self.record("EMPTY_GRID_SKIP", grid, result="skipped")
            self.return_scan_position_to_home(grid)
            return False

        label = str(target["label"])
        confidence = float(target["confidence"])
        self.record("DETECTED", grid, label, "{:.3f}".format(confidence))
        # Bring the object onto the center line before grasping, so the fixed arm
        # coordinates actually line up with it.
        self._center_once(target)
        # The iterative visual alignment is optional (off by default) because it
        # was losing the target mid-correction and aborting the whole run.
        if self.settings["vision"].get("enable_alignment", False):
            target = self.align_with_detection(grid, target)
        self.pick_object(grid, label)
        self.deliver_object(grid, label)
        self.record("OBJECT_COMPLETE", grid, label, "{:.3f}".format(confidence), "success")
        return True

    def run(self) -> None:
        try:
            self.record("WAIT_CAMERA", detail="waiting for first RoboMaster frame")
            self.vision.wait_for_first_frame(timeout=15.0)
            self.record("INITIALIZE_ROBOT")
            self.arm_home()
            self.open_gripper()

            sorted_count = 0
            empty_count = 0
            failed_count = 0
            for grid in range(1, self.settings["layout"]["grid_count"] + 1):
                self.check_stop()
                try:
                    if self.process_grid(grid):
                        sorted_count += 1
                    else:
                        empty_count += 1
                except InterruptedError:
                    raise
                except Exception as error:
                    # One bad grid must not kill the whole task: stop, log it,
                    # and move on to the next grid.
                    failed_count += 1
                    self.record("GRID_FAILED", grid, result="failure", detail=str(error))
                    try:
                        self.emergency_stop()
                    except Exception:
                        pass
                    self.current_world_y = 0.0
                    self.current_yaw_deg = 0.0
                    self.holding = False

            self.record(
                "TASK_COMPLETE",
                result="success" if failed_count == 0 else "partial",
                detail="sorted={} empty={} failed={} log={}".format(
                    sorted_count, empty_count, failed_count, self.log_path
                ),
            )
            self.success = True
        except InterruptedError as error:
            self.error = error
            self.record("TASK_STOPPED", self.current_grid, result="failure", detail=str(error))
            self.emergency_stop()
        except BaseException as error:
            self.error = error
            self.record("TASK_ABORTED", self.current_grid, result="failure", detail=str(error))
            self.emergency_stop()
        finally:
            self.done_event.set()
            self.log_stream.flush()

    def close_log(self) -> None:
        if not self.log_stream.closed:
            self.log_stream.flush()
            self.log_stream.close()


def print_plan(settings: Dict[str, Any]) -> None:
    layout = settings["layout"]
    motion = settings["motion"]
    print("\nRoboMaster EP YOLO sorting plan")
    print("  Model: {}".format(settings["_model_path"]))
    print("  Grids: 1..9, spacing {:.2f} m, home aligned with grid {}".format(
        layout["grid_spacing_m"], layout["home_grid_index"]
    ))
    print("  Camera: RoboMaster EP camera")
    print("  Detection: bottle -> {} bin, box -> {} bin".format(
        settings["class_routes"]["bottle"], settings["class_routes"]["box"]
    ))
    print("  Sequence: scan -> approach -> grasp -> retreat -> turn -> bin -> home")
    print("  Chassis speed: {:.2f} m/s; yaw speed: {:.1f} deg/s".format(
        motion["xy_speed_mps"], motion["yaw_speed_dps"]
    ))
    print("  OpenCV: boxes, labels, and confidence are shown in real time")
    print("  Emergency stop: Q/Esc in the preview, Ctrl+C, or robot power switch\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RoboMaster EP nine-grid bottle/box sorting with YOLO"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the RUN confirmation after the physical safety check",
    )
    args = parser.parse_args()

    try:
        settings = validate_config(load_json(args.config), args.config)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print("[CONFIG ERROR] {}".format(error), file=sys.stderr)
        return 2

    print_plan(settings)
    if not args.yes:
        print("Clear the robot path, secure loose cables, and keep the power switch reachable.")
        if input("Type RUN to connect and start the physical task: ").strip() != "RUN":
            print("Cancelled. The robot was not connected or moved.")
            return 1

    try:
        from robomaster import camera as rm_camera
        from robomaster import config as rm_config
        from robomaster import robot
        from ultralytics import YOLO
    except ImportError as error:
        print("[IMPORT ERROR] {}".format(error), file=sys.stderr)
        return 2

    print("[MODEL] Loading {}".format(settings["_model_path"]), flush=True)
    try:
        model = YOLO(str(settings["_model_path"]))
    except Exception as error:
        print("[MODEL ERROR] {}".format(error), file=sys.stderr)
        return 2

    model_names = model.names
    if isinstance(model_names, dict):
        available_names = set(str(value).strip().lower() for value in model_names.values())
    else:
        available_names = set(str(value).strip().lower() for value in model_names)
    missing_names = {"bottle", "box"} - available_names
    if missing_names:
        print(
            "[MODEL ERROR] Model classes are {}; missing {}".format(
                sorted(available_names), sorted(missing_names)
            ),
            file=sys.stderr,
        )
        return 2
    print("[MODEL] Classes: {}".format(sorted(available_names)), flush=True)

    local_ip = str(settings["connection"].get("local_ip", "")).strip()
    if local_ip:
        rm_config.LOCAL_IP_STR = local_ip

    ep_robot = robot.Robot()
    stop_event = threading.Event()
    done_event = threading.Event()
    vision = None  # type: Optional[VisionRuntime]
    sorter = None  # type: Optional[PhysicalSorter]
    control_thread = None  # type: Optional[threading.Thread]
    video_started = False

    try:
        print("[CONNECT] Connecting to RoboMaster EP...", flush=True)
        ep_robot.initialize(
            conn_type=settings["connection"].get("type", "sta"),
            proto_type=settings["connection"].get("protocol", "tcp"),
        )
        print("[CONNECT] Robot version: {}".format(ep_robot.get_version()), flush=True)

        resolution_name = str(settings["vision"].get("camera_resolution", "720p")).lower()
        resolution_map = {
            "360p": rm_camera.STREAM_360P,
            "540p": rm_camera.STREAM_540P,
            "720p": rm_camera.STREAM_720P,
        }
        if resolution_name not in resolution_map:
            raise ValueError("vision.camera_resolution must be 360p, 540p, or 720p")
        ep_robot.camera.start_video_stream(
            display=False,
            resolution=resolution_map[resolution_name],
        )
        video_started = True

        vision = VisionRuntime(ep_robot.camera, model, settings, stop_event)
        sorter = PhysicalSorter(ep_robot, vision, settings, stop_event, done_event)
        vision.set_emergency_callback(sorter.emergency_stop)
        control_thread = threading.Thread(
            target=sorter.run,
            name="robomaster-sorting-control",
            daemon=True,
        )
        control_thread.start()
        vision.run_display_loop(done_event)
        control_thread.join(timeout=5.0)

        if control_thread.is_alive():
            stop_event.set()
            sorter.emergency_stop()
            control_thread.join(timeout=3.0)
        if sorter.success:
            print("[DONE] Sorting task completed. Log: {}".format(sorter.log_path))
            return 0
        print("[FAILED] {}".format(sorter.error), file=sys.stderr)
        print("[LOG] {}".format(sorter.log_path), file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\n[EMERGENCY STOP] Ctrl+C received", file=sys.stderr)
        stop_event.set()
        if sorter is not None:
            sorter.emergency_stop()
        return 130
    except Exception as error:
        print("[FAILED] {}".format(error), file=sys.stderr)
        stop_event.set()
        if sorter is not None:
            sorter.emergency_stop()
        return 3
    finally:
        stop_event.set()
        if control_thread is not None and control_thread.is_alive():
            control_thread.join(timeout=2.0)
        if sorter is not None:
            sorter.emergency_stop()
            sorter.close_log()
        if video_started:
            try:
                ep_robot.camera.stop_video_stream()
            except Exception:
                pass
        try:
            ep_robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
