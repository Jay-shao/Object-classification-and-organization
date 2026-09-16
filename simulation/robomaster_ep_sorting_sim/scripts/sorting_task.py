#!/usr/bin/env python3
"""Nine-grid RoboMaster EP sorting state machine for Gazebo Fortress."""

import csv
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint
from vision_msgs.msg import Detection2DArray


WORLD_NAME = 'sorting_world'
ROBOT_NAME = 'robomaster_ep'

GRID_Y = {
    1: -1.68,
    2: -1.26,
    3: -0.84,
    4: -0.42,
    5: 0.00,
    6: 0.42,
    7: 0.84,
    8: 1.26,
    9: 1.68,
}

HOME_BASE = (0.45, 0.00, 0.0)
GRID_LANE_X = 0.45
GRID_APPROACH_X = 0.97
BIN_APPROACH_Y = 1.80

# Arm Cartesian targets are expressed in the RoboMaster base frame.
HOME_ARM = (0.23, 0.18)
PICK_ARM = (0.23, 0.055)
PLACE_ARM = (0.30, 0.070)
SAFE_Z = 0.18
OPEN = 0.0
CLOSED = 0.38
OBJECT_GRIPPER_Z_OFFSET = -0.015
MIN_CONFIDENCE = 0.60

# Geometry copied from robomaster_description/urdf/arm.urdf.xacro.
ARM_BASE_WORLD_Z = 0.03465 + 0.0906477
A = (0.0103961, 0.0307410)
B = (0.0018704, 0.1210238)
C = (0.1058557, -0.0561093)
GRIPPER_T = (0.0002793 + 0.124, 0.0001815 - 0.039)
FIXED = (A[0] + GRIPPER_T[0], A[1] + GRIPPER_T[1])
L1 = math.hypot(*B)
L2 = math.hypot(*C)
BETA1 = math.atan2(B[1], B[0])
BETA2 = math.atan2(C[1], C[0])
ARM_LIMITS = ((-0.274, 1.384), (-0.79936, 1.73137), (-0.34732, 1.21475))

ARM_JOINTS = [
    'arm_1_joint', 'rod_joint', 'rod_3_joint', 'arm_2_joint',
    'endpoint_bracket_joint', 'rod_1_joint', 'rod_2_joint',
    'triangle_joint',
]
GRIPPER_JOINTS = [
    'gripper_m_joint',
    'left_gripper_joint_1', 'left_gripper_joint_2',
    'left_gripper_joint_4', 'left_gripper_joint_5',
    'left_gripper_joint_6', 'left_gripper_joint_7',
    'right_gripper_joint_1', 'right_gripper_joint_2',
    'right_gripper_joint_4', 'right_gripper_joint_5',
    'right_gripper_joint_6', 'right_gripper_joint_7',
]
GRIPPER_CLOSED = [
    -0.022918,
    0.687846, -1.764646, -0.019682, 1.088293, -0.933733, -0.306504,
    -0.703549, 1.786527, -0.003254, -1.072585, 0.908795, 0.298343,
]


class SortingTask(Node):
    def __init__(self):
        super().__init__('sorting_task')
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/ep_arm_controller/follow_joint_trajectory',
        )
        self.state_publisher = self.create_publisher(
            String,
            '/sorting/task_state',
            10,
        )
        self.detection_subscription = self.create_subscription(
            Detection2DArray,
            '/sorting/detections',
            self.detection_callback,
            10,
        )
        self.latest_detections = None
        self.base_pose = HOME_BASE
        self.arm_xyz = HOME_ARM
        self.arm_q = self.solve_ik(*HOME_ARM, reference=None)
        self.closure = OPEN
        self.holding_object = None
        workspace = Path(
            os.environ.get(
                'ROBOMASTER_EP_WS',
                str(Path.home() / 'robomaster_ep_ws'),
            )
        )
        log_dir = workspace / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f'sorting_task_{datetime.now():%Y%m%d_%H%M%S}.csv'
        self.log_stream = self.log_path.open('w', newline='', encoding='utf-8')
        self.log_writer = csv.writer(self.log_stream)
        self.log_writer.writerow(
            ['timestamp', 'state', 'grid', 'class', 'result', 'detail']
        )
        self.log_stream.flush()
        self.vector_fallback_announced = False

    def destroy_node(self):
        if hasattr(self, 'log_stream') and not self.log_stream.closed:
            self.log_stream.flush()
            self.log_stream.close()
        super().destroy_node()

    def detection_callback(self, message):
        self.latest_detections = message

    def record(self, state, grid='', class_id='', result='info', detail=''):
        text = f'{state}'
        if grid != '':
            text += f' grid={grid}'
        if class_id:
            text += f' class={class_id}'
        if detail:
            text += f' {detail}'
        self.get_logger().info(text)
        self.state_publisher.publish(String(data=text))
        self.log_writer.writerow(
            [datetime.now().isoformat(), state, grid, class_id, result, detail]
        )
        self.log_stream.flush()

    @staticmethod
    def detection_center_x(detection):
        center = detection.bbox.center
        if hasattr(center, 'position'):
            return float(center.position.x)
        return float(center.x)

    def parse_detections(self):
        parsed = {}
        for detection in self.latest_detections.detections:
            if not detection.results:
                continue
            detection_id = getattr(detection, 'id', '')
            if detection_id.startswith('grid_'):
                grid = int(detection_id.split('_', 1)[1])
            else:
                center_x = self.detection_center_x(detection)
                grid = int(round((center_x - 100.0) / 120.0)) + 1
            if grid not in GRID_Y:
                continue
            hypothesis = detection.results[0].hypothesis
            parsed[grid] = (
                str(hypothesis.class_id),
                float(hypothesis.score),
            )
        return parsed

    def wait_until_ready(self):
        self.record('WAIT_CONTROLLER', detail='waiting for arm action server')
        if not self.arm_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError('ep_arm_controller action server not available')

        self.record('WAIT_DETECTIONS', detail='waiting for Detection2DArray')
        deadline = time.monotonic() + 30.0
        while rclpy.ok() and self.latest_detections is None:
            if time.monotonic() >= deadline:
                raise RuntimeError('/sorting/detections timed out')
            rclpy.spin_once(self, timeout_sec=0.1)

    def solve_ik(self, x, z, reference):
        px = x - FIXED[0]
        pz = (z - ARM_BASE_WORLD_Z) - FIXED[1]
        radius = math.hypot(px, pz)
        if radius < 1e-9 or radius > L1 + L2 or radius < abs(L1 - L2):
            raise ValueError(f'unreachable arm target x={x:.3f}, z={z:.3f}')

        cosine = (radius * radius + L1 * L1 - L2 * L2) / (
            2.0 * radius * L1
        )
        alpha = math.acos(max(-1.0, min(1.0, cosine)))
        phi = math.atan2(pz, px)
        candidates = []
        for theta1 in (phi + alpha, phi - alpha):
            bx = L1 * math.cos(theta1)
            bz = L1 * math.sin(theta1)
            theta2 = math.atan2(pz - bz, px - bx)
            arm_1 = BETA1 - theta1
            rod = BETA2 - theta2
            rod_3 = arm_1 - rod
            measured = (arm_1, rod, rod_3)
            if all(
                low <= value <= high
                for value, (low, high) in zip(measured, ARM_LIMITS)
            ):
                candidates.append(measured)
        if not candidates:
            raise ValueError(f'arm target violates joint limits x={x:.3f}, z={z:.3f}')
        if reference is None:
            return min(candidates, key=lambda q: sum(value * value for value in q))
        return min(
            candidates,
            key=lambda q: sum((value - old) ** 2 for value, old in zip(q, reference)),
        )

    @staticmethod
    def arm_positions(measured):
        arm_1, rod, rod_3 = measured
        return [arm_1, rod, rod_3, -rod_3, -rod, rod, arm_1, -arm_1]

    @staticmethod
    def gripper_positions(closure):
        if not OPEN <= closure <= CLOSED:
            raise ValueError(f'unsafe gripper closure: {closure}')
        return [value * closure for value in GRIPPER_CLOSED]

    @staticmethod
    def set_duration(message, seconds):
        whole = int(seconds)
        message.sec = whole
        message.nanosec = int((seconds - whole) * 1_000_000_000)

    def send_arm_path(self, cartesian_points, closure, seconds):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS + GRIPPER_JOINTS
        reference = self.arm_q
        final_q = reference
        count = len(cartesian_points)
        for index, (x, z) in enumerate(cartesian_points, start=1):
            final_q = self.solve_ik(x, z, reference)
            reference = final_q
            point = JointTrajectoryPoint()
            point.positions = (
                self.arm_positions(final_q) + self.gripper_positions(closure)
            )
            self.set_duration(point.time_from_start, seconds * index / count)
            goal.trajectory.points.append(point)

        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if not handle or not handle.accepted:
            raise RuntimeError('arm trajectory goal rejected')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()
        if wrapped_result is None or wrapped_result.result.error_code != 0:
            raise RuntimeError('arm trajectory execution failed')

        self.arm_xyz = cartesian_points[-1]
        self.arm_q = final_q
        self.closure = closure

    def line_arm_to(self, x, z, closure=None, seconds=1.0, samples=8):
        if closure is None:
            closure = self.closure
        if self.holding_object is not None:
            # Gazebo Fortress user-command services become unreliable when
            # called much faster than about 5 Hz.  Four samples keep the held
            # object visually attached without flooding set_pose.
            samples = min(samples, 4)
        start_x, start_z = self.arm_xyz
        path = [
            (
                start_x + (x - start_x) * step / samples,
                start_z + (z - start_z) * step / samples,
            )
            for step in range(1, samples + 1)
        ]
        if self.holding_object is None:
            self.send_arm_path(path, closure, seconds)
            return

        segment_seconds = seconds / samples
        for point_x, point_z in path:
            self.send_arm_path([(point_x, point_z)], closure, segment_seconds)
            object_x, object_y = self.world_from_base(point_x, 0.0)
            self.set_entity_pose(
                self.holding_object,
                object_x,
                object_y,
                point_z + OBJECT_GRIPPER_Z_OFFSET,
                self.base_pose[2],
            )

    def set_gripper(self, closure, seconds=0.8):
        self.send_arm_path([self.arm_xyz], closure, seconds)

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def world_from_base(self, local_x, local_y):
        base_x, base_y, yaw = self.base_pose
        world_x = base_x + math.cos(yaw) * local_x - math.sin(yaw) * local_y
        world_y = base_y + math.sin(yaw) * local_x + math.cos(yaw) * local_y
        return world_x, world_y

    @staticmethod
    def pose_fields(name, x, y, z, yaw):
        half = yaw / 2.0
        return (
            f'name: "{name}" '
            f'position: {{x: {x:.6f}, y: {y:.6f}, z: {z:.6f}}} '
            'orientation: {'
            f'x: 0.0, y: 0.0, z: {math.sin(half):.8f}, '
            f'w: {math.cos(half):.8f}'
            '}'
        )

    @staticmethod
    def call_ign_service(service, request_type, request, timeout_ms=5000):
        last_detail = ''
        for attempt in range(1, 3):
            try:
                result = subprocess.run(
                    [
                        'ign', 'service', '-s', service,
                        '--reqtype', request_type,
                        '--reptype', 'ignition.msgs.Boolean',
                        '--timeout', str(timeout_ms),
                        '--req', request,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=(timeout_ms / 1000.0) + 3.0,
                    check=False,
                )
                success = (
                    result.returncode == 0 and 'data: true' in result.stdout
                )
                last_detail = (result.stdout + ' ' + result.stderr).strip()
                if success:
                    return True, last_detail
            except subprocess.TimeoutExpired:
                last_detail = f'process timeout on attempt {attempt}'
            time.sleep(0.20)
        return False, last_detail

    def set_entity_pose(self, name, x, y, z, yaw):
        request = self.pose_fields(name, x, y, z, yaw)
        success, detail = self.call_ign_service(
            f'/world/{WORLD_NAME}/set_pose',
            'ignition.msgs.Pose',
            request,
        )
        if not success:
            raise RuntimeError(f'failed to set pose for {name}: {detail}')

    def set_robot_and_held_object(self, x, y, yaw):
        object_x = x + math.cos(yaw) * self.arm_xyz[0]
        object_y = y + math.sin(yaw) * self.arm_xyz[0]
        robot_pose = self.pose_fields(ROBOT_NAME, x, y, 0.0, yaw)
        object_pose = self.pose_fields(
            self.holding_object,
            object_x,
            object_y,
            self.arm_xyz[1] + OBJECT_GRIPPER_Z_OFFSET,
            yaw,
        )
        vector_request = f'pose {{ {robot_pose} }} pose {{ {object_pose} }}'
        success, detail = self.call_ign_service(
            f'/world/{WORLD_NAME}/set_pose_vector',
            'ignition.msgs.Pose_V',
            vector_request,
        )
        if success:
            return

        if not self.vector_fallback_announced:
            self.get_logger().warning(
                'set_pose_vector unavailable; falling back to two set_pose calls: '
                + detail
            )
            self.vector_fallback_announced = True
        self.set_entity_pose(ROBOT_NAME, x, y, 0.0, yaw)
        self.set_entity_pose(
            self.holding_object,
            object_x,
            object_y,
            self.arm_xyz[1] + OBJECT_GRIPPER_Z_OFFSET,
            yaw,
        )

    def drive_to(self, target_x, target_y, target_yaw, seconds=1.2, samples=8):
        start_x, start_y, start_yaw = self.base_pose
        yaw_delta = self.normalize_angle(target_yaw - start_yaw)
        distance = math.hypot(target_x - start_x, target_y - start_y)
        # Limit commanded visual speed, then keep user-command service calls
        # below roughly 4 Hz.  Long cross-field runs therefore stay smooth
        # without flooding Gazebo with hundreds of subprocess requests.
        seconds = max(seconds, distance / 0.65, abs(yaw_delta) / 0.80)
        samples = max(2, min(14, int(seconds / 0.26)))
        for step in range(1, samples + 1):
            ratio = step / samples
            x = start_x + (target_x - start_x) * ratio
            y = start_y + (target_y - start_y) * ratio
            yaw = self.normalize_angle(start_yaw + yaw_delta * ratio)
            if self.holding_object is None:
                self.set_entity_pose(ROBOT_NAME, x, y, 0.0, yaw)
            else:
                self.set_robot_and_held_object(x, y, yaw)
            self.base_pose = (x, y, yaw)
            time.sleep(seconds / samples)

    def grip_object(self, grid):
        self.set_gripper(CLOSED)
        object_name = f'object_grid_{grid}'
        object_x, object_y = self.world_from_base(PICK_ARM[0], 0.0)
        self.set_entity_pose(
            object_name,
            object_x,
            object_y,
            PICK_ARM[1] + OBJECT_GRIPPER_Z_OFFSET,
            self.base_pose[2],
        )
        self.holding_object = object_name

    def release_object(self, class_id):
        object_name = self.holding_object
        self.set_gripper(OPEN)
        object_x, object_y = self.world_from_base(PLACE_ARM[0], 0.0)
        object_z = 0.055 if class_id == 'class_1' else 0.060
        self.set_entity_pose(object_name, object_x, object_y, object_z, 0.0)
        self.holding_object = None
        time.sleep(0.3)

    def recover_home(self, grid, class_id, error):
        self.record('RECOVERY', grid, class_id, 'failure', str(error))
        self.holding_object = None
        try:
            self.set_gripper(OPEN, seconds=0.5)
            self.line_arm_to(*HOME_ARM, closure=OPEN, seconds=0.8)
            self.drive_to(*HOME_BASE, seconds=1.0)
        except Exception as recovery_error:
            self.record(
                'SAFE_STOP', grid, class_id, 'failure', str(recovery_error)
            )

    def process_object(self, grid, class_id):
        grid_y = GRID_Y[grid]
        object_name = f'object_grid_{grid}'
        self.record('TARGET_SELECTED', grid, class_id, detail=object_name)

        # The lateral lane is 75 cm away from the object line.  The robot
        # first aligns with the grid, then approaches straight ahead.
        self.record('ALIGN_GRID', grid, class_id)
        self.drive_to(GRID_LANE_X, grid_y, 0.0, seconds=1.2)
        self.record('APPROACH_GRID', grid, class_id)
        self.drive_to(GRID_APPROACH_X, grid_y, 0.0, seconds=1.0)

        self.record('ARM_DESCEND', grid, class_id)
        self.line_arm_to(PICK_ARM[0], SAFE_Z, seconds=0.6)
        self.line_arm_to(*PICK_ARM, seconds=0.9)
        self.record('GRIP_CLOSE', grid, class_id)
        self.grip_object(grid)
        self.record('ARM_LIFT', grid, class_id)
        self.line_arm_to(PICK_ARM[0], SAFE_Z, seconds=0.9)

        # Required retreat before either turn.
        self.record('RETREAT', grid, class_id)
        self.drive_to(GRID_LANE_X, grid_y, 0.0, seconds=1.0)

        if class_id == 'class_1':
            turn_yaw = math.pi / 2.0
            bin_y = BIN_APPROACH_Y
            turn_name = 'TURN_LEFT'
        else:
            turn_yaw = -math.pi / 2.0
            bin_y = -BIN_APPROACH_Y
            turn_name = 'TURN_RIGHT'

        self.record(turn_name, grid, class_id)
        self.drive_to(GRID_LANE_X, grid_y, turn_yaw, seconds=0.8)
        self.record('DRIVE_TO_BIN', grid, class_id)
        self.drive_to(GRID_LANE_X, bin_y, turn_yaw, seconds=1.4)

        self.record('PLACE_DESCEND', grid, class_id)
        self.line_arm_to(PLACE_ARM[0], SAFE_Z, seconds=0.6)
        self.line_arm_to(*PLACE_ARM, seconds=0.9)
        self.record('GRIP_OPEN', grid, class_id)
        self.release_object(class_id)
        self.record('PLACED', grid, class_id, 'success')

        self.line_arm_to(PLACE_ARM[0], SAFE_Z, seconds=0.8)
        self.line_arm_to(*HOME_ARM, seconds=0.8)
        self.record('RETURN_HOME', grid, class_id)
        self.drive_to(GRID_LANE_X, 0.0, turn_yaw, seconds=1.3)
        self.drive_to(*HOME_BASE, seconds=0.8)
        self.record('OBJECT_COMPLETE', grid, class_id, 'success')

    def run(self):
        self.wait_until_ready()
        detections = self.parse_detections()
        self.record(
            'DETECTION_READY',
            detail=f'{len(detections)} observations received',
        )
        successes = 0
        failures = 0

        for grid in range(1, 10):
            observation = detections.get(grid)
            if observation is None:
                self.record('EMPTY_GRID_SKIP', grid, result='skipped')
                continue

            class_id, confidence = observation
            self.record(
                'DETECTED',
                grid,
                class_id,
                detail=f'confidence={confidence:.2f}',
            )
            if confidence < MIN_CONFIDENCE or class_id not in {'class_1', 'class_2'}:
                self.record(
                    'UNRECOGNIZED_SKIP',
                    grid,
                    class_id,
                    'skipped',
                    f'confidence={confidence:.2f}',
                )
                continue

            try:
                self.process_object(grid, class_id)
                successes += 1
            except Exception as error:
                failures += 1
                self.recover_home(grid, class_id, error)

        self.record(
            'TASK_COMPLETE',
            result='success' if successes >= 5 else 'failure',
            detail=f'sorted={successes}/6 failures={failures}; log={self.log_path}',
        )
        if successes < 5:
            raise RuntimeError(f'acceptance failed: only {successes}/6 sorted')


def main():
    rclpy.init()
    node = SortingTask()
    try:
        node.run()
    except KeyboardInterrupt:
        node.record('INTERRUPTED', result='failure', detail='operator Ctrl+C')
    except Exception as error:
        node.record('TASK_ABORTED', result='failure', detail=str(error))
        raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
