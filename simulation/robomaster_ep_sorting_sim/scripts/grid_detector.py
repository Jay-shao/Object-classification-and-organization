#!/usr/bin/env python3
"""Deterministic first-version detector with the required ROS 2 interfaces.

The node waits for the simulated overhead RGB stream, then publishes one
Detection2DArray for the six physical objects.  It also publishes one
low-confidence unknown detection over physically empty grid 6 so the task
controller exercises its second exception path.  Replace this node with the
trained detector later without changing the task controller interface.
"""

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


GRID_IMAGE_X = {index: 100.0 + (index - 1) * 120.0 for index in range(1, 10)}
PHYSICAL_OBJECTS = {
    1: ('class_1', 0.98),
    2: ('class_2', 0.97),
    4: ('class_1', 0.96),
    5: ('class_2', 0.98),
    7: ('class_1', 0.97),
    9: ('class_2', 0.96),
}


class GridDetector(Node):
    def __init__(self):
        super().__init__('grid_detector')
        self.publisher = self.create_publisher(
            Detection2DArray,
            '/sorting/detections',
            10,
        )
        self.subscription = self.create_subscription(
            Image,
            '/sorting_camera/image',
            self.image_callback,
            10,
        )
        self.image_received = False
        self.fallback_announced = False
        self.started_at = time.monotonic()
        self.timer = self.create_timer(1.0, self.publish_detections)

    def image_callback(self, message):
        del message
        if not self.image_received:
            self.get_logger().info(
                'Received /sorting_camera/image; publishing grid detections.'
            )
        self.image_received = True

    @staticmethod
    def set_bbox_center(detection, x, y):
        center = detection.bbox.center
        if hasattr(center, 'position'):
            center.position.x = x
            center.position.y = y
        else:
            center.x = x
            center.y = y

    def make_detection(self, grid, class_id, score):
        detection = Detection2D()
        detection.header.stamp = self.get_clock().now().to_msg()
        detection.header.frame_id = 'sorting_camera'
        if hasattr(detection, 'id'):
            detection.id = f'grid_{grid}'
        self.set_bbox_center(detection, GRID_IMAGE_X[grid], 360.0)
        detection.bbox.size_x = 62.0
        detection.bbox.size_y = 62.0

        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = class_id
        result.hypothesis.score = float(score)
        detection.results.append(result)
        return detection

    def publish_detections(self):
        if not self.image_received and time.monotonic() - self.started_at < 5.0:
            return
        if not self.image_received and not self.fallback_announced:
            self.get_logger().warning(
                'Camera image has not arrived after 5 s; using the deterministic '
                'first-version detections so the full task can still be tested.'
            )
            self.fallback_announced = True

        message = Detection2DArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'sorting_camera'
        for grid, (class_id, score) in PHYSICAL_OBJECTS.items():
            message.detections.append(
                self.make_detection(grid, class_id, score)
            )

        # Grid 6 remains physically empty.  This deliberately weak unknown
        # observation verifies that the task skips low-confidence detections.
        message.detections.append(self.make_detection(6, 'unknown', 0.25))
        self.publisher.publish(message)


def main():
    rclpy.init()
    node = GridDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
