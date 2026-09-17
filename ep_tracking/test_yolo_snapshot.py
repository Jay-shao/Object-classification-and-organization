# -*- coding: utf-8 -*-
"""
对 test_camera.py 存下来的 snapshot 跑 YOLO —— 跑在 Python 3.12。

单独看：真机摄像头拍到的画面里，YOLO 到底能不能认出物体、类别对不对、有几个。

用法（先在第 3.8 里跑 test_camera.py 按 s 存几张图）：
    python test_yolo_snapshot.py snapshots/
    python test_yolo_snapshot.py snapshots/ --conf 0.3
"""
import argparse
import glob
import os

import cv2

from vision import YoloDetector, DEFAULT_WEIGHT

CLASS_COLORS = {"bottle": (0, 255, 0), "box": (255, 0, 0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default="snapshots")
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    detector = YoloDetector(weight=DEFAULT_WEIGHT, conf=args.conf)
    files = sorted(glob.glob(os.path.join(args.folder, "*.jpg")))
    if not files:
        print(f"'{args.folder}' 里没有 .jpg。请先在 3.8 里跑 test_camera.py，按 s 存图。")
        return

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        dets = detector.detect(img)
        for d in dets:
            color = CLASS_COLORS.get(d.name, (0, 255, 255))
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
            cv2.putText(img, f"{d.name} {d.confidence:.2f}",
                        (int(d.x1), max(14, int(d.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow(os.path.basename(f), img)
        summary = ", ".join(f"{d.name}({d.confidence:.2f})" for d in dets) or "无"
        print(f"{os.path.basename(f)}: 检测到 {len(dets)} 个 -> {summary}")
        print("  按任意键看下一张，q 退出")
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
