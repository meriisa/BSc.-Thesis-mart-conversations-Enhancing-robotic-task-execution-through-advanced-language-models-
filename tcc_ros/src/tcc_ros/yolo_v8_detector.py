#!/usr/bin/env python3

"""
Description:
YOLOv8-based perception module.

This module subscribes to YOLO detection results from /detection_result
and converts YOLO class IDs into COCO class names.
"""

import rospy
from vision_msgs.msg import Detection2DArray


YOLOV8_COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}


class YOLOv8Detector:

    def __init__(self):
        self.latest_detections = []

        rospy.Subscriber(
            "/detection_result",
            Detection2DArray,
            self.detection_callback
        )

        rospy.loginfo("[YOLOv8Detector] Initialized and subscribed to /detection_result")

    def detection_callback(self, msg):
        detections = []

        for detection in msg.detections:
            if not detection.results:
                continue

            try:
                result = detection.results[0]

                class_id = int(result.id)
                label = YOLOV8_COCO_NAMES.get(class_id, f"unknown class {class_id}")
                confidence = float(result.score)

                detections.append({
                    "class_id": class_id,
                    "label": label,
                    "confidence": confidence,
                    "source": "YOLOv8"
                })

                rospy.loginfo(
                    f"[YOLOv8Detector] {label} detected with confidence {confidence:.2f}"
                )

            except Exception as e:
                rospy.logwarn(f"[YOLOv8Detector] Failed to parse detection: {e}")

        self.latest_detections = detections

    def detect_objects(self):
        return self.latest_detections

    def get_detected_objects(self):
        labels = []

        for det in self.latest_detections:
            labels.append(det["label"])

        return labels

    def describe_detections(self):
        if not self.latest_detections:
            return "No objects detected by YOLOv8."

        response = []

        for det in self.latest_detections:
            label = det.get("label", "unknown")
            confidence = det.get("confidence", 0.0)

            response.append(
                f"{label}: detected with confidence {confidence * 100:.0f}%."
            )

        return "I can see with YOLOv8:\n" + "\n".join(response)