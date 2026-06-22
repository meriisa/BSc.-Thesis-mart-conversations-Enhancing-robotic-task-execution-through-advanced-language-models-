#!/usr/bin/env python3

import rospy
import csv
import time
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class YOLOv8Evaluator:

    def __init__(self):
        rospy.init_node("yolov8_evaluator")

        self.bridge = CvBridge()
        self.latest_image = None

        self.model = YOLO("yolov8m.pt")

        self.image_topic = "/rgb/image"
        self.output_file = "/root/yolov8m_results.csv"

        rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback
        )

        rospy.loginfo("[YOLOv8 Eval] Waiting for camera image...")

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

    def evaluate_once(self, test_name, expected_object):
        if self.latest_image is None:
            rospy.logwarn("[YOLOv8 Eval] No camera image received yet.")
            return

        expected_object = expected_object.lower().strip()
        open_vocab = expected_object not in COCO_CLASSES

        start_time = time.time()

        cv2.imwrite("/root/yolo_debug_image.jpg", self.latest_image)

        results = self.model(self.latest_image, verbose=False)

        inference_time_ms = (time.time() - start_time) * 1000

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.model.names[class_id].lower().strip()

                detections.append({
                    "label": label,
                    "confidence": confidence
                })

        print("ALL DETECTIONS:", detections)

        if detections:
            best_detection = max(detections, key=lambda x: x["confidence"])
            predicted_label = best_detection["label"]
            confidence = best_detection["confidence"]
        else:
            predicted_label = "none"
            confidence = 0.0

        correct = False

        for det in detections:
            if det["label"] == expected_object:
                correct = True
                predicted_label = det["label"]
                confidence = det["confidence"]
                break

        number_of_detections = len(detections)
        all_detected_labels = "; ".join([det["label"] for det in detections])

        if correct:
            error_type = "true_positive"
        else:
            if number_of_detections == 0:
                error_type = "false_negative"
            else:
                error_type = "false_positive"

        with open(self.output_file, "a", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow([
                test_name,
                expected_object,
                predicted_label,
                round(confidence, 3),
                round(inference_time_ms, 2),
                correct,
                error_type,
                number_of_detections,
                all_detected_labels,
                open_vocab
            ])

        rospy.loginfo(
            f"Test: {test_name} | "
            f"Expected: {expected_object} | "
            f"Predicted: {predicted_label} | "
            f"Confidence: {confidence:.2f} | "
            f"Time: {inference_time_ms:.2f} ms | "
            f"Correct: {correct} | "
            f"Error type: {error_type} | "
            f"Open vocab: {open_vocab}"
        )

    def run(self):
        with open(self.output_file, "w", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow([
                "test_name",
                "expected_object",
                "predicted_label",
                "confidence",
                "inference_time_ms",
                "correct",
                "error_type",
                "number_of_detections",
                "all_detected_labels",
                "open_vocab"
            ])

        rospy.loginfo("[YOLOv8 Eval] Ready.")
        rospy.loginfo("Format:")
        rospy.loginfo("test_name expected_object")
        rospy.loginfo("Examples:")
        rospy.loginfo("chair_test chair")
        rospy.loginfo("table_test dining table")
        rospy.loginfo("bookshelf_test bookshelf")
        rospy.loginfo("Type q to quit.")

        while not rospy.is_shutdown():
            user_input = input("\nEnter test case: ")

            if user_input.lower() == "q":
                break

            parts = user_input.split(maxsplit=1)

            if len(parts) < 2:
                print("Please enter: test_name expected_object")
                print("Example: chair_test chair")
                print("Example with two-word label: table_test dining table")
                continue

            test_name = parts[0]
            expected_object = parts[1]

            self.evaluate_once(
                test_name,
                expected_object
            )

        rospy.loginfo(f"[YOLOv8 Eval] Results saved to {self.output_file}")


if __name__ == "__main__":
    evaluator = YOLOv8Evaluator()
    evaluator.run()
