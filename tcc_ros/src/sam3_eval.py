#!/usr/bin/env python3

import time
import csv
import cv2
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics.models.sam import SAM3SemanticPredictor


class SAM3Evaluator:

    def __init__(self):
        rospy.init_node("sam3_evaluator")

        self.bridge = CvBridge()
        self.latest_image = None

        self.model_name = rospy.get_param(
            "~model_name",
            "/root/catkin_ws/src/tcc_ros/src/sam3.pt"
        )
        self.image_topic = rospy.get_param("~image_topic", "/rgb/image")
        self.output_file = rospy.get_param("~output_file", "/root/sam3_results.csv")
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.25)

        self.model_type = "open_vocabulary_segmentation"

        rospy.loginfo(f"[SAM3 Eval] Loading model: {self.model_name}")

        overrides = dict(
            model=self.model_name,
            task="segment",
            mode="predict",
            conf=self.conf_threshold,
            save=False,
            verbose=False,
        )

        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1
        )

        rospy.loginfo("[SAM3 Eval] Waiting for camera image...")

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

    def evaluate_once(self, test_name, expected_object):
        if self.latest_image is None:
            rospy.logwarn("[SAM3 Eval] No camera image received yet.")
            return

        expected_object = expected_object.lower().strip()

        debug_image_path = f"/root/debug_sam3_{test_name}.jpg"
        cv2.imwrite(debug_image_path, self.latest_image)

        img = self.latest_image.copy()

        start_time = time.time()

        try:
            self.predictor.set_image(img)
            results = self.predictor(text=[expected_object])
        except Exception as e:
            rospy.logerr(f"SAM3 inference failed: {e}")
            return

        inference_time_ms = (time.time() - start_time) * 1000

        result = results[0] if isinstance(results, list) else results

        mask_count = 0
        number_of_detections = 0
        predicted_label = "none"
        confidence = 0.0
        all_detected_labels = ""

        try:
            if result.masks is not None:
                mask_count = len(result.masks)

            detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    label = str(result.names[class_id]).lower().strip()

                    detections.append({
                        "label": label,
                        "confidence": conf
                    })

            number_of_detections = len(detections)

            if number_of_detections > 0:
                best_detection = max(detections, key=lambda x: x["confidence"])
                predicted_label = best_detection["label"]
                confidence = best_detection["confidence"]
                all_detected_labels = "; ".join([det["label"] for det in detections])

        except Exception as e:
            rospy.logwarn(f"Could not parse SAM3 result: {e}")

        if number_of_detections == 0:
            predicted_label = "none"
            correct = False
            error_type = "false_negative"

        elif predicted_label == expected_object:
            correct = True
            error_type = "true_positive"

        else:
            correct = False
            error_type = "false_positive"

        with open(self.output_file, "a", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.model_name,
                self.model_type,
                test_name,
                expected_object,
                predicted_label,
                round(confidence, 3),
                round(inference_time_ms, 2),
                correct,
                error_type,
                number_of_detections,
                mask_count,
                all_detected_labels,
                self.conf_threshold,
                debug_image_path
            ])

        rospy.loginfo(
            f"Model: SAM3 | "
            f"Test: {test_name} | "
            f"Expected: {expected_object} | "
            f"Predicted: {predicted_label} | "
            f"Confidence: {confidence:.2f} | "
            f"Masks: {mask_count} | "
            f"Time: {inference_time_ms:.2f} ms | "
            f"Correct: {correct} | "
            f"Error: {error_type}"
        )

    def run(self):
        with open(self.output_file, "w", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name",
                "model_type",
                "test_name",
                "expected_object",
                "predicted_label",
                "confidence",
                "inference_time_ms",
                "correct",
                "error_type",
                "number_of_detections",
                "mask_count",
                "all_detected_labels",
                "conf_threshold",
                "debug_image_path"
            ])

        rospy.loginfo("[SAM3 Eval] Ready.")
        rospy.loginfo("Format: test_name expected_object")
        rospy.loginfo("Example: person_front person")
        rospy.loginfo("Example: trash_close trash bin")
        rospy.loginfo("Type q to quit.")

        while not rospy.is_shutdown():
            user_input = input("\nEnter test case: ").strip()

            if user_input.lower() == "q":
                break

            parts = user_input.split(maxsplit=1)

            if len(parts) < 2:
                print("Please enter: test_name expected_object")
                print("Example: trash_close trash bin")
                continue

            test_name = parts[0]
            expected_object = parts[1]

            self.evaluate_once(test_name, expected_object)

        rospy.loginfo(f"[SAM3 Eval] Results saved to {self.output_file}")


if __name__ == "__main__":
    evaluator = SAM3Evaluator()
    evaluator.run()