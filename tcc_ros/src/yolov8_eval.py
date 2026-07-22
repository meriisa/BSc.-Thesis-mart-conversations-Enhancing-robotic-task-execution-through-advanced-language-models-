#!/usr/bin/env python3

import rospy
import csv
import time
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


class YOLOv8Evaluator:

    def __init__(self):
        # Initialize this script as a ROS node.
        rospy.init_node("yolov8_evaluator")

        # CvBridge is needed to convert ROS image messages into OpenCV images.
        self.bridge = CvBridge()

        # This variable always stores the latest camera image received from ROS.
        self.latest_image = None

        # Parameters are used so that different YOLOv8 models can be evaluated
        # without changing the source code.
        self.model_name = rospy.get_param("~model_name", "yolov8m.pt")
        self.image_topic = rospy.get_param("~image_topic", "/rgb/image")
        self.output_file = rospy.get_param("~output_file", "/root/yolov8_results.csv")
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.25)

        # YOLOv8 is treated as a closed-set detector because it is limited
        # to its predefined training classes.
        self.model_type = "closed_set_detection"

        # Load the selected YOLOv8 model.
        rospy.loginfo(f"[YOLOv8 Eval] Loading model: {self.model_name}")
        self.model = YOLO(self.model_name)

        # Subscribe to the RGB camera topic from Gazebo.
        # Every time a new image is published, image_callback() is called.
        rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback
        )

        rospy.loginfo("[YOLOv8 Eval] Waiting for camera image...")

    def image_callback(self, msg):
        # Convert the ROS image message into an OpenCV image.
        # YOLO expects an image array, not a ROS message.
        self.latest_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

    def evaluate_once(self, test_name, expected_object):
        # If no camera image has been received yet, the test cannot be executed.
        if self.latest_image is None:
            rospy.logwarn("[YOLOv8 Eval] No camera image received yet.")
            return

        # Normalize the expected object label.
        # This avoids problems with capital letters or unnecessary spaces.
        expected_object = expected_object.lower().strip()

        # Save the current camera image for documentation/debugging.
        # This makes it possible to check later what the robot actually saw.
        debug_image_path = f"/root/debug_{self.model_name}_{test_name}.jpg"
        cv2.imwrite(debug_image_path, self.latest_image)

        # Start time measurement directly before the model prediction.
        start_time = time.time()

        # Run YOLOv8 on the current camera image.
        # The confidence threshold removes detections with very low confidence.
        results = self.model.predict(
            self.latest_image,
            conf=self.conf_threshold,
            verbose=False
        )

        # Calculate the inference time in milliseconds.
        inference_time_ms = (time.time() - start_time) * 1000

        # Store all detected objects in a simple list.
        detections = []

        # Extract labels and confidence values from the YOLO result.
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Convert the class ID to the actual object name.
                label = self.model.names[class_id].lower().strip()

                detections.append({
                    "label": label,
                    "confidence": confidence
                })

        print("ALL DETECTIONS:", detections)

        # Default values if no correct object is found.
        predicted_label = "none"
        confidence = 0.0
        correct = False

        # The test is counted as correct if the expected object appears
        # anywhere in the detections, not only as the strongest detection.
        for det in detections:
            if det["label"] == expected_object:
                predicted_label = det["label"]
                confidence = det["confidence"]
                correct = True
                break

        # If the expected object was not found, but YOLO detected something else,
        # the strongest detection is stored as the predicted label.
        if not correct and len(detections) > 0:
            best_detection = max(detections, key=lambda x: x["confidence"])
            predicted_label = best_detection["label"]
            confidence = best_detection["confidence"]

        # Store additional information about the detection result.
        number_of_detections = len(detections)
        all_detected_labels = "; ".join([det["label"] for det in detections])

        # Classify the result for later evaluation.
        if correct:
            error_type = "true_positive"
        elif number_of_detections == 0:
            error_type = "false_negative"
        else:
            error_type = "false_positive"

        # Write the result of this test case into the CSV file.
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
                all_detected_labels,
                self.conf_threshold,
                debug_image_path
            ])

        # Print the result also in the ROS terminal.
        rospy.loginfo(
            f"Model: {self.model_name} | "
            f"Test: {test_name} | "
            f"Expected: {expected_object} | "
            f"Predicted: {predicted_label} | "
            f"Confidence: {confidence:.2f} | "
            f"Time: {inference_time_ms:.2f} ms | "
            f"Correct: {correct} | "
            f"Error: {error_type}"
        )

    def run(self):
        # Create the CSV file and write the header row.
        # This makes all evaluation files directly comparable.
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
                "all_detected_labels",
                "conf_threshold",
                "debug_image_path"
            ])

        rospy.loginfo("[YOLOv8 Eval] Ready.")
        rospy.loginfo("Format: test_name expected_object")
        rospy.loginfo("Example: chair_close chair")
        rospy.loginfo("Type q to quit.")

        # Manual input loop for the evaluation.
        # Each entered line corresponds to one test case.
        while not rospy.is_shutdown():
            user_input = input("\nEnter test case: ")

            # Stop the evaluation.
            if user_input.lower() == "q":
                break

            # Split input into test name and expected object.
            # maxsplit=1 allows object names with spaces, e.g. "dining table".
            parts = user_input.split(maxsplit=1)

            if len(parts) < 2:
                print("Please enter: test_name expected_object")
                print("Example: chair_close chair")
                continue

            test_name = parts[0]
            expected_object = parts[1]

            self.evaluate_once(test_name, expected_object)

        rospy.loginfo(f"[YOLOv8 Eval] Results saved to {self.output_file}")


if __name__ == "__main__":
    evaluator = YOLOv8Evaluator()
    evaluator.run()