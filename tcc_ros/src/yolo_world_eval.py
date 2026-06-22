#!/usr/bin/env python3

import rospy
import csv
import time
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLOWorld


class YOLOWorldEvaluator:

    def __init__(self):
        # Initialize this script as a ROS node.
        rospy.init_node("yolo_world_evaluator")

        # CvBridge converts ROS image messages into OpenCV images.
        self.bridge = CvBridge()

        # This variable stores the latest image received from the camera topic.
        self.latest_image = None

        # Parameters are used to keep the script flexible.
        # This allows testing small and medium models without changing the code.
        self.model_name = rospy.get_param("~model_name", "yolov8m-world.pt")
        self.image_topic = rospy.get_param("~image_topic", "/rgb/image")
        self.output_file = rospy.get_param("~output_file", "/root/yolo_world_results.csv")
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.05)

        # YOLO-World is evaluated as an open-vocabulary detector.
        # The searched class is provided dynamically during each test case.
        self.model_type = "open_vocabulary_detection"

        # Load the selected YOLO-World model.
        rospy.loginfo(f"[YOLO-World Eval] Loading model: {self.model_name}")
        self.model = YOLOWorld(self.model_name)

        # Subscribe to the camera image topic.
        # Each new image updates self.latest_image.
        rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback
        )

        rospy.loginfo("[YOLO-World Eval] Waiting for camera image...")

    def image_callback(self, msg):
        # Convert the ROS image message into an OpenCV image.
        # The detector works with OpenCV image arrays.
        self.latest_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

    def evaluate_once(self, test_name, expected_object):
        # The model cannot be evaluated before the first camera image arrives.
        if self.latest_image is None:
            rospy.logwarn("[YOLO-World Eval] No camera image received yet.")
            return

        # Normalize user input.
        # This prevents problems caused by capital letters or spaces.
        expected_object = expected_object.lower().strip()

        # Set the searched class dynamically.
        # This simulates the real robot use case:
        # a user command defines the target object.
        self.model.set_classes([expected_object])

        # Save the current image for documentation and later checking.
        debug_image_path = f"/root/debug_{self.model_name}_{test_name}.jpg"
        cv2.imwrite(debug_image_path, self.latest_image)

        # Start time measurement directly before inference.
        start_time = time.time()

        # Run YOLO-World on the latest camera image.
        # The model searches only for the dynamically defined target class.
        results = self.model.predict(
            self.latest_image,
            conf=self.conf_threshold,
            verbose=False
        )

        # Calculate inference time in milliseconds.
        inference_time_ms = (time.time() - start_time) * 1000

        # Store all detections in a simple list.
        detections = []

        # Extract the detected labels and confidence values.
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

        # Default values if no object is detected.
        predicted_label = "none"
        confidence = 0.0
        correct = False

        # Since YOLO-World is only asked to search for the expected object,
        # the best detection is used as the prediction.
        if len(detections) > 0:
            best_detection = max(detections, key=lambda x: x["confidence"])
            predicted_label = best_detection["label"]
            confidence = best_detection["confidence"]

            # The result is correct if the predicted label matches the target object.
            if predicted_label == expected_object:
                correct = True

        # Store additional information for later analysis.
        number_of_detections = len(detections)
        all_detected_labels = "; ".join([det["label"] for det in detections])

        # Classify the result type.
        if correct:
            error_type = "true_positive"
        elif number_of_detections == 0:
            error_type = "false_negative"
        else:
            error_type = "false_positive"

        # Write this test result into the CSV file.
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

        # Print the test result in the ROS terminal.
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
        # Create the result CSV file and write the header row.
        # The structure is kept equal to the YOLOv8 evaluator.
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

        rospy.loginfo("[YOLO-World Eval] Ready.")
        rospy.loginfo("Format: test_name expected_object")
        rospy.loginfo("Example: bookshelf_front bookshelf")
        rospy.loginfo("Example: trash_close trash bin")
        rospy.loginfo("Type q to quit.")

        # Manual evaluation loop.
        # The user enters one test case at a time.
        while not rospy.is_shutdown():
            user_input = input("\nEnter test case: ")

            # End the evaluation.
            if user_input.lower() == "q":
                break

            # Split into test name and expected object.
            # maxsplit=1 is important for labels with spaces, e.g. "trash bin".
            parts = user_input.split(maxsplit=1)

            if len(parts) < 2:
                print("Please enter: test_name expected_object")
                print("Example: trash_close trash bin")
                continue

            test_name = parts[0]
            expected_object = parts[1]

            self.evaluate_once(test_name, expected_object)

        rospy.loginfo(f"[YOLO-World Eval] Results saved to {self.output_file}")


if __name__ == "__main__":
    evaluator = YOLOWorldEvaluator()
    evaluator.run()