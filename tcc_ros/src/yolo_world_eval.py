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

        # Stores the latest image received from the camera topic.
        self.latest_image = None

        # Parameters.
        self.model_name = rospy.get_param(
            "~model_name",
            "yolov8m-world.pt"
        )

        self.image_topic = rospy.get_param(
            "~image_topic",
            "/rgb/image"
        )

        self.output_file = rospy.get_param(
            "~output_file",
            "/root/yolo_world_results.csv"
        )

        self.conf_threshold = rospy.get_param(
            "~conf_threshold",
            0.05
        )

        # YOLO-World is evaluated as an open-vocabulary detector.
        self.model_type = "open_vocabulary_detection"

        # Load YOLO-World.
        rospy.loginfo(
            f"[YOLO-World Eval] Loading model: {self.model_name}"
        )

        self.model = YOLOWorld(self.model_name)

        # Explicitly move the model to the NVIDIA GPU.
        self.model.to("cuda:0")

        # Remember the currently configured text class.
        # This prevents set_classes() from being called repeatedly
        # for consecutive tests of the same object class.
        self.current_class = None

        # Subscribe to the simulated RGB camera.
        rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1
        )

        rospy.loginfo(
            "[YOLO-World Eval] Waiting for camera image..."
        )

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format.
        self.latest_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

    def evaluate_once(self, test_name, expected_object):

        # Evaluation requires a camera image.
        if self.latest_image is None:
            rospy.logwarn(
                "[YOLO-World Eval] No camera image received yet."
            )
            return

        # Normalize object name.
        expected_object = expected_object.lower().strip()

        # Only update the YOLO-World text prompt if the requested
        # object class has changed.
        if self.current_class != expected_object:
            self.model.set_classes([expected_object])
            self.current_class = expected_object

        # Save the unannotated input image for documentation.
        debug_image_path = (
            f"/root/debug_{self.model_name}_{test_name}.jpg"
        )

        cv2.imwrite(
            debug_image_path,
            self.latest_image
        )

        # ---------------------------------------------------------
        # GPU INFERENCE
        # ---------------------------------------------------------

        # Start timing immediately before model inference.
        start_time = time.time()

        results = self.model.predict(
            self.latest_image,
            conf=self.conf_threshold,
            device=0,
            verbose=False
        )

        # Stop timing immediately after inference.
        # Plotting and saving the annotated image are therefore
        # not included in the measured inference time.
        inference_time_ms = (
            time.time() - start_time
        ) * 1000

        # ---------------------------------------------------------
        # SAVE ANNOTATED RESULT
        # ---------------------------------------------------------

        annotated_image = results[0].plot()

        annotated_image_path = (
            f"/root/yolo_world_"
            f"{self.model_name}_{test_name}.jpg"
        )

        cv2.imwrite(
            annotated_image_path,
            annotated_image
        )

        rospy.loginfo(
            f"Annotated image saved to: "
            f"{annotated_image_path}"
        )

        # ---------------------------------------------------------
        # PROCESS DETECTIONS
        # ---------------------------------------------------------

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                label = (
                    self.model.names[class_id]
                    .lower()
                    .strip()
                )

                detections.append({
                    "label": label,
                    "confidence": confidence
                })

        print("ALL DETECTIONS:", detections)

        # Default result if nothing was detected.
        predicted_label = "none"
        confidence = 0.0
        correct = False

        # Use the detection with the highest confidence.
        if len(detections) > 0:

            best_detection = max(
                detections,
                key=lambda x: x["confidence"]
            )

            predicted_label = best_detection["label"]
            confidence = best_detection["confidence"]

            if predicted_label == expected_object:
                correct = True

        number_of_detections = len(detections)

        all_detected_labels = "; ".join(
            [det["label"] for det in detections]
        )

        # Determine evaluation outcome.
        if correct:
            error_type = "true_positive"

        elif number_of_detections == 0:
            error_type = "false_negative"

        else:
            error_type = "false_positive"

        # ---------------------------------------------------------
        # WRITE RESULT TO CSV
        # ---------------------------------------------------------

        with open(
            self.output_file,
            "a",
            newline="",
            encoding="utf-8",
            errors="replace"
        ) as f:

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

        # Print result.
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

        # Create a fresh CSV file for this evaluation run.
        with open(
            self.output_file,
            "w",
            newline="",
            encoding="utf-8",
            errors="replace"
        ) as f:

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

        rospy.loginfo(
            "[YOLO-World Eval] Ready."
        )

        rospy.loginfo(
            "GPU device: cuda:0"
        )

        rospy.loginfo(
            "Format: test_name expected_object"
        )

        rospy.loginfo(
            "Example: person_right_side person"
        )

        rospy.loginfo(
            "Example: trash_front_close trash bin"
        )

        rospy.loginfo(
            "Type q to quit."
        )

        # Manual evaluation loop.
        while not rospy.is_shutdown():

            user_input = input(
                "\nEnter test case: "
            ).strip()

            if user_input.lower() == "q":
                break

            # maxsplit=1 allows object names containing spaces,
            # e.g. "trash bin".
            parts = user_input.split(maxsplit=1)

            if len(parts) < 2:

                print(
                    "Please enter: "
                    "test_name expected_object"
                )

                print(
                    "Example: "
                    "trash_front_close trash bin"
                )

                continue

            test_name = parts[0]
            expected_object = parts[1]

            self.evaluate_once(
                test_name,
                expected_object
            )

        rospy.loginfo(
            f"[YOLO-World Eval] "
            f"Results saved to {self.output_file}"
        )


if __name__ == "__main__":

    evaluator = YOLOWorldEvaluator()
    evaluator.run()