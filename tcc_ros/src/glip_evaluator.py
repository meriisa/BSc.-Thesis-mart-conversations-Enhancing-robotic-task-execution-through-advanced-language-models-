#!/usr/bin/env python3

import rospy
import csv
import time
import cv2
import torch

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo


class GLIPEvaluator:

    def __init__(self):
        # Initialize this script as a ROS node.
        rospy.init_node("glip_evaluator")

        # CvBridge converts ROS image messages into OpenCV images.
        self.bridge = CvBridge()

        # This variable stores the latest image received from the camera topic.
        self.latest_image = None

        # Parameters are used to keep paths and thresholds configurable.
        self.image_topic = rospy.get_param("~image_topic", "/rgb/image")
        self.output_file = rospy.get_param("~output_file", "/root/glip_results.csv")

        self.config_file = rospy.get_param(
            "~config_file",
            "/root/GLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
        )

        self.weight_file = rospy.get_param(
            "~weight_file",
            "/root/GLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
        )

        self.conf_threshold = rospy.get_param("~conf_threshold", 0.3)

        # GLIP is evaluated as an open-vocabulary phrase-grounding model.
        self.model_name = "GLIP"
        self.model_type = "open_vocabulary_grounding"

        # Select GPU if available, otherwise CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        rospy.loginfo("[GLIP Eval] Loading GLIP model...")
        rospy.loginfo(f"[GLIP Eval] Config: {self.config_file}")
        rospy.loginfo(f"[GLIP Eval] Weights: {self.weight_file}")
        rospy.loginfo(f"[GLIP Eval] Device: {self.device}")

        # Load GLIP configuration.
        cfg.local_rank = 0
        cfg.num_gpus = 1 if self.device == "cuda" else 0

        cfg.merge_from_file(self.config_file)
        cfg.merge_from_list([
            "MODEL.WEIGHT", self.weight_file,
            "MODEL.DEVICE", self.device
        ])
        cfg.freeze()

        # Create GLIP demo model.
        # confidence_threshold is also used internally by GLIPDemo.
        self.model = GLIPDemo(
            cfg,
            min_image_size=800,
            confidence_threshold=self.conf_threshold,
            show_mask_heatmaps=False
        )

        # Subscribe to the Gazebo RGB camera topic.
        rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback
        )

        rospy.loginfo("[GLIP Eval] Waiting for camera image...")

    def image_callback(self, msg):
        # Convert the ROS image message into an OpenCV image.
        # ROS provides an Image message, while GLIP expects a normal image array.
        self.latest_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

    def evaluate_once(self, test_name, expected_object):
        # The evaluation can only start after the first camera image was received.
        if self.latest_image is None:
            rospy.logwarn("[GLIP Eval] No camera image received yet.")
            return

        # Normalize input so that labels are handled consistently.
        expected_object = expected_object.lower().strip()

        # GLIP works with natural-language prompts.
        # For this evaluation, the target object is used directly as the prompt.
        prompt = expected_object + "."

        # Save the current image for documentation and later debugging.
        debug_image_path = f"/root/debug_glip_{test_name}.jpg"
        cv2.imwrite(debug_image_path, self.latest_image)

        # Convert image from BGR to RGB.
        # OpenCV uses BGR, while many vision models expect RGB images.
        image_rgb = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)

        # Start measuring inference time directly before model execution.
        start_time = time.time()

        # Run GLIP on the current image with the text prompt.
        predictions = self.model.compute_prediction(
            image_rgb,
            prompt
        )

        # Remove predictions below the confidence threshold.
        predictions = self.model._post_process(
            predictions,
            threshold=self.conf_threshold
        )

        # Calculate inference time in milliseconds.
        inference_time_ms = (time.time() - start_time) * 1000

        # Store detections in the same simple format as the YOLO evaluators.
        detections = []

        # If GLIP finds boxes for the given prompt, their scores are extracted.
        # Since the prompt contains only one object phrase, the label is the expected object.
        if len(predictions) > 0:
            scores = predictions.get_field("scores").tolist()

            for score in scores:
                detections.append({
                    "label": expected_object,
                    "confidence": float(score)
                })

        print("ALL DETECTIONS:", detections)

        # Default values if no object is found.
        predicted_label = "none"
        confidence = 0.0
        correct = False

        # If at least one detection exists, the strongest one is used.
        if len(detections) > 0:
            best_detection = max(detections, key=lambda x: x["confidence"])
            predicted_label = best_detection["label"]
            confidence = best_detection["confidence"]

            # Because GLIP only receives one target prompt in this evaluation,
            # any valid detection for that prompt is counted as correct.
            correct = True

        # Store additional information for later evaluation.
        number_of_detections = len(detections)
        all_detected_labels = "; ".join([det["label"] for det in detections])

        # Classify the result type.
        if correct:
            error_type = "true_positive"
        else:
            error_type = "false_negative"

        # Write this test result into the CSV file.
        # The header structure is aligned with the YOLOv8 and YOLO-World evaluators.
        with open(self.output_file, "a", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.model_name,
                self.model_type,
                test_name,
                expected_object,
                prompt,
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

        # Print the result in the ROS terminal.
        rospy.loginfo(
            f"Model: {self.model_name} | "
            f"Test: {test_name} | "
            f"Prompt: {prompt} | "
            f"Expected: {expected_object} | "
            f"Predicted: {predicted_label} | "
            f"Confidence: {confidence:.2f} | "
            f"Time: {inference_time_ms:.2f} ms | "
            f"Correct: {correct} | "
            f"Error: {error_type}"
        )

    def run(self):
        # Create the CSV file and write the header row.
        with open(self.output_file, "w", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name",
                "model_type",
                "test_name",
                "expected_object",
                "prompt",
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

        rospy.loginfo("[GLIP Eval] Ready.")
        rospy.loginfo("Format: test_name expected_object")
        rospy.loginfo("Example: chair_close chair")
        rospy.loginfo("Example: bookshelf_front bookshelf")
        rospy.loginfo("Example: trash_close trash bin")
        rospy.loginfo("Type q to quit.")

        # Manual evaluation loop.
        while not rospy.is_shutdown():
            user_input = input("\nEnter test case: ")

            if user_input.lower() == "q":
                break

            # maxsplit=1 allows object names with spaces, e.g. "trash bin".
            parts = user_input.split(maxsplit=1)

            if len(parts) < 2:
                print("Please enter: test_name expected_object")
                print("Example: trash_close trash bin")
                continue

            test_name = parts[0]
            expected_object = parts[1]

            self.evaluate_once(test_name, expected_object)

        rospy.loginfo(f"[GLIP Eval] Results saved to {self.output_file}")


if __name__ == "__main__":
    evaluator = GLIPEvaluator()
    evaluator.run()