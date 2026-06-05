
#!/usr/bin/env python3

"""
Description:
YOLOv8-based perception module.

This module subscribes to YOLO detection results and
stores the latest detected objects in a standardized format.
"""

import rospy

from vision_msgs.msg import Detection2DArray


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
        """
        Callback for YOLOv8 detections.

        The incoming ROS message contains detections from ultralytics_ros.
        This method converts them into a simple list of dictionaries.
        """

        detections = []

        for detection in msg.detections:
            if not detection.results:
                continue

            try:
                result = detection.results[0]

                label = str(result.id)
                detections.append({
                    "class_id": class_id,
                    "label": label,
                    "confidence": float(result.score),
                    "source": "YOLOv8"
                })


            except Exception as e:
                rospy.logwarn(f"[YOLOv8Detector] Failed to parse detection: {e}")

        self.latest_detections = detections

    def detect_objects(self):
        """
        Returns the latest YOLOv8 detections.

        This function has the same conceptual role as detect_objects()
        in the original SAM+CLIP perception module.
        """

        return self.latest_detections

    def get_detected_objects(self):
        """
        Returns only the labels of the detected objects.
        """

        labels = []

        for det in self.latest_detections:
            labels.append(det["label"])

        return labels

    def describe_detections(self):
        """
        Creates a human-readable description for ChatGUI.

        Example:
        I can see:
        trash bin 1 with confidence 49%.
        """

        if not self.latest_detections:
            return "No objects detected by YOLOv8."

        response = []

        for idx, det in enumerate(self.latest_detections):
            label = det.get("label", "unknown")
            confidence = det.get("confidence", 0.0)

            response.append(
                f"{label}: detected with confidence {confidence * 100:.0f}%."
            )

        return "I can see with YOLOv8:\n" + "\n".join(response)