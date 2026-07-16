#!/usr/bin/env python3
"""
SAM3 perception module for the TCC robot.

SAM3 performs text-prompted open-vocabulary segmentation.

The module provides two levels of perception:

1. Visual detections:
   SAM3 recognizes objects in the RGB image, even if no valid depth exists.

2. Object locations:
   Visual detections are combined with the depth image and transformed
   into the navigation frame for move_base.

Public interface:
    get_detected_objects(object_queries=None)
    get_object_locations(object_queries=None)
    get_object_pose(object_name)
    get_angle_to_object(object_name)
    send_latest_image()
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import message_filters
import numpy as np
import rospy
import tf

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from ultralytics.models.sam import SAM3SemanticPredictor


class PerceptionModule:
    def __init__(self, data_logger=None):
        self.data_logger = data_logger
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()

        self.rgb_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.last_image_time: Optional[rospy.Time] = None
        self.robot_pose = None

        self.fx = float(
            rospy.get_param(
                "perception/camera_fx",
                268.0
            )
        )

        self.fy = float(
            rospy.get_param(
                "perception/camera_fy",
                268.0
            )
        )

        self.cx = float(
            rospy.get_param(
                "perception/camera_cx",
                464.0
            )
        )

        self.cy = float(
            rospy.get_param(
                "perception/camera_cy",
                400.0
            )
        )

        self.camera_frame = rospy.get_param(
            "perception/camera_frame",
            "camera_rgbd_frame"
        )

        self.target_frame = rospy.get_param(
            "perception/navigation_frame",
            "map"
        )

        self.default_labels = self._normalise_queries(
            rospy.get_param(
                "perception/default_class_labels",
                [
                    "person",
                    "chair",
                    "table",
                    "door",
                    "cup",
                    "bottle",
                    "sofa",
                    "bookshelf",
                    "remote control",
                    "bench",
                    "desk",
                ]
            )
        )

        self.conf_threshold = float(
            rospy.get_param(
                "perception/detection_confidence_threshold",
                0.25
            )
        )

        self.max_depth_m = float(
            rospy.get_param(
                "perception/max_object_depth_m",
                8.0
            )
        )

        self.image_max_age_s = float(
            rospy.get_param(
                "perception/image_max_age_s",
                2.0
            )
        )

        self.tf_timeout_s = float(
            rospy.get_param(
                "perception/tf_timeout_s",
                1.0
            )
        )

        model_name = rospy.get_param(
            "models/sam3_checkpoint",
            "/root/catkin_ws/src/tcc_ros/src/sam3.pt"
        )

        rospy.loginfo(
            "[SAM3] Loading model: %s",
            model_name
        )

        overrides = {
            "model": model_name,
            "task": "segment",
            "mode": "predict",
            "conf": self.conf_threshold,
            "save": False,
            "verbose": False,
        }

        self.predictor = SAM3SemanticPredictor(
            overrides=overrides
        )

        self.image_publisher = rospy.Publisher(
            "/llm_image_output",
            Image,
            queue_size=10
        )

        self.latest_visual_detections: List[
            Dict[str, Any]
        ] = []

        self.latest_detections: List[
            Dict[str, Any]
        ] = []

        self._setup_subscribers()

        rospy.loginfo(
            "[SAM3] Ready. camera_frame=%s "
            "target_frame=%s labels=%s",
            self.camera_frame,
            self.target_frame,
            self.default_labels
        )

    def _setup_subscribers(self) -> None:
        rgb_topic = rospy.get_param(
            "topics/camera_color",
            "/rgb/image"
        )

        depth_topic = rospy.get_param(
            "topics/camera_depth",
            "/depth/image"
        )

        camera_info_topic = rospy.get_param(
            "topics/camera_info",
            "/rgb/camera_info"
        )

        odom_topic = rospy.get_param(
            "topics/odom",
            "/odom"
        )

        rgb_sub = message_filters.Subscriber(
            rgb_topic,
            Image
        )

        depth_sub = message_filters.Subscriber(
            depth_topic,
            Image
        )

        synchronizer = (
            message_filters.ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub],
                queue_size=10,
                slop=0.20
            )
        )

        synchronizer.registerCallback(
            self._image_callback
        )

        rospy.Subscriber(
            camera_info_topic,
            CameraInfo,
            self._camera_info_callback,
            queue_size=1
        )

        rospy.Subscriber(
            odom_topic,
            Odometry,
            self._odom_callback,
            queue_size=1
        )

        rospy.loginfo(
            "[SAM3] Subscribed to RGB=%s "
            "depth=%s camera_info=%s",
            rgb_topic,
            depth_topic,
            camera_info_topic
        )

    def _image_callback(
        self,
        rgb_msg: Image,
        depth_msg: Image
    ) -> None:
        try:
            self.rgb_image = (
                self.bridge.imgmsg_to_cv2(
                    rgb_msg,
                    desired_encoding="bgr8"
                )
            )

            self.depth_image = (
                self.bridge.imgmsg_to_cv2(
                    depth_msg,
                    desired_encoding="passthrough"
                )
            )

            self.last_image_time = rospy.Time.now()

            if rgb_msg.header.frame_id:
                self.camera_frame = (
                    rgb_msg.header.frame_id
                )

        except Exception as exc:
            rospy.logerr(
                "[SAM3] Failed to convert camera "
                "images: %s",
                exc
            )

    def _camera_info_callback(
        self,
        msg: CameraInfo
    ) -> None:
        try:
            if (
                len(msg.K) >= 6
                and msg.K[0] > 0
                and msg.K[4] > 0
            ):
                self.fx = float(msg.K[0])
                self.fy = float(msg.K[4])
                self.cx = float(msg.K[2])
                self.cy = float(msg.K[5])

        except Exception as exc:
            rospy.logwarn(
                "[SAM3] Could not read camera "
                "intrinsics: %s",
                exc
            )

    def _odom_callback(
        self,
        msg: Odometry
    ) -> None:
        self.robot_pose = msg.pose.pose

    @staticmethod
    def _normalise_label(
        label: str
    ) -> str:
        text = str(label).lower().strip()

        prefixes = (
            "detected ",
            "the ",
            "a ",
            "an "
        )

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text[
                        len(prefix):
                    ].strip()

                    changed = True

        return " ".join(text.split())

    def _normalise_queries(
        self,
        object_queries: Optional[
            Sequence[str]
        ]
    ) -> List[str]:
        if object_queries is None:
            return []

        if isinstance(object_queries, str):
            object_queries = [object_queries]

        queries: List[str] = []

        for query in object_queries:
            normalised = self._normalise_label(
                query
            )

            if (
                normalised
                and normalised not in queries
            ):
                queries.append(normalised)

        return queries

    def _images_are_ready(self) -> bool:
        if self.rgb_image is None:
            rospy.logwarn(
                "[SAM3] No RGB image available."
            )
            return False

        if self.depth_image is None:
            rospy.logwarn(
                "[SAM3] No depth image available."
            )
            return False

        if self.last_image_time is None:
            rospy.logwarn(
                "[SAM3] No image timestamp available."
            )
            return False

        age = (
            rospy.Time.now()
            - self.last_image_time
        ).to_sec()

        if age > self.image_max_age_s:
            rospy.logwarn(
                "[SAM3] Camera image is stale "
                "(%.2f s).",
                age
            )
            return False

        return True

    def _run_inference(
        self,
        queries: Sequence[str]
    ):
        image = self.rgb_image.copy()

        self.predictor.set_image(image)

        results = self.predictor(
            text=list(queries)
        )

        if isinstance(results, list):
            return results[0]

        return results

    @staticmethod
    def _box_to_mask(
        box_xyxy: Sequence[float],
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        height, width = image_shape

        x1, y1, x2, y2 = [
            int(round(value))
            for value in box_xyxy
        ]

        x1 = max(
            0,
            min(width - 1, x1)
        )

        x2 = max(
            0,
            min(width, x2)
        )

        y1 = max(
            0,
            min(height - 1, y1)
        )

        y2 = max(
            0,
            min(height, y2)
        )

        mask = np.zeros(
            (height, width),
            dtype=bool
        )

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True

        return mask

    def _extract_mask(
        self,
        result,
        detection_index: int,
        box_xyxy: Sequence[float]
    ) -> np.ndarray:
        height, width = (
            self.rgb_image.shape[:2]
        )

        try:
            if (
                result.masks is not None
                and result.masks.data is not None
            ):
                mask_tensor = (
                    result.masks.data[
                        detection_index
                    ]
                )

                mask = (
                    mask_tensor
                    .detach()
                    .cpu()
                    .numpy()
                )

                if mask.shape != (
                    height,
                    width
                ):
                    mask = cv2.resize(
                        mask.astype(
                            np.float32
                        ),
                        (width, height),
                        interpolation=(
                            cv2.INTER_NEAREST
                        )
                    )

                return mask > 0.5

        except Exception as exc:
            rospy.logwarn(
                "[SAM3] Could not extract mask "
                "%d; using box fallback: %s",
                detection_index,
                exc
            )

        return self._box_to_mask(
            box_xyxy,
            (height, width)
        )

    @staticmethod
    def _mask_centroid(
        mask: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        ys, xs = np.where(mask)

        if xs.size == 0:
            return None

        return (
            int(np.median(xs)),
            int(np.median(ys))
        )

    def _depth_to_metres(
        self,
        values: np.ndarray
    ) -> np.ndarray:
        values = values.astype(
            np.float32
        )

        if np.issubdtype(
            self.depth_image.dtype,
            np.integer
        ):
            values = values / 1000.0

        else:
            finite = values[
                np.isfinite(values)
                & (values > 0)
            ]

            if (
                finite.size
                and float(
                    np.median(finite)
                ) > 50.0
            ):
                values = values / 1000.0

        return values

    def _get_navigation_point(
        self,
        mask: np.ndarray,
        centroid: Tuple[int, int],
        bbox: Sequence[float]
    ) -> Optional[Tuple[int, int, float]]:
        """
        Finds a usable image pixel and its depth for navigation.

        Priority:
        1. Valid depth inside the SAM3 mask
        2. Valid depth directly below the object's bounding box
        3. Valid depth in progressively expanded regions around the object

        Returns:
            (u, v, depth_m)
        """
        depth = np.asarray(self.depth_image)

        if depth.ndim > 2:
            depth = depth[..., 0]

        image_height, image_width = depth.shape[:2]
        original_mask_shape = mask.shape

        if mask.shape != depth.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        depth_m = self._depth_to_metres(depth)

        valid_depth_map = (
            np.isfinite(depth_m)
            & (depth_m > 0.10)
            & (depth_m < self.max_depth_m)
        )

        # ---------------------------------------------------------
        # 1. Search directly inside the SAM3 mask
        # ---------------------------------------------------------
        valid_inside_mask = mask & valid_depth_map
        ys, xs = np.where(valid_inside_mask)

        rospy.loginfo(
            "[SAM3 Depth] mask_pixels=%d valid_inside_mask=%d",
            int(np.count_nonzero(mask)),
            int(xs.size)
        )

        if xs.size > 0:
            values = depth_m[ys, xs]
            median_depth = float(np.median(values))

            closest_indices = np.argsort(
                np.abs(values - median_depth)
            )

            selected_index = int(
                closest_indices[len(closest_indices) // 2]
            )

            u = int(xs[selected_index])
            v = int(ys[selected_index])
            selected_depth = float(depth_m[v, u])

            rospy.loginfo(
                "[SAM3 Depth] Using mask pixel "
                "u=%d v=%d depth=%.3f m",
                u,
                v,
                selected_depth
            )

            return u, v, selected_depth

        # ---------------------------------------------------------
        # 2. Search on the floor directly below the bounding box
        # ---------------------------------------------------------
        x1, y1, x2, y2 = [
            int(round(value))
            for value in bbox
        ]

        x1 = max(0, min(image_width - 1, x1))
        x2 = max(0, min(image_width, x2))
        y2 = max(0, min(image_height - 1, y2))

        horizontal_margin = max(
            20,
            int((x2 - x1) * 0.25)
        )

        floor_x1 = max(0, x1 - horizontal_margin)
        floor_x2 = min(
            image_width,
            x2 + horizontal_margin
        )

        floor_y1 = min(
            image_height - 1,
            y2 + 5
        )

        floor_y2 = min(
            image_height,
            y2 + 350
        )

        if (
            floor_x2 > floor_x1
            and floor_y2 > floor_y1
        ):
            floor_valid = valid_depth_map[
                floor_y1:floor_y2,
                floor_x1:floor_x2
            ]

            floor_ys, floor_xs = np.where(
                floor_valid
            )

            rospy.loginfo(
                "[SAM3 Depth] floor region valid=%d",
                int(floor_xs.size)
            )

            if floor_xs.size > 0:
                floor_xs = floor_xs + floor_x1
                floor_ys = floor_ys + floor_y1

                # Prefer a pixel horizontally close to the object center
                object_center_x = int((x1 + x2) / 2)

                distances = (
                    np.abs(floor_xs - object_center_x)
                    + 0.2 * np.abs(floor_ys - y2)
                )

                selected_index = int(
                    np.argmin(distances)
                )

                u = int(floor_xs[selected_index])
                v = int(floor_ys[selected_index])
                selected_depth = float(depth_m[v, u])

                rospy.loginfo(
                    "[SAM3 Depth] Using floor pixel "
                    "u=%d v=%d depth=%.3f m",
                    u,
                    v,
                    selected_depth
                )

                return u, v, selected_depth

        # ---------------------------------------------------------
        # 3. Expand around the complete object mask
        # ---------------------------------------------------------
        mask_uint8 = mask.astype(np.uint8)

        for radius in [15, 30, 60, 100, 150, 250]:
            kernel_size = radius * 2 + 1

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size)
            )

            expanded_mask = cv2.dilate(
                mask_uint8,
                kernel
            ).astype(bool)

            search_ring = (
                expanded_mask
                & ~mask
                & valid_depth_map
            )

            ys, xs = np.where(search_ring)

            rospy.loginfo(
                "[SAM3 Depth] expansion radius=%d valid=%d",
                radius,
                int(xs.size)
            )

            if xs.size > 0:
                center_u, center_v = centroid

                distances = (
                    (xs - center_u) ** 2
                    + (ys - center_v) ** 2
                )

                selected_index = int(
                    np.argmin(distances)
                )

                u = int(xs[selected_index])
                v = int(ys[selected_index])
                selected_depth = float(depth_m[v, u])

                rospy.loginfo(
                    "[SAM3 Depth] Using nearest valid pixel "
                    "u=%d v=%d depth=%.3f m",
                    u,
                    v,
                    selected_depth
                )

                return u, v, selected_depth

        rospy.logwarn(
            "[SAM3 Depth] No usable navigation depth "
            "was found for the object."
        )

        return None

    def _pixel_to_camera_point(
        self,
        u: int,
        v: int,
        depth_m: float
    ) -> Tuple[float, float, float]:
        x = (
            (float(u) - self.cx)
            * depth_m
            / self.fx
        )

        y = (
            (float(v) - self.cy)
            * depth_m
            / self.fy
        )

        z = depth_m

        return x, y, z

    def _transform_to_target(
        self,
        centroid: Tuple[int, int],
        depth_m: float
    ) -> Optional[PoseStamped]:
        try:
            (
                camera_x,
                camera_y,
                camera_z
            ) = self._pixel_to_camera_point(
                centroid[0],
                centroid[1],
                depth_m
            )

            point = PointStamped()

            point.header.frame_id = (
                self.camera_frame
            )

            point.header.stamp = (
                rospy.Time(0)
            )
            #_pixel_to_camera_point() liefert optische Kamerakoordinaten:
            #camera_x = rechts
            #camera_y = unten 
            #camera_z = nach vorne
            #
            #camera_rgbd_frame ist jedoch ein normaler ROS-Link-Frame:
            #x = nach vorne
            #y = nach links
            #z = nach oben
            
            point.point.x = camera_z
            point.point.y = -camera_x
            point.point.z = -camera_y

            self.tf_listener.waitForTransform(
                self.target_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(
                    self.tf_timeout_s
                )
            )

            transformed = (
                self.tf_listener.transformPoint(
                    self.target_frame,
                    point
                )
            )

            pose = PoseStamped()

            pose.header.frame_id = (
                self.target_frame
            )

            pose.header.stamp = (
                rospy.Time.now()
            )

            pose.pose.position = (
                transformed.point
            )

            pose.pose.orientation.w = 1.0

            return pose

        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException
        ) as exc:
            rospy.logwarn(
                "[SAM3] TF transform %s -> %s "
                "failed: %s",
                self.camera_frame,
                self.target_frame,
                exc
            )

            return None

        except Exception as exc:
            rospy.logerr(
                "[SAM3] 3D projection failed: %s",
                exc
            )

            return None

    def detect_visual_objects(
        self,
        object_queries: Optional[
            Sequence[str]
        ] = None
    ) -> List[Dict[str, Any]]:
        queries = self._normalise_queries(
            object_queries
        )

        if not queries:
            queries = list(
                self.default_labels
            )

        if not queries:
            rospy.logwarn(
                "[SAM3] No object queries "
                "configured."
            )

            self.latest_visual_detections = []

            return []

        if not self._images_are_ready():
            self.latest_visual_detections = []

            return []

        try:
            rospy.loginfo(
                "[SAM3] Searching for: %s",
                queries
            )

            result = self._run_inference(
                queries
            )

        except Exception as exc:
            rospy.logerr(
                "[SAM3] Inference failed: %s",
                exc
            )

            self.latest_visual_detections = []

            return []

        detections: List[
            Dict[str, Any]
        ] = []

        try:
            if (
                result.boxes is None
                or len(result.boxes) == 0
            ):
                rospy.loginfo(
                    "[SAM3] No matching objects "
                    "detected."
                )

                self.latest_visual_detections = []

                return []

            for index, box in enumerate(
                result.boxes
            ):
                confidence = float(
                    box.conf[0]
                )

                if (
                    confidence
                    < self.conf_threshold
                ):
                    continue

                class_id = int(
                    box.cls[0]
                )

                label = self._normalise_label(
                    str(
                        result.names[
                            class_id
                        ]
                    )
                )

                box_xyxy = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                mask = self._extract_mask(
                    result,
                    index,
                    box_xyxy
                )

                centroid = (
                    self._mask_centroid(
                        mask
                    )
                )

                if centroid is None:
                    rospy.logwarn(
                        "[SAM3] Empty mask for %s.",
                        label
                    )

                    continue

                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "mask": mask,
                        "bbox": box_xyxy,
                        "centroid": centroid,
                    }
                )

                rospy.loginfo(
                    "[SAM3 Visual] %s %.2f",
                    label,
                    confidence
                )

        except Exception as exc:
            rospy.logerr(
                "[SAM3] Could not parse visual "
                "inference result: %s",
                exc
            )

        self.latest_visual_detections = (
            detections
        )

        return detections

    def detect_objects(
        self,
        object_queries: Optional[
            Sequence[str]
        ] = None
    ) -> List[Dict[str, Any]]:
        visual_detections = (
            self.detect_visual_objects(
                object_queries
            )
        )

        detections: List[
            Dict[str, Any]
        ] = []

        for visual_detection in (
            visual_detections
        ):
            label = visual_detection[
                "label"
            ]

            confidence = visual_detection[
                "confidence"
            ]

            mask = visual_detection["mask"]

            centroid = visual_detection[
                "centroid"
            ]

            box_xyxy = visual_detection[
                "bbox"
            ]

            navigation_point = self._get_navigation_point(
                mask,
                centroid,
                box_xyxy
            )

            if navigation_point is None:
                rospy.logwarn(
                    "[SAM3] No navigation point for %s.",
                    label
                )
                continue

            navigation_u, navigation_v, depth_m = (
                navigation_point
            )

            pose = self._transform_to_target(
                (navigation_u, navigation_v),
                depth_m
            )

            if pose is None:
                rospy.logwarn(
                    "[SAM3] Could not transform "
                    "object '%s' to frame '%s'.",
                    label,
                    self.target_frame
                )

                continue

            detection = {
                "label": label,
                "confidence": confidence,
                "pose": pose,
                "mask": mask,
                "bbox": box_xyxy,
                "depth_m": depth_m,
            }

            detections.append(detection)

            rospy.loginfo(
                "[SAM3] %s %.2f at "
                "x=%.2f y=%.2f z=%.2f",
                label,
                confidence,
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z
            )

        self.latest_detections = (
            detections
        )

        return detections

    def get_object_locations(
        self,
        object_queries: Optional[
            Sequence[str]
        ] = None
    ) -> Dict[
        str,
        List[Dict[str, Any]]
    ]:
        detections = self.detect_objects(
            object_queries
        )

        locations: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

        for detection in detections:
            pose = detection["pose"]
            position = pose.pose.position

            locations.setdefault(
                detection["label"],
                []
            ).append(
                {
                    "x": float(
                        position.x
                    ),
                    "y": float(
                        position.y
                    ),
                    "z": float(
                        position.z
                    ),
                    "confidence": float(
                        detection[
                            "confidence"
                        ]
                    ),
                    "pose": pose,
                }
            )

        return locations

    def get_detected_objects(
        self,
        object_queries: Optional[
            Sequence[str]
        ] = None
    ) -> List[str]:
        """
        Return visually detected labels.

        A label is returned even when no valid depth
        or navigation pose is available.
        """
        detections = (
            self.detect_visual_objects(
                object_queries
            )
        )

        labels: List[str] = []

        for detection in detections:
            label = detection["label"]

            if label not in labels:
                labels.append(label)

        return labels

    def get_object_pose(
        self,
        object_name: str,
        prob_thresh: Optional[
            float
        ] = None
    ) -> Optional[PoseStamped]:
        threshold = (
            self.conf_threshold
            if prob_thresh is None
            else float(prob_thresh)
        )

        label = self._normalise_label(
            object_name
        )

        detections = self.detect_objects(
            [label]
        )

        candidates = [
            detection
            for detection in detections
            if (
                detection["label"] == label
                and detection[
                    "confidence"
                ] >= threshold
            )
        ]

        if not candidates:
            rospy.logwarn(
                "[SAM3] Object '%s' not found "
                "with a valid navigation pose.",
                object_name
            )

            return None

        best = max(
            candidates,
            key=lambda item: item[
                "confidence"
            ]
        )

        return best["pose"]

    def get_angle_to_object(
        self,
        object_name: str
    ) -> Optional[float]:
        pose = self.get_object_pose(
            object_name
        )

        if pose is None:
            return None

        if self.robot_pose is None:
            rospy.logwarn(
                "[SAM3] Robot odometry is "
                "unavailable."
            )

            return None

        robot_x = (
            self.robot_pose.position.x
        )

        robot_y = (
            self.robot_pose.position.y
        )

        q = self.robot_pose.orientation

        _, _, robot_yaw = (
            tf.transformations
            .euler_from_quaternion(
                [
                    q.x,
                    q.y,
                    q.z,
                    q.w
                ]
            )
        )

        target_angle = math.atan2(
            pose.pose.position.y - robot_y,
            pose.pose.position.x - robot_x
        )

        relative_angle = math.atan2(
            math.sin(
                target_angle - robot_yaw
            ),
            math.cos(
                target_angle - robot_yaw
            )
        )

        return math.degrees(
            relative_angle
        )

    def send_latest_image(self) -> None:
        if self.rgb_image is None:
            rospy.logwarn(
                "[SAM3] No RGB image available "
                "to publish."
            )

            return

        try:
            image_msg = (
                self.bridge.cv2_to_imgmsg(
                    self.rgb_image,
                    encoding="bgr8"
                )
            )

            self.image_publisher.publish(
                image_msg
            )

        except Exception as exc:
            rospy.logerr(
                "[SAM3] Could not publish "
                "image: %s",
                exc
            )