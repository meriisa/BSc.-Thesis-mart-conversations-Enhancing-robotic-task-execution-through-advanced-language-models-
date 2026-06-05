#!/usr/bin/env python3

"""
object_grounding.py

My first module for object grounding in TCC.
29.06.2024

Purpose:
This module extracts object-based navigation goals from natural language commands.

Examples:
    "go to the red chair"
    "find the blue bottle"
    "look for a person"

The module does not perform object detection itself.

Instead, it prepares a structured object query that can later be
used by YOLO, CLIP and SAM.

Pipeline:

User Command
    ↓
Object Grounding
    ↓
YOLO Detection
    ↓
CLIP Verification
    ↓
SAM Segmentation
    ↓
Robot Action

Author:
Merisa Husovic
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ObjectQuery:
    """
    Structured representation of an object request.
    """

    action: str
    object_name: Optional[str]
    color: Optional[str]
    confidence: float


class ObjectGrounder:
    """
    Converts natural-language commands into structured object queries.
    """

    def __init__(self):

        self.actions = [
            "go",
            "navigate",
            "find",
            "search",
            "look for",
            "detect"
        ]

        self.objects = [
            "chair",
            "table",
            "person",
            "bottle",
            "cup",
            "door",
            "desk",
            "bench",
            "bookshelf"
        ]

        self.colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "orange"
        ]

    def ground(self, command: str) -> ObjectQuery:
        """
        Main function of the object grounder.

        It receives a natural-language command and extracts:
        - intended action
        - target object
        - visual attribute such as color
        - confidence score
        """

        text = command.lower()

        action = self._detect_action(text)
        object_name = self._detect_object(text)
        color = self._detect_color(text)

        confidence = self._calculate_confidence(action, object_name, color)

        return ObjectQuery(
            action=action,
            object_name=object_name,
            color=color,
            confidence=confidence
        )

    def _detect_action(self, text: str) -> str:
        """
        Detects what the robot should do.
        """

        if "go" in text or "navigate" in text:
            return "navigate"

        if "find" in text or "search" in text or "look for" in text or "detect" in text:
            return "find"

        return "unknown"

    def _detect_object(self, text: str) -> Optional[str]:
        """
        Detects the requested object class.
        """

        for obj in self.objects:
            if obj in text:
                return obj

        return None

    def _detect_color(self, text: str) -> Optional[str]:
        """
        Detects the requested object color.
        """

        for color in self.colors:
            if color in text:
                return color

        return None

    def _calculate_confidence(
        self,
        action: str,
        object_name: Optional[str],
        color: Optional[str]
    ) -> float:
        """
        Calculates a simple confidence score.

        This is not a neural network score.
        It is an explainable rule-based score.

        The score increases if:
        - an action is detected
        - an object is detected
        - a color is detected
        """

        score = 0.0

        if action != "unknown":
            score += 0.3

        if object_name is not None:
            score += 0.5

        if color is not None:
            score += 0.2

        return score      