```python
#!/usr/bin/env python3

import os
import csv
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import rospy

from tcc_ros.llm_interface import LLMInterface
from tcc_ros.commands_parser import CommandParser


class LLMEvaluator:
    """
    Evaluates the existing language-understanding pipeline:

        User command
            -> LLMInterface
            -> natural-language action output
            -> CommandParser
            -> structured robot actions

    The resulting actions are evaluated without executing them on the robot.
    """

    def __init__(self):
        rospy.init_node("llm_evaluator")

        self.input_file = rospy.get_param(
            "~input_file",
            "/root/llm_test_commands.csv"
        )

        self.output_file = rospy.get_param(
            "~output_file",
            "/root/llm_results.csv"
        )

        # API key: first use environment variable, then config.yaml parameter.
        self.api_key = (
            os.getenv("OPENAI_API_KEY")
            or rospy.get_param("models/llm_api_key", "")
        )

        # Use the same destinations as the real robotic system.
        self.destinations = rospy.get_param("destinations", {})

        if not self.api_key:
            rospy.logwarn(
                "[LLM Eval] No API key found in OPENAI_API_KEY "
                "or models/llm_api_key."
            )

        # Use the original project classes.
        self.llm_interface = LLMInterface(
            api_key=self.api_key,
            destinations=self.destinations
        )

        # The parser only needs the ActionExecutor during physical execution.
        # For parsing and evaluation, None is sufficient.
        self.command_parser = CommandParser(action_executor=None)

        self.model_name = self.llm_interface.MODEL_NAME

        rospy.loginfo("[LLM Eval] Initialized.")
        rospy.loginfo("[LLM Eval] Input file: %s", self.input_file)
        rospy.loginfo("[LLM Eval] Output file: %s", self.output_file)
        rospy.loginfo("[LLM Eval] Model: %s", self.model_name)

    @staticmethod
    def normalize_text(value: Any) -> str:
        """
        Normalize text values for comparisons.
        """
        if value is None:
            return ""

        return " ".join(str(value).lower().strip().split())

    @staticmethod
    def normalize_number(value: Any) -> Any:
        """
        Convert numeric values to floats so that, for example,
        2 and 2.0 are considered equal.
        """
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value

        return value

    def normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a parsed action before comparing it to the expected action.
        """
        normalized = {}

        for key, value in action.items():
            normalized_key = self.normalize_text(key)

            if isinstance(value, dict):
                normalized[normalized_key] = {
                    self.normalize_text(inner_key):
                    self.normalize_number(inner_value)
                    for inner_key, inner_value in value.items()
                }

            elif isinstance(value, str):
                normalized[normalized_key] = self.normalize_text(value)

            else:
                normalized[normalized_key] = self.normalize_number(value)

        if "action" in normalized:
            normalized["action"] = str(normalized["action"]).upper()

        return normalized

    def normalize_actions(
        self,
        actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return [
            self.normalize_action(action)
            for action in actions
            if isinstance(action, dict)
        ]

    def read_expected_actions(
        self,
        row: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Read expected actions from the CSV.

        Preferred format:
            expected_actions_json

        Example:
            [{"action":"FORWARD","distance":2.0}]

        A simple fallback format is also supported:
            expected_action
            expected_target
        """
        expected_json = row.get("expected_actions_json", "").strip()

        if expected_json:
            try:
                expected_actions = json.loads(expected_json)

                if isinstance(expected_actions, dict):
                    expected_actions = [expected_actions]

                if not isinstance(expected_actions, list):
                    raise ValueError(
                        "expected_actions_json must contain a list or object."
                    )

                return self.normalize_actions(expected_actions)

            except Exception as exc:
                raise ValueError(
                    "Invalid expected_actions_json in test "
                    f"'{row.get('test_name', 'unknown')}': {exc}"
                )

        expected_action = row.get("expected_action", "").strip()

        if not expected_action:
            return []

        action = {
            "action": expected_action
        }

        expected_target = row.get("expected_target", "").strip()

        if expected_target:
            normalized_action = expected_action.upper()

            if normalized_action == "MOVE_TO_OBJECT":
                action["object_name"] = expected_target

            elif normalized_action == "ROTATE_TO_FACE":
                action["object_name"] = expected_target

            elif normalized_action == "NAVIGATE_TO_DESTINATION":
                action["destination_name"] = expected_target

        return self.normalize_actions([action])

    @staticmethod
    def numbers_equal(
        first: Any,
        second: Any,
        tolerance: float = 0.001
    ) -> bool:
        """
        Compare numerical values with a small tolerance.
        """
        if isinstance(first, (int, float)) and isinstance(
            second,
            (int, float)
        ):
            return abs(float(first) - float(second)) <= tolerance

        return first == second

    def dictionaries_match(
        self,
        predicted: Dict[str, Any],
        expected: Dict[str, Any]
    ) -> bool:
        """
        Check whether a predicted action contains all expected values.

        Additional predicted fields are allowed. This prevents a test from
        failing only because the parser added a default value such as speed.
        """
        for key, expected_value in expected.items():
            if key not in predicted:
                return False

            predicted_value = predicted[key]

            if isinstance(expected_value, dict):
                if not isinstance(predicted_value, dict):
                    return False

                if not self.dictionaries_match(
                    predicted_value,
                    expected_value
                ):
                    return False

            elif not self.numbers_equal(
                predicted_value,
                expected_value
            ):
                return False

        return True

    def compare_actions(
        self,
        predicted_actions: List[Dict[str, Any]],
        expected_actions: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Compare action count, order, action types and expected parameters.
        """
        if not predicted_actions:
            return False, "no_actions_parsed"

        if len(predicted_actions) != len(expected_actions):
            return False, "wrong_action_count"

        for index, expected_action in enumerate(expected_actions):
            predicted_action = predicted_actions[index]

            predicted_type = predicted_action.get("action", "")
            expected_type = expected_action.get("action", "")

            if predicted_type != expected_type:
                return False, "wrong_action"

            if not self.dictionaries_match(
                predicted_action,
                expected_action
            ):
                return False, "wrong_parameters"

        return True, "correct"

    def run_pipeline(
        self,
        user_command: str
    ) -> Tuple[str, List[Dict[str, Any]], float, str]:
        """
        Run the same LLM interface and parser used by the robot.

        No actions are executed.
        """
        start_time = time.time()

        llm_result = self.llm_interface.process_input(
            user_command,
            current_yaw="0.0",
            cardinal_direction="north",
            position_x="0.0",
            position_y="0.0",
            position_z="0.0"
        )

        latency_ms = (time.time() - start_time) * 1000.0

        raw_response = str(llm_result.get("content", ""))

        parsed_result = self.command_parser.parse_input(llm_result)

        result_type = parsed_result.get("type", "UNKNOWN")

        if result_type == "ACTIONS":
            actions = parsed_result.get("content", [])
        else:
            actions = []

        normalized_actions = self.normalize_actions(actions)

        return (
            raw_response,
            normalized_actions,
            latency_ms,
            result_type
        )

    def evaluate_test(
        self,
        row: Dict[str, str]
    ) -> Dict[str, Any]:
        test_name = row.get("test_name", "unknown")
        user_command = row.get("user_command", "").strip()

        expected_actions = self.read_expected_actions(row)

        try:
            (
                raw_response,
                predicted_actions,
                latency_ms,
                result_type
            ) = self.run_pipeline(user_command)

            parser_success = (
                result_type == "ACTIONS"
                and len(predicted_actions) > 0
            )

            response_correct, error_type = self.compare_actions(
                predicted_actions,
                expected_actions
            )

            return {
                "model_name": self.model_name,
                "predicted_actions": predicted_actions,
                "expected_actions": expected_actions,
                "latency_ms": round(latency_ms, 2),
                "parser_success": parser_success,
                "response_correct": response_correct,
                "error_type": error_type,
                "raw_response": raw_response.replace("\n", " ")
            }

        except Exception as exc:
            rospy.logerr(
                "[LLM Eval] Test '%s' failed: %s",
                test_name,
                exc
            )

            return {
                "model_name": self.model_name,
                "predicted_actions": [],
                "expected_actions": expected_actions,
                "latency_ms": 0.0,
                "parser_success": False,
                "response_correct": False,
                "error_type": "pipeline_failed",
                "raw_response": str(exc).replace("\n", " ")
            }

    def write_header(self):
        with open(
            self.output_file,
            "w",
            newline="",
            encoding="utf-8",
            errors="replace"
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "model_name",
                "test_name",
                "category",
                "user_command",
                "expected_actions",
                "predicted_actions",
                "latency_ms",
                "parser_success",
                "response_correct",
                "error_type",
                "raw_response"
            ])

    def write_result(
        self,
        row: Dict[str, str],
        result: Dict[str, Any]
    ):
        with open(
            self.output_file,
            "a",
            newline="",
            encoding="utf-8",
            errors="replace"
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                result["model_name"],
                row.get("test_name", ""),
                row.get("category", ""),
                row.get("user_command", ""),
                json.dumps(
                    result["expected_actions"],
                    ensure_ascii=False
                ),
                json.dumps(
                    result["predicted_actions"],
                    ensure_ascii=False
                ),
                result["latency_ms"],
                result["parser_success"],
                result["response_correct"],
                result["error_type"],
                result["raw_response"]
            ])

    def run(self):
        if not os.path.exists(self.input_file):
            rospy.logerr(
                "[LLM Eval] Input file not found: %s",
                self.input_file
            )
            return

        self.write_header()

        total_tests = 0
        correct_tests = 0

        with open(
            self.input_file,
            "r",
            newline="",
            encoding="utf-8",
            errors="replace"
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                if rospy.is_shutdown():
                    break

                total_tests += 1

                rospy.loginfo(
                    "[LLM Eval] Running test: %s",
                    row.get("test_name", "unknown")
                )

                result = self.evaluate_test(row)

                if result["response_correct"]:
                    correct_tests += 1

                self.write_result(row, result)

                rospy.loginfo(
                    "[LLM Eval] Model: %s | Test: %s | "
                    "Expected: %s | Predicted: %s | "
                    "Parser success: %s | Correct: %s | "
                    "Latency: %.2f ms | Error: %s",
                    result["model_name"],
                    row.get("test_name", "unknown"),
                    result["expected_actions"],
                    result["predicted_actions"],
                    result["parser_success"],
                    result["response_correct"],
                    result["latency_ms"],
                    result["error_type"]
                )

        accuracy = (
            correct_tests / total_tests * 100.0
            if total_tests > 0
            else 0.0
        )

        rospy.loginfo(
            "[LLM Eval] Finished: %d/%d correct = %.2f%%",
            correct_tests,
            total_tests,
            accuracy
        )

        rospy.loginfo(
            "[LLM Eval] Results saved to: %s",
            self.output_file
        )


if __name__ == "__main__":
    evaluator = LLMEvaluator()
    evaluator.run()
