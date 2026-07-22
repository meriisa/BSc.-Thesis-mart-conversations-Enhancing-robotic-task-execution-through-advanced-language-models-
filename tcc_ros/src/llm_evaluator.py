#!/usr/bin/env python3

import csv
import json
import os
import time
from typing import Any, Dict, List, Tuple

import rospy

from tcc_ros.llm_node import LLMInterface
from tcc_ros.commands_parser import CommandParser


class LLMEvaluator:
    """Evaluate the LLM + CommandParser pipeline without executing actions."""

    API_ERROR_RESPONSES = (
        "i'm sorry, i couldn't process that request",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "service unavailable",
        "api error",
    )

    def __init__(self):
        rospy.init_node("llm_evaluator")

        self.input_file = rospy.get_param(
            "~input_file", "/root/llm_test_commands.csv"
        )
        self.output_file = rospy.get_param(
            "~output_file", "/root/llm_results.csv"
        )

        # Retry settings can be changed from rosrun.
        self.max_retries = int(rospy.get_param("~max_retries", 5))
        self.retry_wait_seconds = float(
            rospy.get_param("~retry_wait_seconds", 8.0)
        )
        self.test_delay_seconds = float(
            rospy.get_param("~test_delay_seconds", 2.0)
        )

        self.api_key = (
            os.getenv("OPENAI_API_KEY")
            or rospy.get_param("models/llm_api_key", "")
        )
        self.destinations = rospy.get_param("destinations", {})

        self.llm_interface = LLMInterface(
            api_key=self.api_key,
            destinations=self.destinations
        )
        self.command_parser = CommandParser(action_executor=None)

        self.model_name = getattr(
            self.llm_interface,
            "MODEL_NAME",
            rospy.get_param("models/llm_name", "unknown")
        )

        rospy.loginfo("[LLM Eval] Initialized.")
        rospy.loginfo("[LLM Eval] Input file: %s", self.input_file)
        rospy.loginfo("[LLM Eval] Output file: %s", self.output_file)
        rospy.loginfo("[LLM Eval] Model: %s", self.model_name)
        rospy.loginfo(
            "[LLM Eval] Retry settings: max=%d, wait=%.1fs, test delay=%.1fs",
            self.max_retries,
            self.retry_wait_seconds,
            self.test_delay_seconds
        )

    @staticmethod
    def normalize_text(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).lower().strip().split())

    @staticmethod
    def normalize_number(value: Any) -> Any:
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

    def normalize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            normalized = {}
            for key, item in value.items():
                normalized_key = self.normalize_text(key)
                normalized[normalized_key] = self.normalize_value(item)

            if "action" in normalized:
                normalized["action"] = str(
                    normalized["action"]
                ).upper()
            return normalized

        if isinstance(value, list):
            return [self.normalize_value(item) for item in value]

        if isinstance(value, str):
            return self.normalize_number(self.normalize_text(value))

        return self.normalize_number(value)

    def normalize_actions(
        self,
        actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not isinstance(actions, list):
            return []

        return [
            self.normalize_value(action)
            for action in actions
            if isinstance(action, dict)
        ]

    def read_expected_actions(
        self,
        row: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        expected_json = row.get("expected_actions_json", "").strip()

        if not expected_json:
            return []

        expected_actions = json.loads(expected_json)

        if isinstance(expected_actions, dict):
            expected_actions = [expected_actions]

        if not isinstance(expected_actions, list):
            raise ValueError(
                "expected_actions_json must contain a list or object."
            )

        return self.normalize_actions(expected_actions)

    @staticmethod
    def numbers_equal(
        first: Any,
        second: Any,
        tolerance: float = 0.001
    ) -> bool:
        if (
            isinstance(first, (int, float))
            and not isinstance(first, bool)
            and isinstance(second, (int, float))
            and not isinstance(second, bool)
        ):
            return abs(float(first) - float(second)) <= tolerance

        return first == second

    def dictionaries_match(
        self,
        predicted: Dict[str, Any],
        expected: Dict[str, Any]
    ) -> bool:
        """All expected fields must match; extra predicted fields are allowed."""
        for key, expected_value in expected.items():
            if key not in predicted:
                return False

            predicted_value = predicted[key]

            if isinstance(expected_value, dict):
                if not isinstance(predicted_value, dict):
                    return False
                if not self.dictionaries_match(
                    predicted_value, expected_value
                ):
                    return False
            elif not self.numbers_equal(
                predicted_value, expected_value
            ):
                return False

        return True

    def compare_actions(
        self,
        predicted: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        predicted = self.normalize_actions(predicted)
        expected = self.normalize_actions(expected)

        if len(predicted) != len(expected):
            return False, "action_count_mismatch"

        for predicted_action, expected_action in zip(
            predicted, expected
        ):
            if not self.dictionaries_match(
                predicted_action, expected_action
            ):
                predicted_type = predicted_action.get("action", "")
                expected_type = expected_action.get("action", "")

                if predicted_type != expected_type:
                    return False, "wrong_action"

                return False, "wrong_parameter"

        return True, ""

    def is_temporary_api_failure(
        self,
        llm_result: Any,
        raw_response: str
    ) -> bool:
        """Detect failures swallowed by LLMInterface and returned as text."""
        response_text = self.normalize_text(raw_response)

        if any(
            marker in response_text
            for marker in self.API_ERROR_RESPONSES
        ):
            return True

        if not isinstance(llm_result, dict):
            return True

        result_type = self.normalize_text(
            llm_result.get("type", "")
        )

        # An entirely empty result is treated as a temporary pipeline failure.
        if not result_type and not response_text:
            return True

        return False

    def call_llm_with_retry(
        self,
        user_command: str
    ) -> Tuple[Dict[str, Any], float, int]:
        """
        Call the LLM and retry temporary API failures.

        Only the latency of the successful API request is returned. Waiting time
        caused by rate limiting is not counted as model inference latency.
        """
        last_result: Dict[str, Any] = {
            "type": "RESPONSE",
            "content": ""
        }

        for attempt in range(1, self.max_retries + 1):
            start_time = time.time()

            try:
                llm_result = self.llm_interface.process_input(
                    user_command,
                    current_yaw="0.0",
                    cardinal_direction="north",
                    position_x="0.0",
                    position_y="0.0",
                    position_z="0.0"
                )
            except Exception as exc:
                request_latency_ms = (
                    time.time() - start_time
                ) * 1000.0
                llm_result = {
                    "type": "RESPONSE",
                    "content": str(exc)
                }
                rospy.logwarn(
                    "[LLM Eval] API call failed on attempt %d/%d: %s",
                    attempt,
                    self.max_retries,
                    exc
                )
            else:
                request_latency_ms = (
                    time.time() - start_time
                ) * 1000.0

            if not isinstance(llm_result, dict):
                llm_result = {
                    "type": "RESPONSE",
                    "content": str(llm_result)
                }

            last_result = llm_result
            raw_response = str(llm_result.get("content", ""))

            if not self.is_temporary_api_failure(
                llm_result, raw_response
            ):
                return llm_result, request_latency_ms, attempt

            if attempt < self.max_retries:
                rospy.logwarn(
                    "[LLM Eval] Temporary API failure. "
                    "Retry %d/%d in %.1f seconds.",
                    attempt + 1,
                    self.max_retries,
                    self.retry_wait_seconds
                )
                rospy.sleep(self.retry_wait_seconds)

        raise RuntimeError(
            "API request failed after "
            f"{self.max_retries} attempts. Last response: "
            f"{last_result.get('content', '')}"
        )

    def run_pipeline(
        self,
        user_command: str
    ) -> Tuple[str, List[Dict[str, Any]], float, str, int]:
        llm_result, latency_ms, attempts = self.call_llm_with_retry(
            user_command
        )

        raw_response = str(llm_result.get("content", ""))
        parsed_result = self.command_parser.parse_input(llm_result)
        result_type = parsed_result.get("type", "UNKNOWN")

        if result_type == "ACTIONS":
            actions = parsed_result.get("content", [])
        else:
            actions = []

        return (
            raw_response,
            self.normalize_actions(actions),
            latency_ms,
            result_type,
            attempts
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
                result_type,
                attempts
            ) = self.run_pipeline(user_command)

            # For ambiguous/invalid commands, [] is the desired result.
            if expected_actions:
                parser_success = (
                    result_type == "ACTIONS"
                    and len(predicted_actions) > 0
                )
            else:
                parser_success = len(predicted_actions) == 0

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
                "attempts": attempts,
                "raw_response": raw_response.replace("\n", " ")
            }

        except Exception as exc:
            rospy.logerr(
                "[LLM Eval] Test '%s' could not be evaluated: %s",
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
                "error_type": "api_failed_after_retries",
                "attempts": self.max_retries,
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
                "attempts",
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
                result["attempts"],
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
        valid_outputs = 0
        failed_api_tests = 0
        successful_latencies = []

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
                test_name = row.get("test_name", "unknown")

                rospy.loginfo(
                    "[LLM Eval] Running test: %s",
                    test_name
                )

                result = self.evaluate_test(row)
                self.write_result(row, result)

                if result["error_type"] == "api_failed_after_retries":
                    failed_api_tests += 1
                else:
                    successful_latencies.append(
                        result["latency_ms"]
                    )

                if result["parser_success"]:
                    valid_outputs += 1

                if result["response_correct"]:
                    correct_tests += 1

                rospy.loginfo(
                    "[LLM Eval] %s | Correct: %s | "
                    "Latency: %.2f ms | Attempts: %d | Error: %s",
                    test_name,
                    result["response_correct"],
                    result["latency_ms"],
                    result["attempts"],
                    result["error_type"]
                )

                if self.test_delay_seconds > 0:
                    rospy.sleep(self.test_delay_seconds)

        evaluable_tests = total_tests - failed_api_tests

        accuracy = (
            correct_tests / evaluable_tests * 100.0
            if evaluable_tests > 0 else 0.0
        )
        valid_output_rate = (
            valid_outputs / evaluable_tests * 100.0
            if evaluable_tests > 0 else 0.0
        )
        average_latency = (
            sum(successful_latencies) / len(successful_latencies)
            if successful_latencies else 0.0
        )

        rospy.loginfo(
            "[LLM Eval] Finished: %d/%d evaluable tests correct.",
            correct_tests,
            evaluable_tests
        )
        rospy.loginfo(
            "[LLM Eval] Action Accuracy: %.2f%%",
            accuracy
        )
        rospy.loginfo(
            "[LLM Eval] Valid Output Rate: %.2f%%",
            valid_output_rate
        )
        rospy.loginfo(
            "[LLM Eval] Average successful-request latency: %.2f ms",
            average_latency
        )
        rospy.loginfo(
            "[LLM Eval] API failures after all retries: %d",
            failed_api_tests
        )
        rospy.loginfo(
            "[LLM Eval] Results saved to: %s",
            self.output_file
        )


if __name__ == "__main__":
    evaluator = LLMEvaluator()
    evaluator.run()
