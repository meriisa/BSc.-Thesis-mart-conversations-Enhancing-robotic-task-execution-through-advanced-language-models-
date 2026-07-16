#!/usr/bin/env python3

import os
import csv
import json
import time
import requests
import rospy
import openai


class LLMEvaluator:

    def __init__(self):
        rospy.init_node("llm_evaluator")

        self.input_file = rospy.get_param("~input_file", "/root/llm_test_commands.csv")
        self.output_file = rospy.get_param("~output_file", "/root/llm_results.csv")

        self.gpt_model = rospy.get_param("~gpt_model", "gpt-4")
        self.llama_model = rospy.get_param("~llama_model", "llama3")
        self.ollama_url = rospy.get_param("~ollama_url", "http://localhost:11434/api/generate")

        self.temperature = float(rospy.get_param("~temperature", 0.0))
        self.test_gpt = bool(rospy.get_param("~test_gpt", True))
        self.test_llama = bool(rospy.get_param("~test_llama", False))

        openai.api_key = os.getenv("OPENAI_API_KEY")

        rospy.loginfo("[LLM Eval] Initialized.")
        rospy.loginfo(f"[LLM Eval] Input file: {self.input_file}")
        rospy.loginfo(f"[LLM Eval] Output file: {self.output_file}")
        rospy.loginfo(f"[LLM Eval] GPT model: {self.gpt_model}")
        rospy.loginfo(f"[LLM Eval] LLaMA model: {self.llama_model}")
        rospy.loginfo(f"[LLM Eval] Test GPT: {self.test_gpt}")
        rospy.loginfo(f"[LLM Eval] Test LLaMA: {self.test_llama}")

    def normalize_text(self, text):
        return str(text).lower().strip()

    def build_prompt(self, detected_objects, user_command):
        prompt = f"""
You are the language understanding component of a mobile service robot.

The robot receives natural language commands from a user.
Your task is to convert the command into a structured robot action.

Detected objects in the current scene:
{detected_objects}

User command:
"{user_command}"

Return only valid JSON in exactly this format:

{{
  "action": "navigate",
  "target_object": "object_name",
  "confidence": 0.0
}}

Rules:
- Choose the target_object only from the detected objects.
- If the command refers to an object that is not detected, use "none".
- If the command is too ambiguous or incomplete, use "none".
- Do not explain your answer.
- Do not add text before or after the JSON.
"""
        return prompt

    def parse_json_response(self, response_text):
        try:
            clean_text = response_text.strip()

            if clean_text.startswith("```json"):
                clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text.replace("```", "").strip()

            data = json.loads(clean_text)

            action = self.normalize_text(data.get("action", "none"))
            target_object = self.normalize_text(data.get("target_object", "none"))
            confidence = float(data.get("confidence", 0.0))

            return action, target_object, confidence, True

        except Exception:
            return "none", "none", 0.0, False

    def call_gpt(self, prompt):
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        start_time = time.time()

        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You convert natural language robot commands into structured JSON commands."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        latency_ms = (time.time() - start_time) * 1000
        answer = response["choices"][0]["message"]["content"]

        return answer, latency_ms

    def call_llama(self, prompt):
        start_time = time.time()

        payload = {
            "model": self.llama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        response = requests.post(self.ollama_url, json=payload, timeout=120)
        response.raise_for_status()

        latency_ms = (time.time() - start_time) * 1000
        answer = response.json().get("response", "")

        return answer, latency_ms

    def evaluate_response(
        self,
        model_name,
        raw_response,
        latency_ms,
        expected_action,
        expected_object,
        detected_objects
    ):
        predicted_action, predicted_object, confidence, valid_json = self.parse_json_response(raw_response)

        expected_action = self.normalize_text(expected_action)
        expected_object = self.normalize_text(expected_object)

        detected_object_list = [
            self.normalize_text(obj)
            for obj in detected_objects.split(";")
            if obj.strip()
        ]

        response_correct = (
            predicted_action == expected_action and
            predicted_object == expected_object
        )

        grounding_correct = (
            predicted_object in detected_object_list or predicted_object == "none"
        )

        if response_correct and grounding_correct:
            error_type = "correct"
        elif not valid_json:
            error_type = "invalid_json"
        elif not grounding_correct:
            error_type = "grounding_error"
        elif predicted_object != expected_object:
            error_type = "wrong_target_object"
        elif predicted_action != expected_action:
            error_type = "wrong_action"
        else:
            error_type = "unknown_error"

        return {
            "model_name": model_name,
            "predicted_action": predicted_action,
            "predicted_object": predicted_object,
            "confidence": round(confidence, 3),
            "latency_ms": round(latency_ms, 2),
            "valid_json": valid_json,
            "response_correct": response_correct,
            "grounding_correct": grounding_correct,
            "error_type": error_type,
            "raw_response": raw_response.replace("\n", " ")
        }

    def write_header(self):
        with open(self.output_file, "w", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name",
                "test_name",
                "category",
                "user_command",
                "detected_objects",
                "expected_action",
                "expected_object",
                "predicted_action",
                "predicted_object",
                "confidence",
                "latency_ms",
                "valid_json",
                "response_correct",
                "grounding_correct",
                "error_type",
                "raw_response"
            ])

    def write_result(self, row, result):
        with open(self.output_file, "a", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow([
                result["model_name"],
                row["test_name"],
                row["category"],
                row["user_command"],
                row["detected_objects"],
                row["expected_action"],
                row["expected_object"],
                result["predicted_action"],
                result["predicted_object"],
                result["confidence"],
                result["latency_ms"],
                result["valid_json"],
                result["response_correct"],
                result["grounding_correct"],
                result["error_type"],
                result["raw_response"]
            ])

    def evaluate_model(self, model_name, model_function, row):
        prompt = self.build_prompt(
            row["detected_objects"],
            row["user_command"]
        )

        try:
            raw_response, latency_ms = model_function(prompt)

            result = self.evaluate_response(
                model_name=model_name,
                raw_response=raw_response,
                latency_ms=latency_ms,
                expected_action=row["expected_action"],
                expected_object=row["expected_object"],
                detected_objects=row["detected_objects"]
            )

        except Exception as e:
            rospy.logerr(f"[LLM Eval] {model_name} failed on {row['test_name']}: {e}")

            result = {
                "model_name": model_name,
                "predicted_action": "none",
                "predicted_object": "none",
                "confidence": 0.0,
                "latency_ms": 0.0,
                "valid_json": False,
                "response_correct": False,
                "grounding_correct": False,
                "error_type": "model_call_failed",
                "raw_response": str(e)
            }

        self.write_result(row, result)

        rospy.loginfo(
            f"Model: {model_name} | "
            f"Test: {row['test_name']} | "
            f"Expected: {row['expected_object']} | "
            f"Predicted: {result['predicted_object']} | "
            f"Correct: {result['response_correct']} | "
            f"Grounding: {result['grounding_correct']} | "
            f"Time: {result['latency_ms']} ms | "
            f"Error: {result['error_type']}"
        )

    def run(self):
        if not os.path.exists(self.input_file):
            rospy.logerr(f"[LLM Eval] Input file not found: {self.input_file}")
            return

        self.write_header()

        with open(self.input_file, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)

            for row in reader:
                rospy.loginfo(f"[LLM Eval] Running test: {row['test_name']}")

                if self.test_gpt:
                    self.evaluate_model("GPT-4", self.call_gpt, row)

                if self.test_llama:
                    self.evaluate_model("LLaMA3", self.call_llama, row)

        rospy.loginfo(f"[LLM Eval] Results saved to {self.output_file}")


if __name__ == "__main__":
    evaluator = LLMEvaluator()
    evaluator.run()