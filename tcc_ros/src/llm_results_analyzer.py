#!/usr/bin/env python3

import csv
import math
from collections import defaultdict


INPUT_FILE = "/root/llm_results.csv"
OUTPUT_FILE = "/root/llm_summary.csv"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def percent(value, total):
    return (value / total * 100) if total > 0 else 0.0


def main():
    rows = []

    with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No results found.")
        return

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_name"]].append(row)

    summary_rows = []

    print("\nLLM Evaluation Summary")
    print("=" * 80)

    for model_name, model_rows in by_model.items():
        total = len(model_rows)

        correct = sum(row["response_correct"] == "True" for row in model_rows)
        grounding_correct = sum(row["grounding_correct"] == "True" for row in model_rows)
        valid_json = sum(row["valid_json"] == "True" for row in model_rows)

        latencies = [float(row["latency_ms"]) for row in model_rows]

        print(f"\nModel: {model_name}")
        print("-" * 80)
        print(f"Total test cases:        {total}")
        print(f"Response Accuracy:       {correct}/{total} = {percent(correct, total):.2f}%")
        print(f"Grounding Accuracy:      {grounding_correct}/{total} = {percent(grounding_correct, total):.2f}%")
        print(f"Valid JSON Rate:         {valid_json}/{total} = {percent(valid_json, total):.2f}%")
        print(f"Average latency:         {mean(latencies):.2f} ms")
        print(f"Latency std. deviation:  {std(latencies):.2f} ms")

        summary_rows.append([
            model_name,
            "overall",
            total,
            correct,
            round(percent(correct, total), 2),
            grounding_correct,
            round(percent(grounding_correct, total), 2),
            valid_json,
            round(percent(valid_json, total), 2),
            round(mean(latencies), 2),
            round(std(latencies), 2)
        ])

        by_category = defaultdict(list)
        for row in model_rows:
            by_category[row["category"]].append(row)

        print("\nAccuracy by category:")
        for category, category_rows in by_category.items():
            category_total = len(category_rows)
            category_correct = sum(row["response_correct"] == "True" for row in category_rows)

            print(
                f"  {category:15s}: "
                f"{category_correct}/{category_total} = "
                f"{percent(category_correct, category_total):.2f}%"
            )

            category_latencies = [float(row["latency_ms"]) for row in category_rows]

            summary_rows.append([
                model_name,
                category,
                category_total,
                category_correct,
                round(percent(category_correct, category_total), 2),
                "",
                "",
                "",
                "",
                round(mean(category_latencies), 2),
                round(std(category_latencies), 2)
            ])

        failure_cases = [row for row in model_rows if row["response_correct"] != "True"]

        print("\nFailure cases:")
        if not failure_cases:
            print("  None")
        else:
            for row in failure_cases:
                print(
                    f"  {row['test_name']} | "
                    f"category={row['category']} | "
                    f"expected={row['expected_object']} | "
                    f"predicted={row['predicted_object']} | "
                    f"error={row['error_type']}"
                )

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8", errors="replace") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name",
            "category",
            "total_tests",
            "correct",
            "response_accuracy_percent",
            "grounding_correct",
            "grounding_accuracy_percent",
            "valid_json",
            "valid_json_percent",
            "avg_latency_ms",
            "std_latency_ms"
        ])
        writer.writerows(summary_rows)

    print(f"\nSummary saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()