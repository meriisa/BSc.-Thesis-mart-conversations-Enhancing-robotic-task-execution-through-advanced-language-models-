# Bachelor Thesis – Smart Conversations: Enhancing Robotic Task Execution through Advanced Language Models

## Overview

This repository contains the implementation developed as part of my Bachelor's thesis at the Chair of Cyber-Physical Systems (CPS), Montanuniversität Leoben.

The project extends the **TCC-IRoNL** framework by integrating modern Vision-Language Models (VLMs) and Large Language Models (LLMs) into a robot system. The objective is to improve natural human-robot interaction by enabling robots to understand natural language instructions, perceive their environment, ground commands to detected objects, and execute navigation tasks in a simulated environment.

The complete system is implemented in **ROS1 Noetic** and evaluated in **Gazebo** using a modular architecture that separates perception, language understanding, planning, and execution.

---

# Thesis Objectives

The primary objectives of this thesis are:

- Integrate Vision-Language Models into the robotic perception pipeline.
- Integrate and compare multiple Large Language Models for natural language understanding.
- Improve command parsing and semantic grounding.
- Evaluate perception and language models using reproducible benchmarks.
- Validate the complete perception-to-action pipeline within a simulated robotic environment.

---

# System Architecture

The overall pipeline consists of four main stages:

```
Natural Language Command
            │
     Large Language Model
      (GPT-4o / Llama 3)
            │
     Parsed Robot Actions
            │
 Vision-Language Model
(YOLOv8 / YOLO-World / SAM3)
            │
 Object Grounding
(Depth + Coordinate Estimation)
            │
 Navigation & Robot Execution
```

---

# Implemented Components

## Robot Platform

- ROS1 Noetic
- Gazebo Simulation
- RViz
- move_base Navigation
- Actionlib
- ChatGUI Interface

---

## Vision-Language Models (VLM)

The following perception models have been integrated and evaluated:

- YOLOv8-s
- YOLOv8-m
- YOLO-World-s
- YOLO-World-m
- SAM3

The perception module performs:

- Object Detection
- Open-Vocabulary Detection
- Semantic Segmentation
- Depth Estimation
- Object Coordinate Extraction
- Grounding of detected objects

---

## Large Language Models (LLM)

The language understanding component uses a modular provider architecture that allows different LLMs to be evaluated within the same framework.

Implemented models:

- GPT-4o
- Llama 3

The LLM is responsible for:

- Natural language understanding
- Command parsing
- Action generation
- Object reference resolution
- Structured robot action planning

---

# Evaluation

## Vision-Language Model Evaluation

Different perception models are compared using standardized test scenarios.

Evaluation metrics include:

- Detection Accuracy
- Precision
- Recall
- F1-score
- Confidence
- Average Inference Time

---

## Large Language Model Evaluation

The language models are evaluated regarding:

- Command interpretation accuracy
- Robustness against paraphrased instructions
- Action generation correctness
- Task completion
- Response latency
  
---

# Running the System

## Start the Navigation Stack

```bash
roslaunch romr_ros romr_navigation.launch
```

## Start the Backend

```bash
export OPENAI_API_KEY="YOUR_API_KEY"

roslaunch tcc_ros tcc_ros.launch
```

## Start the Chat Interface

```bash
roslaunch tcc_ros chatGUI.launch
```
---

# Current Project Status

Implemented:

- ROS1 integration
- Gazebo simulation
- Chat interface
- GPT-4o integration
- Llama 3 integration
- Vision-language model integration
- Modular LLM provider
- Natural language command parser
- Object grounding
- Robot navigation framework
- VLM evaluation framework
- LLM evaluation framework

Current work:

- Final end-to-end experiments
- Performance analysis
- Thesis documentation

---

# References

This project builds upon the TCC-IRoNL framework developed at the Chair of Cyber-Physical Systems.

Related project:

https://cps.unileoben.ac.at/open-project-msc-or-bsc-thesis-multimodal-human-autonomous-agents-interaction-using-pre-trained-language-and-visual-foundation-models/

Original repository:

https://github.com/LinusNEP/TCC-IRoNL
