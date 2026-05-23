# Bachelor Thesis – Multimodal Human Robot Interaction

## Project Overview

This repository contains the implementation and debugging progress of my bachelor thesis project based on:

- ROS1 Noetic
- Gazebo
- RViz
- TCC-IRoNL framework
- multimodal robot interaction
- pre-trained language and vision foundation models

The project focuses on natural-language interaction with autonomous robotic agents using navigation, perception, and language-based commands.

---

## Current Functionality

The following components are currently working:

- Gazebo simulation
- RViz visualization
- ChatGUI
- ROS backend
- OpenAI API integration
- semantic command recognition
- destination resolution (e.g. "navigate to kitchen")
- global path generation in RViz
- manual robot movement via `/cmd_vel`

---

## Current Issue

The remaining issue is related to autonomous navigation execution in Gazebo.

Observed behavior:

- global path planning works
- RViz and Gazebo are synchronized
- `move_base` receives goals correctly
- manual `/cmd_vel` commands move the robot
- however, the robot does not reliably follow the generated navigation path

The issue is suspected to be related to one of the following:

- local planner behavior
- differential drive configuration
- odometry / TF consistency
- `move_base` execution
- Gazebo robot controller configuration

---

## Launch Instructions

### Navigation Stack

```bash
roslaunch /root/catkin_ws/src/romr_ros/launch/romr_navigation.launch
