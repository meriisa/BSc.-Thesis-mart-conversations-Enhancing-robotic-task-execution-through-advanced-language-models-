# Bachelor Thesis – Smart conversations: Enhancing robotic task execution through advanced language models

## Project Overview

This repository contains the implementation and debugging progress of my bachelor thesis project based on ROS1 Noetic, Gazebo, RViz, TCC-IRoNL, multimodal robot interaction, and pre-trained language and vision foundation models.

## Current Functionality

- Gazebo simulation starts
- RViz visualization starts
- ChatGUI starts
- ROS backend starts
- OpenAI API integration works
- semantic command recognition works
- destination resolution works, e.g. `navigate to kitchen`
- global path generation in RViz works
- manual robot movement via `/cmd_vel` works

## Current Issue

The remaining issue is related to autonomous navigation execution in Gazebo.

Observed behavior:

- global path planning works
- RViz and Gazebo are synchronized
- `move_base` receives goals correctly
- manual `/cmd_vel` commands move the robot
- however, the robot does not reliably follow the generated navigation path

The issue is suspected to be related to local planner behavior, differential drive configuration, odometry / TF consistency, `move_base` execution, or Gazebo robot controller configuration.

## Launch Instructions

### Navigation Stack

```bash
roslaunch /root/catkin_ws/src/romr_ros/launch/romr_navigation.launch
```

### Backend

```bash
export OPENAI_API_KEY="YOUR_KEY"
roslaunch tcc_ros tcc_ros.launch
```

### Chat GUI

```bash
roslaunch tcc_ros chatGUI.launch
```

## Example Command

```text
navigate to kitchen
```

## Technologies Used

- ROS1 Noetic
- Gazebo
- RViz
- Docker
- Python
- OpenAI API
- move_base
- actionlib
- multimodal robotics frameworks
