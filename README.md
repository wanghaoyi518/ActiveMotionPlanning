# ActiveMotionPlanning
Active Inference-based Planning for Safe Human-Robot Interaction: Concurrent Considertation of Human Characteristic and Rationality

## Overall Architecture  
<img src="https://github.com/HMCL-UNIST/ActiveMotionPlanning/assets/86097031/3530b829-8913-4702-ad31-3a84791c691d" width="600" height="350"/>

We develop the algorithm for a autonomous vehicle to safely and actively interact with an uncertain human-driven vehicle.

## Paper
Y. Nam and C. Kwon “Active Inference-Based Planning for Safe Human-Robot Interaction: Concurrent Consideration of Human Characteristic and Rationality,” IEEE Robotics and Automation Letters, Vol. 9, No. 8, Page 7086-7093, August 2024 [(Link)](https://ieeexplore.ieee.org/document/10561626)

## Experiment Results
[![Video Label](http://img.youtube.com/vi/6081U1P5cSU/0.jpg)](https://youtu.be/6081U1P5cSU?feature=shared)

## Dependency
1. NVIDIA Graphic card and torch is required
   - pip install torch
   
## Information
1. Directory  
main.py       --- Main algorithm file  
human_model    --- Folder for human decision making algorithms  
motion_planning   --- Folder for robot motion planning algorithms  
utils         --- Folder for visualization and reference trajectory algorithms  
result        --- Simulation results will be saved in this folder    
README.md     --- Instruction file  

2. How to run
   - python main.py

