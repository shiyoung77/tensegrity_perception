# tensegrity_perception

## Introduction
This is the official repository for the paper "6N-DoF Pose Tracking for Tensegrity Robot". This work aims to address what has been recognized as a grand challenge in this domain, i.e., the pose tracking of tensegrity robots, through a markerless, vision-based method, as well as novel, on-board sensors that can measure the length of the robot's cables.  In particular, an iterative optimization process is proposed to estimate the 6-DoF poses of each rigid element of a tensegrity robot from an RGB-D video as well as endcap distance measurements from the cable sensors. To ensure the pose estimates of rigid elements are physically feasible, i.e., they are not resulting in collisions between rods or with the environment, physical constraints are introduced during the optimization.

## Installation
1. Install Miniconda/Anaconda
2. conda env create -f environment.yml

## Dataset
