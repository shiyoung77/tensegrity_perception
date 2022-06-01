# tensegrity_perception
This is the official repository for the paper "6N-DoF Pose Tracking for Tensegrity Robot". [Paper Link](https://arxiv.org/abs/2205.14764)

## Introduction
 This work aims to address the pose tracking problem of an N-bar tensegrity robots through a markerless, vision-based method, as well as novel, on-board sensors that can measure the length of the robot's cables.  In particular, an iterative optimization process is proposed to estimate the 6-DoF poses of each rigid element of a tensegrity robot from an RGB-D video as well as endcap distance measurements from the cable sensors. To ensure the pose estimates of rigid elements are physically feasible, i.e., they are not resulting in collisions between rods or with the environment, physical constraints are introduced during the optimization.

![](https://i.imgur.com/pzNl5ek.gif)

## Installation
1. Install Miniconda/Anaconda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
2. `conda env create -f environment.yml`
3. `conda activate tensegrity`

## Dataset
The dataset that is used in the experiment section could be downloaded from [here](https://drive.google.com/file/d/1UzOfJ6mC3cEGLbyEmspnyonsnr9eirTZ/view?usp=sharing). It contains 16 RGB-D videos and the corresponding distance measurements for each frame.

## Usage
1. Either unzip the dataset in the repository and change its name to "dataset", or create a soft link in the repository `ln -s PATH_TO_THE_DATASET dataset`
2. A bash script named `tracking.sh` is provided, which contains commands to run the algorithm, evaluate, and generate videos for visualization.

## Citation
```
@misc{lu20226ndof,
    title={6N-DoF Pose Tracking for Tensegrity Robots},
    author={Shiyang Lu and William R. Johnson III and Kun Wang and Xiaonan Huang and Joran Booth and Rebecca Kramer-Bottiglio and Kostas Bekris},
    year={2022},
    eprint={2205.14764},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```