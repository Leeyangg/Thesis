# Thesis

-> This project aims at studying the fusion of depth estimates coming from a convolutional neural network and stereo vision system (https://www.stereolabs.com/).  The neural network is trained online with supervisory input coming from a sparse (stereo) depth map. Ideally, the final system should be able to run on real-time so that can be used for indoor navigation on a drone or a MAV. The system is being implemented on the NVIDIA TX1 Development board (http://www.nvidia.com/object/jetson-tx1-module.html).

-> This repository will be used to keep track of the progress, which means that the code might not always be fully optimized, well documented or fully functional. There are a few important dependencies:
 - OpenCV
 - ZED SDK
 - CUDA
 - Caffe

## Features

The program receives as input a pre-trained CNN, which will be retrained online, and two images, which are fed to a stereo vision algorithm and to the CNN. The final output is a depth map which makes uses of both estimations to generate one with lower error than the individual estimations.

## Milestones 
- Extracting depth maps from CNN
- Extracting depth maps from ZED camera
- (Current State) Implementing merging algorithm presented in Mancini et al. 

 
## References

Facil, J. et al - Deep Single and Direct Multi-View Depth Fusion
Mancini, M. et al -  Towards Domain Independence for Learning-Based
Eigen et al  - Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture
