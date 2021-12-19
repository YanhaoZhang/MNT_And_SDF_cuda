# Parallel Moore-Neighbour Tracing and 2D SDF

This code is part of our TRO submission. It is a CUDA implementation of our parallel Moore-Neighbour Tracing and the 2D SDF.  

## Introduction

Moore-Neighbour Tracing is a common method to extract the boundary pixels from a binary silhouette. Please refer to [here](http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/moore.html#:~:text=The%20general%20idea%20is%3A%20every,you%20hit%20a%20black%20pixel.) for some more details. The traditional MNT is not run in a parallel process. Here, we improve the traditional MNT such that it is calculated in a different way, and run in a parallel process. The CUDA implementation of our method is released. We also release the code of calculating the 2D SDF from the extracted boundary pixels. 

Some more details:

* MATLAB code to calculate 2D SDF `calculate_sdf2D.m`
* MATLAB code to calculate 2D normal vector `LineNormals2D.m` (based on [here](https://au.mathworks.com/matlabcentral/fileexchange/32696-2d-line-curvature-and-normals?requestedDomain=)).
* CUDA code in `./src/`:
  * main function link MATLAB `main_sdf.cpp`
  * main function calculate SDF from a binary silhouette `main_sdf_cuda.cu`, `main_sdf_cuda.h`
  * some useful vector operators `vector_ops.hpp`
  * Calculate the normal vector of boundary  `line_normal.cu`
  * calculate Parallel Moore Neighbour Tracing `feature_contour.cu`, `feature_contour_utils.hpp`
  * calculate SDF and the gradient of SDF. `sdf_contour.cu`, `sdf_config.h`
  * Some other functions `cuda_utils.h`

## Usage

This code has been tested on Ubuntu 16.04 with CUDA 9.0, and MATLAB 2018a.

* Compile: In MATLAB command window, run `calculate_sdf2D.m` to compile the CUDA code. Some modification may be needed. 
* Run: an example on how to use the code and a comparison with the MATLAB code is in `main_PMNT_and_SDF2D.m`.

## Contacts:

Yanhao Zhang: yanhao.zhang@student.uts.edu.au. Any discussions or concerns are welcomed :)