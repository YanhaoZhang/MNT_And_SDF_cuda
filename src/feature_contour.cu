// ---------------------------------------------------------
// Author: Yanhao Zhang, University of Technoligy Sydney, 2020.03.27
// boundary tracing from a binarize image
// ---------------------------------------------------------

// ---------------------------------------------------------
// usdful doc: 
// cuda basic introduction https://kezunlin.me/post/7d7131f4/
//                         https://github.com/yszheda/wiki/wiki/CUDA
//
// printf in kernel function:
//                         cudaStream_t stream;
//                         CalculateTSDF <<< grid, block ,bits,stream>>>()
//                         cudaStreamSynchronize(stream);
// ---------------------------------------------------------


// first try 2020-03-27

// #pragma once

#ifndef CU_MEX_FEATURE_CONTOUR
#define CU_MEX_FEATURE_CONTOUR

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>



#include "feature_contour_utils.hpp"
#include "vector_ops.hpp"
#include "line_normal.cu"
#include "sdf_config.h"

#include <cuda.h>
#include <stdio.h>
#include <math.h> 
// #include<time.h>
// #define IMAGE_SIZE 512  // image size
#define NUM_MOORE_PIXEL 8

using namespace aortawarp; 


// calculate the index of 8 moore neighbour pixels
// input: pt_grid_x pt_grid_y grid_dim_x
// output: moore_id
__device__
void MooreNeighbourPixelId(int * moore_id, int pt_grid_x, int pt_grid_y, int grid_dim_x) {
  
  moore_id[0] =  pt_grid_y    * grid_dim_x + (pt_grid_x+1);
  moore_id[1] = (pt_grid_y-1) * grid_dim_x + (pt_grid_x+1);
  moore_id[2] = (pt_grid_y-1) * grid_dim_x + pt_grid_x;
  moore_id[3] = (pt_grid_y-1) * grid_dim_x + (pt_grid_x-1);
  moore_id[4] =  pt_grid_y    * grid_dim_x + (pt_grid_x-1);
  moore_id[5] = (pt_grid_y+1) * grid_dim_x + (pt_grid_x-1);
  moore_id[6] = (pt_grid_y+1) * grid_dim_x +  pt_grid_x;
  moore_id[7] = (pt_grid_y+1) * grid_dim_x + (pt_grid_x+1);
}
// same function with additional output moore_id_x and moore_id_y
__device__
void MooreNeighbourPixelId(int * moore_id, int * moore_id_x, int * moore_id_y, int pt_grid_x, int pt_grid_y, int grid_dim_x) {
  
  moore_id[0] =  pt_grid_y    * grid_dim_x + (pt_grid_x+1);
  moore_id[1] = (pt_grid_y-1) * grid_dim_x + (pt_grid_x+1);
  moore_id[2] = (pt_grid_y-1) * grid_dim_x + pt_grid_x;
  moore_id[3] = (pt_grid_y-1) * grid_dim_x + (pt_grid_x-1);
  moore_id[4] =  pt_grid_y    * grid_dim_x + (pt_grid_x-1);
  moore_id[5] = (pt_grid_y+1) * grid_dim_x + (pt_grid_x-1);
  moore_id[6] = (pt_grid_y+1) * grid_dim_x +  pt_grid_x;
  moore_id[7] = (pt_grid_y+1) * grid_dim_x + (pt_grid_x+1);

  // additional output
  moore_id_x[0] = pt_grid_x+1; 
  moore_id_x[1] = pt_grid_x+1; 
  moore_id_x[2] = pt_grid_x; 
  moore_id_x[3] = pt_grid_x-1; 
  moore_id_x[4] = pt_grid_x-1; 
  moore_id_x[5] = pt_grid_x-1; 
  moore_id_x[6] = pt_grid_x; 
  moore_id_x[7] = pt_grid_x+1; 
  
  moore_id_y[0] = pt_grid_y;
  moore_id_y[1] = pt_grid_y-1;
  moore_id_y[2] = pt_grid_y-1;
  moore_id_y[3] = pt_grid_y-1;
  moore_id_y[4] = pt_grid_y;
  moore_id_y[5] = pt_grid_y+1;
  moore_id_y[6] = pt_grid_y+1;
  moore_id_y[7] = pt_grid_y+1;

}


// calculate a the boundary from a binarized image
// input: gpu_binarized_image (512*512)*1 binarized segmentation from previous function
// output: gpu_boundary_image: (512*512)*1 boundary of binarized segmentation. 1 represent the boundary
// modified based on: http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/moore.html
// refer to my note
__global__ 
void MooreNeighbourBoundary(bool * gpu_boundary_image, bool * gpu_binarized_image, unsigned int grid_dim_x, unsigned int grid_dim_y){ 

  int pt_grid_x = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  // do not consider the surrounding edge
  // if (pt_grid_x==0 || pt_grid_y==0 || pt_grid_x==grid_dim_x || pt_grid_y==grid_dim_y){
  //   return;
  // }

  // For a 2D block(Dx,Dy), the id of thread is (x+y∗Dx). // https://kezunlin.me/post/7d7131f4/
  int volume_idx = pt_grid_y * grid_dim_x + pt_grid_x;

  bool grid_occupation = gpu_binarized_image[volume_idx];

  if(grid_occupation){    // binarization=1 means this grid is occupied
    // int num_moore_pixel = 8;
    // int * moore_id = new int[num_moore_pixel];   // will be slow if using this
    int moore_id[NUM_MOORE_PIXEL];     
    MooreNeighbourPixelId(moore_id,  pt_grid_x, pt_grid_y, grid_dim_x);

    // moore neighbour pixel id
    // int moore_id[8];
    // moore_id[0] =  pt_grid_y    * grid_dim_x + (pt_grid_x+1);
    // moore_id[1] = (pt_grid_y-1) * grid_dim_x + (pt_grid_x+1);
    // moore_id[2] = (pt_grid_y-1) * grid_dim_x + pt_grid_x;
    // moore_id[3] = (pt_grid_y-1) * grid_dim_x + (pt_grid_x-1);
    // moore_id[4] =  pt_grid_y    * grid_dim_x + (pt_grid_x-1);
    // moore_id[5] = (pt_grid_y+1) * grid_dim_x + (pt_grid_x-1);
    // moore_id[6] = (pt_grid_y+1) * grid_dim_x +  pt_grid_x;
    // moore_id[7] = (pt_grid_y+1) * grid_dim_x + (pt_grid_x+1);


    // loop all moore neighbour pixels
    int sum_moore_value = 0;    // store the sum of values of all neighbour pixels
    for (int i=0; i <NUM_MOORE_PIXEL; ++i){

      bool b_moore_value = gpu_binarized_image[moore_id[i]];   
      // int  d_moore_value = b_moore_value;                      
      sum_moore_value = sum_moore_value + (int)b_moore_value; // 强制类型转换

      // store the boundary image
      if(!b_moore_value){
        gpu_boundary_image[volume_idx] = 1;
      }
    }

    // remove the redundant 90 deg point
    if(sum_moore_value==7 && 
      (gpu_binarized_image[moore_id[1]]==0 || gpu_binarized_image[moore_id[3]]==0 || gpu_binarized_image[moore_id[5]]==0 || gpu_binarized_image[moore_id[7]]==0)
    ){
      gpu_boundary_image[volume_idx] = 0;
    }

  }   
}


/**  store the cuda id of boundary **/
__global__ 
void BoundaryID(unsigned int * gpu_bound_id_image, bool * gpu_boundary_image, unsigned int grid_dim_x, unsigned int grid_dim_y){ 

  int pt_grid_x = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  // do not consider the surrounding edge
  if (pt_grid_x==0 || pt_grid_y==0 || pt_grid_x==grid_dim_x || pt_grid_y==grid_dim_y){
    return;
  }

  // For a 2D block(Dx,Dy), the id of thread is (x+y∗Dx). // https://kezunlin.me/post/7d7131f4/
  int volume_idx = pt_grid_y * grid_dim_x + pt_grid_x;

  bool grid_occupation = gpu_boundary_image[volume_idx];  // if this grid id a boundary pixel


  gpu_bound_id_image[volume_idx] = 0;  // if it is not a boundary, store zero
  if(grid_occupation){
    gpu_bound_id_image[volume_idx] = volume_idx;  // else store cuda id

  }
}


/**  store the cuda boundary id in a compact array **/
// here the boundary id is rewrited from calculated boundary
// this function is on cpu, not gpu
void BoundaryIDRaw(unsigned int * bound_id_raw, unsigned int bound_max_num, bool * bound_seg, unsigned int grid_dim_x, unsigned int grid_dim_y){ 

  int num_bound = 0;
  
  for (int i=0; i <grid_dim_x * grid_dim_y; ++i){
    // bound_id_raw[num_bound] = 0;      // initialize bound_id_raw as zero
    if(bound_seg[i]) {
      bound_id_raw[num_bound+1] = i;     // put the first boundary id as second element
      num_bound++;
    }
  }

  // make sure the threshold of bound_max_num is large enough
  if (num_bound>bound_max_num){
    std::cout << 'BoundaryIDCompact error: num_bound>bound_max_num' << '\n';
  }

  bound_id_raw[0] = num_bound;   // store the actual number of boundary pixels as the first element
}

/* stack the boundary id and boundary pixel */
// output: bound_id_stack, bound_pixel_stack, 
// input:  bound_id_raw, mum_bound
void StackBoundaryIDAndPixel(unsigned int * bound_id_stack, float2 * bound_pixel_stack, unsigned int * bound_id_raw, 
                             unsigned int mum_bound, float voxel_grid_origin_x, unsigned int grid_dim_x){ 

  for (int i=0; i <mum_bound; ++i){
    bound_id_stack[i] = bound_id_raw[i+1];    // store boundary id
    
    int divid =  bound_id_stack[i] + 1;
    bound_pixel_stack[i].y = (float) (divid / grid_dim_x) + voxel_grid_origin_x;     // notice: inverse y and x
    bound_pixel_stack[i].x = (float) (divid % grid_dim_x);
  }
}




// calculate the orientation of three point
// input: 2d array of pa pb pc
// output: det (sgn can be check in kernel function)
// algorithm: http://www.cs.cmu.edu/~quake/robust.html
__device__
float Orientation( float * pa, float * pb, float * pc) {
  
  float detleft, detright, det;
  
  detleft  = (pa[0] - pc[0]) * (pb[1] - pc[1]);
  detright  = (pa[1] - pc[1]) * (pb[0] - pc[0]);
  det = detleft - detright;

  // return sign(det);   // does this work?
  return det;  
}

// calculate a the boundary order
// input: gpu_boundary_image: (512*512)*1 boundary of binarized segmentation. 1 represent the boundary
//        gpu_binarized_image: binarized segmentation
// output: gpu_boundary_next_id: order of next boundary in ccw and cw
// modified based on: 
// refer to my note
__global__ 
void SortBoundary(NextID2D * gpu_boundary_next_id, bool * gpu_boundary_image, bool * gpu_binarized_image, unsigned int grid_dim_x, unsigned int grid_dim_y) {

  int pt_grid_x = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  // do not consider the surrounding edge
  if (pt_grid_x==0 || pt_grid_y==0 || pt_grid_x==grid_dim_x-1 || pt_grid_y==grid_dim_y-1){
    return;
  }

  // For a 2D block(Dx,Dy), the id of thread is (x+y∗Dx). // https://kezunlin.me/post/7d7131f4/
  int volume_idx = pt_grid_y * grid_dim_x + pt_grid_x;

  bool grid_occupation = gpu_boundary_image[volume_idx];

  if(grid_occupation){                 // binarization=1 means this grid is occupied
    int moore_id[NUM_MOORE_PIXEL];     // get moore neighbour id
    int moore_idx[NUM_MOORE_PIXEL];    // get moore neighbour id_x
    int moore_idy[NUM_MOORE_PIXEL];    // get moore neighbour id_y
    MooreNeighbourPixelId(moore_id,moore_idx,moore_idy,  pt_grid_x, pt_grid_y, grid_dim_x);

    // loop all moore neighbour pixels
    float sum_unoccupied_moore_value_x = 0.0;    // store the sum of values of all unoccupied neighbour pixels, namely zero in binarized segmentaiton
    float sum_unoccupied_moore_value_y = 0.0;
    int   num_unoccupied_moore_value   = 0;
    
    // calculate the mean of all unoccupied neighbour pixels
    for (int i=0; i <NUM_MOORE_PIXEL; ++i){
      // load binarized segmentation value
      bool b_moore_value = gpu_binarized_image[moore_id[i]];   
      // store the sum of unoccupied neighbour
      if(b_moore_value){
        sum_unoccupied_moore_value_x = sum_unoccupied_moore_value_x + moore_idx[i];
        sum_unoccupied_moore_value_y = sum_unoccupied_moore_value_y + moore_idy[i];
        num_unoccupied_moore_value++;
      }
    }
    float mean_unoccupied_moore_value_x = sum_unoccupied_moore_value_x/num_unoccupied_moore_value;
    float mean_unoccupied_moore_value_y = sum_unoccupied_moore_value_y/num_unoccupied_moore_value;

    // calculate order
    float dist_square_threshold = 20.0;   // just give a value larger than 2. 
    float dist_square_min_ccw = dist_square_threshold;
    float dist_square_min_cw =  dist_square_threshold;
    for (int i=0; i <NUM_MOORE_PIXEL; ++i){ 
      // check moore neighbour in boundary image, loop all neighbour boundary
      bool neighbour_boundary_occupation = gpu_boundary_image[moore_id[i]];
      if(neighbour_boundary_occupation){
        //check the orientation
        float pa[2] = {(float)pt_grid_x, (float)pt_grid_y};  
        float pb[2] = {(float)moore_idx[i], (float)moore_idy[i]};
        float pc[2] = {mean_unoccupied_moore_value_x, mean_unoccupied_moore_value_y};
        float dist_square = (pa[0] - pb[0])*(pa[0] - pb[0]) + (pa[1] - pb[1])*(pa[1] - pb[1]);   // distance between pa and pb
        float det = Orientation(pa,pb,pc);

        if(det<0){
          if(dist_square<dist_square_min_ccw){
            gpu_boundary_next_id[volume_idx].idx_ccw = moore_idx[i];   // store ccw id
            gpu_boundary_next_id[volume_idx].idy_ccw = moore_idy[i];
            dist_square_min_ccw = dist_square;
          }
        }

        if(det>0){
          if(dist_square<dist_square_min_cw){
            gpu_boundary_next_id[volume_idx].idx_cw = moore_idx[i];   // store cw id
            gpu_boundary_next_id[volume_idx].idy_cw = moore_idy[i];
            dist_square_min_cw = dist_square;
          }
        }

      }

    }

  }

}




#endif