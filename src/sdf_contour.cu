// ---------------------------------------------------------
// Author: Yanhao Zhang, University of Technology Sydney, 2021.03.22
// ---------------------------------------------------------

// yanhao modify, 2019-12-29
// cuda basic introduction https://kezunlin.me/post/7d7131f4/
//                         https://github.com/yszheda/wiki/wiki/CUDA

// printf in kernel function:
  // cudaStream_t stream;
  // CalculateTSDF <<< grid, block ,bits,stream>>>()
  // cudaStreamSynchronize(stream);



#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
// #include "sdf_contour_utils.hpp"
#include "vector_ops.hpp"

// yanhao
#include <cuda.h>
#include <stdio.h>
#include <math.h> 
#include<time.h>
// #define MUM_obser 1228  // number of pixel observaiton
// #define DIM   2           // dimension
// #define M_PI           3.14159265358979323846  /* pi */
// #define cudaSafeCall(expr)  pcl::gpu::___cudaSafeCall(expr, __FILE__, __LINE__, __func__)

/* CUDA kernel function to calculate a TSDF */
// output: voxel_grid_TSDF: SDF
//         voxel_grid_normalvector: normal vector of corresponding boundary pixel
//         voxel_grid_obserID: corresponding boundary id just for debug
// input: bound_id: boundary pixels id w.r.t. cuda. This id also represents pixel coordinate
//        sdf_config: sdf configuration
//todo: now, the sdf grid size is fixed as 1.         
__global__
void CalculateTSDF(float * voxel_grid_TSDF, float * voxel_grid_normalvector, /*debug*/ int * voxel_grid_obserID, 
                   float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_size, float trunc_margin, float theta_threshold_innerproduct,
                   unsigned int * bound_id, float2 * bound_pixel, float2 * bound_normalvector, unsigned int num_bound, 
                   unsigned int grid_dim_x, unsigned int grid_dim_y){

  int pt_grid_x = blockIdx.x;
  int pt_grid_y = threadIdx.x;

   // do not consider the surrounding edge
  //  if (pt_grid_x==0 || pt_grid_y==0 || pt_grid_x==grid_dim_x || pt_grid_y==grid_dim_y){
  //   return;
  // }


  // Convert voxel center from grid coordinates to base frame camera coordinates
  float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
  float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;

  float2 pt_pix = make_float2(roundf(pt_base_x), roundf(pt_base_y));    // [i,j]

  //todo vectorilization 
  // find the min euclidian distance
  float dist_min = 10000.0f;
  int   id_min = -1;
  for (int i = 0; i < num_bound; ++i) {
      // loop each observation point
      float2 obser_pix = bound_pixel[i];
      float dist =  norm(pt_pix - obser_pix);  // distance between this point and observation

      //todo: this can be optimized
      if (dist>=dist_min){
        continue;
      } else {
        dist_min = dist;
        id_min = i;
      }
  }
  // get the minimum observation
  float2 obser_min = bound_pixel[id_min];  // the corresponding boundary pixel with shortest distance between pt_pix
  int obser_id_min = bound_id[id_min];     // id of corresponding boundary normal vector
  float2 obser_min_nv = bound_normalvector[obser_id_min];

  // normal vector between this point and the corresponding normal vector 
  float2 diff_nv = pt_pix - obser_min;  
  dist_min = std::abs(dot(obser_min_nv, diff_nv));  // calculate point to plane distance
  normalize(diff_nv);  // normalize this vector

  // sign
  float sign = 0.0f;
  float innerproduct_sign = dot(diff_nv, obser_min_nv);
  if(innerproduct_sign>1e-6){
    sign = 1.0f;
  } else if (innerproduct_sign<-1e-6){
    sign = -1.0f;
  } else {
    sign = 0.0f;
  }

  // debug
  // printf("[i j], innerproduct_sign, sign: [%d %d], %f, %f\n", pt_pix_x,pt_pix_y, innerproduct_sign,sign);
  
  // check whether we update the tsdf
   bool  b_update_tsdf = true; 
  // check distance condition
  // float dist_tsdf = dist_min;    
  if(dist_min>=trunc_margin){
     b_update_tsdf = false;  
    // dist_tsdf = trunc_margin;
  }
  //debug
  // printf("[i j], dist_min>=trunc_margin: [%d %d], %d\n", pt_pix_x,pt_pix_y, b_update_tsdf);

  // check theta condition
  // float innerproduct_abs = sign*innerproduct_sign;   //absolute value
  // if(innerproduct_abs < theta_threshold_innerproduct){
  //   b_update_tsdf = false; 
  // }

  //debug
  // printf("[i j], innerproduct_sign, sign innerproduct_abs b_update_tsdf: [%d %d], %f, %f, %f, %d\n", pt_pix_x,pt_pix_y, innerproduct_sign,sign,innerproduct_abs,b_update_tsdf);
  // printf("[i j], theta_threshold_innerproduct: [%d %d], %f\n", pt_pix_x,pt_pix_y, theta_threshold_innerproduct);

  // smoothness for un continuous pixels
  if(dist_min<3.0f){
    b_update_tsdf = true;
  }
  // if( ((dist_min-1.0f)<= 1e-6) && ((1.0f-dist_min)<= 1e-6) ){
  //   b_update_tsdf = true; 
  // }
  // if( (((float) (dist_min-std::sqrt(2.0f)))<= 1e-6) &&
  //     (((float) (std::sqrt(2.0f)-dist_min))<= 1e-6) ){
  //   b_update_tsdf = true; 
  // }
  
  // store tsdf  
  // id for store tsdf: For a 2D block(Dx,Dy), the id of thread is (x+y∗Dx). // https://kezunlin.me/post/7d7131f4/
  int volume_idx = pt_grid_y * grid_dim_x + pt_grid_x;
  // voxel_grid_TSDF[volume_idx] = trunc_margin;     // initialize 
  if(b_update_tsdf){
    voxel_grid_TSDF[volume_idx] = sign*dist_min;
  } else {
    voxel_grid_TSDF[volume_idx] = sign*trunc_margin;
  }
  
  // store normal vector as theta
  voxel_grid_normalvector[volume_idx] = (float) std::atan2(obser_min_nv.y,obser_min_nv.x);

  // for debug store observation id
  voxel_grid_obserID[volume_idx] =  id_min;

  //yh debug
  // printf("[i j], dist_min, theta: [%d %d], %f, %f\n", pt_pix_x,pt_pix_y,voxel_grid_TSDF[volume_idx],voxel_grid_normalvector[volume_idx]);
   
}

/* CUDA kernel function to calculate the gradient */
// output: voxel_grid_gradient: SDF gradient
// input: bound_id: boundary pixels id w.r.t. cuda. This id also represents pixel coordinate
//        sdf_config: sdf configuration
//todo: now, the sdf grid size is fixed as 1.  
// in matlab, simply call function [FY,FX] = gradient(DT,step,step);  DT: SDF  step=1;   
__global__
void Gradient2D(float2 * voxel_grid_gradient, 
                float * voxel_grid_TSDF, 
                float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_size, unsigned int grid_dim_x, unsigned int grid_dim_y){

 // voxel_grid_gradient_x is gradient along row axis (up2down)
 // voxel_grid_gradient_y is gradient along col axis (left2right)

  int pt_grid_x = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  // Convert voxel center from grid coordinates to base frame camera coordinates
  float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
  float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;

  int pt_pix_x = roundf(pt_base_x);   // i
  int pt_pix_y = roundf(pt_base_y);   // j

  // float2 pt_pix = make_float2(roundf(pt_base_x), roundf(pt_base_y));  // [i,j]

  // debug
  // printf("[pt_grid_x, pt_grid_y], [pt_pix_x, pt_pix_y]: [%d %d], [%d %d]\n", pt_grid_x,pt_grid_y, pt_pix_x,pt_pix_y);

  // Min and max of pixel id
  float pt_base_x_min = voxel_grid_origin_x;
  float pt_base_y_min = voxel_grid_origin_y;
  float pt_base_x_max = voxel_grid_origin_x + (grid_dim_x-1) * voxel_size;
  float pt_base_y_max = voxel_grid_origin_y + (grid_dim_y-1) * voxel_size;

  int pt_pix_x_min = roundf(pt_base_x_min);   // i
  int pt_pix_y_min = roundf(pt_base_y_min);   // j
  int pt_pix_x_max = roundf(pt_base_x_max);   // i
  int pt_pix_y_max = roundf(pt_base_y_max);   // j

  // grid coordinate of the two neighbouring grid
  int pt_grid_x_before, pt_grid_x_after,   volume_idx_before, volume_idx_after;
  int pt_grid_y_before, pt_grid_y_after,   volume_idy_before, volume_idy_after;

  // grid id for output
  int volume_idx = pt_grid_y * grid_dim_x + pt_grid_x;


  // debug
  // printf("[pt_pix_x_min, pt_pix_y_min, pt_pix_x_max, pt_pix_y_max]: [%d %d %d %d]\n", pt_pix_x_min, pt_pix_y_min, pt_pix_x_max, pt_pix_y_max);
  // printf("[i j], obser_x, obser_y, dist: [%d %d], %f, %f, %f\n", pt_pix_x,pt_pix_y,obser_x,obser_y,dist);

  // calculate x gradient: https://au.mathworks.com/help/matlab/ref/gradient.html
  // notice: y is actually changed
  if(pt_pix_y != pt_pix_y_min && pt_pix_y != pt_pix_y_max){
    // neighbour grid id
    pt_grid_y_before  = pt_grid_y - 1;
    pt_grid_y_after   = pt_grid_y + 1;
    volume_idx_before = pt_grid_y_before * grid_dim_x + pt_grid_x ;   // voxel_grid_TSDF[] start from zero
    volume_idx_after  = pt_grid_y_after * grid_dim_x + pt_grid_x;
    voxel_grid_gradient[volume_idx].x = 0.5f*(1.0f/voxel_size) * (voxel_grid_TSDF[volume_idx_after]-voxel_grid_TSDF[volume_idx_before]);

    // debug
    // if(pt_pix_x == 241 && pt_pix_y == 22) {
    //   printf("[pt_grid_x, pt_grid_y], [pt_pix_x, pt_pix_y]: [%d %d], [%d %d]\n", pt_grid_x,pt_grid_y, pt_pix_x,pt_pix_y);
    //   printf("[pt_pix_x_min, pt_pix_y_min, pt_pix_x_max, pt_pix_y_max]: [%d %d %d %d]\n", pt_pix_x_min, pt_pix_y_min, pt_pix_x_max, pt_pix_y_max);
    //   float dbg_grad_before = voxel_grid_TSDF[volume_idx_before];
    //   float dbg_grad_after  = voxel_grid_TSDF[volume_idx_after];
    //   float dbg_grad = voxel_grid_gradient_x[volume_idx];
    //   printf("[pt_grid_y_before, pt_grid_y_after, volume_idx_before, volume_idx_after]: [%d %d %d %d]\n", pt_grid_y_before, pt_grid_y_after, volume_idx_before, volume_idx_after);
    //   printf("grad_before, dbg_grad_after, dbg_grad: %f, %f, %f\n", dbg_grad_before, dbg_grad_after, dbg_grad);

    // }

  } else if (pt_pix_y == pt_pix_y_min) {
    // neighbour grid id
    pt_grid_y_before  = pt_grid_y;
    pt_grid_y_after   = pt_grid_y + 1;
    volume_idx_before = pt_grid_y_before * grid_dim_x + pt_grid_x ;   // voxel_grid_TSDF[] start from zero
    volume_idx_after  = pt_grid_y_after * grid_dim_x + pt_grid_x;
    voxel_grid_gradient[volume_idx].x = (1.0f/voxel_size) * (voxel_grid_TSDF[volume_idx_after]-voxel_grid_TSDF[volume_idx_before]);

  } else {
    // neighbour grid id
    pt_grid_y_before  = pt_grid_y - 1;
    pt_grid_y_after   = pt_grid_y;
    volume_idx_before = pt_grid_y_before * grid_dim_x + pt_grid_x ;   // voxel_grid_TSDF[] start from zero
    volume_idx_after  = pt_grid_y_after * grid_dim_x + pt_grid_x;
    voxel_grid_gradient[volume_idx].x = (1.0f/voxel_size) * (voxel_grid_TSDF[volume_idx_after]-voxel_grid_TSDF[volume_idx_before]);
  }

  // calculate y gradient
  if (pt_pix_x != pt_pix_x_min && pt_pix_x != pt_pix_x_max){
    // neighbour grid id
    pt_grid_x_before = pt_grid_x - 1;
    pt_grid_x_after  = pt_grid_x + 1;
    volume_idy_before = pt_grid_y * grid_dim_x + pt_grid_x_before;
    volume_idy_after  = pt_grid_y  * grid_dim_x + pt_grid_x_after;
    voxel_grid_gradient[volume_idx].y = 0.5f*(1.0f/voxel_size) * (voxel_grid_TSDF[volume_idy_after]-voxel_grid_TSDF[volume_idy_before]);
    
  } else if (pt_pix_x == pt_pix_x_min) {
    // neighbour grid id
    pt_grid_x_before = pt_grid_x;
    pt_grid_x_after  = pt_grid_x + 1;
    volume_idy_before = pt_grid_y * grid_dim_x + pt_grid_x_before;
    volume_idy_after  = pt_grid_y  * grid_dim_x + pt_grid_x_after;
    voxel_grid_gradient[volume_idx].y = (1.0f/voxel_size) * (voxel_grid_TSDF[volume_idy_after]-voxel_grid_TSDF[volume_idy_before]);

  } else {
    // neighbour grid id
    pt_grid_x_before = pt_grid_x - 1;
    pt_grid_x_after  = pt_grid_x;
    volume_idy_before = pt_grid_y * grid_dim_x + pt_grid_x_before;
    volume_idy_after  = pt_grid_y  * grid_dim_x + pt_grid_x_after;
    voxel_grid_gradient[volume_idx].y = (1.0f/voxel_size) * (voxel_grid_TSDF[volume_idy_after]-voxel_grid_TSDF[volume_idy_before]);
  }
}