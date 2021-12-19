// ---------------------------------------------------------
// Author: Yanhao Zhang, University of Technoligy Sydney, 2020.04.02
// calculate normal vector 
// based on https://au.mathworks.com/matlabcentral/fileexchange/32696-2d-line-curvature-and-normals
// ---------------------------------------------------------



// #pragma once
#ifndef CU_MEX_LINE_NORMAL
#define CU_MEX_LINE_NORMAL

#include <cuda.h>
#include <math.h>
#include "feature_contour_utils.hpp"
#include "vector_ops.hpp"

#define RADIUS_NORMALVECTOR 3   // left and right neighbour pixes used to calculate normal vector. A little bit different than the matlab code.


using namespace aortawarp; 



// calculate a the normal vector using the boundary
// input: gpu_boundary_next_id: order of next boundary in ccw and cw
//        gpu_boundary_image: (512*512)*1 boundary of binarized segmentation. 1 represent the boundary
// output: gpu_normalvector_image: normal vector
// modified based on: 
__global__ 
void LineNormal2D(float2 * gpu_normalvector_image,  NextID2D * gpu_boundary_next_id, bool * gpu_boundary_image, unsigned int grid_dim_x, unsigned int grid_dim_y){
  
    int pt_grid_x = blockIdx.x;
    int pt_grid_y = threadIdx.x;
    int volume_idx = pt_grid_y * grid_dim_x + pt_grid_x;   // For a 2D block(Dx,Dy), the id of thread is (x+y∗Dx). // https://kezunlin.me/post/7d7131f4/

    // do not consider the surrounding edge
    // if (pt_grid_x==0 || pt_grid_y==0 || pt_grid_x==grid_dim_x || pt_grid_y==grid_dim_y){
    //    return;
    // }

    float2 normal_vector = make_float2(0.0f, 0.0f);

    bool grid_occupation = gpu_boundary_image[volume_idx]; // get boundary
    if(grid_occupation){          // binarization=1 means this grid is occupied
        // get position of this point
        float2 boundary = make_float2((float)pt_grid_x, (float)pt_grid_y);
        // boundary.x = (float)pt_grid_x;
        // boundary.y = (float)pt_grid_y;

        // loop left and right RADIUS_NORMALVECTOR points
        float2 normal_vector_ccw = make_float2(0.0f, 0.0f);
        float2 normal_vector_cw = make_float2(0.0f, 0.0f);

        // float2 * tangent_vector_cw[RADIUS_NORMALVECTOR];
        int volume_next_id_ccw = volume_idx;
        int volume_next_id_cw = volume_idx;
        for (int i=1; i<=RADIUS_NORMALVECTOR; ++i){
            // get ccw&cw nextpoint
            int pt_next_ccw_x = gpu_boundary_next_id[volume_next_id_ccw].idx_ccw;
            int pt_next_ccw_y = gpu_boundary_next_id[volume_next_id_ccw].idy_ccw;
            volume_next_id_ccw = pt_next_ccw_y * grid_dim_x + pt_next_ccw_x;
            int pt_next_cw_x = gpu_boundary_next_id[volume_next_id_cw].idx_cw;
            int pt_next_cw_y = gpu_boundary_next_id[volume_next_id_cw].idy_cw;
            volume_next_id_cw = pt_next_cw_y * grid_dim_x + pt_next_cw_x;

            // tangent and normal vector ccw
            float2 next_boundary_ccw = make_float2((float)pt_next_ccw_x, (float)pt_next_ccw_y);  // get the point. if smoothing is used, replace this part
            float2 tangent_ccw = next_boundary_ccw - boundary;
            tangent_ccw = tangent_ccw * (1.0f / squared_norm(tangent_ccw));     // use squared norm as a weight: 1/dist^2
            normal_vector_ccw = normal_vector_ccw + make_float2(-tangent_ccw.y, tangent_ccw.x);   // R = [0 -1; 1 0] https://gamedev.stackexchange.com/questions/160344/algorithms-for-calculating-vertex-normals-in-2d-polygon

            // tangent and normal vector cw
            float2 next_boundary_cw = make_float2((float)pt_next_cw_x, (float)pt_next_cw_y);
            float2 tangent_cw = next_boundary_cw - boundary;                 
            tangent_cw = tangent_cw * (1.0f / squared_norm(tangent_cw));     // use squared norm as a weight: 1/dist^2
            normal_vector_cw = normal_vector_cw + make_float2(tangent_cw.y, -tangent_cw.x);  // R = [0 1; -1 0]
        }




        normal_vector = normal_vector_ccw + normal_vector_cw;   // 就是向量和再取单位向量作为normal vector
        normalize(normal_vector);
        
        // normalize( make_float2(-tangent_vector.y, tangent_vector.x));
  }                 



  gpu_normalvector_image[volume_idx] = normal_vector;


}

#endif