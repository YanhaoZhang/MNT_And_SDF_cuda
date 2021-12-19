// ---------------------------------------------------------
// Author: Yanhao Zhang, University of Technology Sydney, 2021.03.22
// main function calculates the following from a binarilized segmentation:
// 1. boundary pixels, 
// 2. normal of boundary
// 3. cuda id of boundary pixel
// 4. sdf
// 5. gradient of sdf
// ---------------------------------------------------------

#include "cuda_utils.h"
#include "main_sdf_cuda.h"
#include "feature_contour_utils.hpp"
#include "vector_ops.hpp"
#include "line_normal.cu"
#include "feature_contour.cu"
#include "sdf_config.h"
#include "sdf_contour.cu"
// #include "sdf_contour_debug.cu"



// assume the total number of boundary pixel is less than 10000 (it is around 1000-2000 for my case)
#define MAX_NUM_BOUND_PIXELS 5000   

using namespace aortawarp; 


/*  main function to calculate sdf  */
// this is called in mexFunction
// output: SDF: (500*500); FX/FY: gradient image of X/Y direction; NV_theta: theta
// input: seg_img: binary image (500*500) from unet
void main_sdf_cuda(float *SDF, float *FX, float *FY, float *NV_theta,     /* pure debug */  float *obserID, float *BoundID, float *BoundPixelx, float *BoundPixely,
    bool *seg_img, unsigned int grid_dim_x, unsigned int grid_dim_y){


    /* some configuration for SDF   */
    // Voxel grid parameters (change these to change voxel grid resolution, etc.)
   float voxel_grid_origin_x = SDF_ORIGIN_X; // Location of voxel grid origin in base frame camera coordinates
   float voxel_grid_origin_y = SDF_ORIGIN_Y;
   float voxel_size = VOXEL_SIZE;          //todo: currently, we only deal with voxel_size as 1
   float trunc_margin = TRUNC_MARGIN;      // actually use SDF, not tsdf

   // also store theta threshold, this is for bring more robust
   float theta_threshold_innerproduct = THETA_THRESHOLD_INNER;




    // Load binarized segmentation to gpu
    bool * gpu_seg;                                       // store binarized image in gpu
    cudaMalloc(&gpu_seg,         grid_dim_x*grid_dim_y*sizeof(bool));
    cudaMemcpy(gpu_seg, seg_img, grid_dim_x*grid_dim_y*sizeof(bool), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());


    /** 01. calculate 2d boundary from segmentation  **/
    // Initialize matrix to store boundary segmentation (0 for not a boundary, 1 for is a boundary)
    // bool * bound_seg = new bool[grid_dim_x * grid_dim_y];    // store boundary as a binarized image in cpu
    // Load variables to GPU memory
    bool * gpu_bound_seg;                                    // store boundary image in gpu
    cudaMalloc(&gpu_bound_seg,                grid_dim_x*grid_dim_y*sizeof(bool));
    // cudaMemcpy(gpu_bound_seg,  bound_seg, grid_dim_x*grid_dim_y*sizeof(bool), cudaMemcpyHostToDevice);


    // Initialize next index of each skeleton (for each 1 in gpu_boundary_segmentation, store the index of next boundary)
    // NextID2D * bound_next_id = new NextID2D[grid_dim_x*grid_dim_y];    // store boundary as a binarized image in cpu
    // Load variables to GPU memory
    NextID2D * gpu_bound_next_id;                                    // store boundary ccw index in gpu
    cudaMalloc(&gpu_bound_next_id,             grid_dim_x*grid_dim_y*sizeof(NextID2D));
    // cudaMemcpy(gpu_bound_next_id, bound_next_id, grid_dim_x*grid_dim_y*sizeof(NextID2D), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    // calculate boundary from binarilized image
    MooreNeighbourBoundary <<< grid_dim_x, grid_dim_y >>> (gpu_bound_seg,  gpu_seg, grid_dim_x, grid_dim_y);


    /*  02 remove zero element from boundary id  */
    // store boundary as a binarized image in cpu
    bool * bound_seg = new bool[grid_dim_x * grid_dim_y];
    cudaMemcpy(bound_seg, gpu_bound_seg, grid_dim_x * grid_dim_y * sizeof(bool), cudaMemcpyDeviceToHost);
    
    // rewrite boundary id from boundary
    unsigned int * bound_id_raw = new unsigned int[MAX_NUM_BOUND_PIXELS];   // initialize an array to store the compact boundary
    memset(bound_id_raw, 0, MAX_NUM_BOUND_PIXELS*sizeof(unsigned int));     // initialize to zero. but actually no need to do this. 
    BoundaryIDRaw(bound_id_raw, MAX_NUM_BOUND_PIXELS, bound_seg, grid_dim_x, grid_dim_y);  

    // rewrite boundary id and boundary coordinate using the actual boundary size
    unsigned int num_bound = bound_id_raw[0];   // actual number of boundary;
    unsigned int * bound_id_stack = new unsigned int[num_bound];   // initialize an array to store the actual boundary
    float2 * bound_pixel_stack = new float2[num_bound];            // store boundary pixels in cpu
    StackBoundaryIDAndPixel(bound_id_stack, bound_pixel_stack, bound_id_raw, num_bound, voxel_grid_origin_x, grid_dim_x);

    // put bound_id_stack, bound_pixel_stack into gpu
    unsigned int * gpu_bound_id_stack;
    float2 * gpu_bound_pixel_stack;
    cudaMalloc(&gpu_bound_id_stack,  num_bound*sizeof(unsigned int));
    cudaMalloc(&gpu_bound_pixel_stack,  num_bound*sizeof(float2));
    cudaMemcpy(gpu_bound_id_stack,  bound_id_stack, num_bound*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_bound_pixel_stack,  bound_pixel_stack, num_bound*sizeof(float2), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());


    //todo there must be better way to remove zero element from boundary. 
    // the method I used here is too slow. 


    // Sort order of boundary in GPU.
    //the inner hole will also be considered
    SortBoundary <<< grid_dim_x, grid_dim_y >>> (gpu_bound_next_id,  gpu_bound_seg, gpu_seg, grid_dim_x, grid_dim_y);



    /** 03. calculate 2d normal vector  **/
    // Initialize boundary segmentation (0 for not a boundary, 1 for is a boundary)
    // float2 * bound_normalvector = new float2[grid_dim_x * grid_dim_y];    // store boundary as a binarized image in cpu
    // Load variables to GPU memory
    float2 * gpu_bound_normalvector;                                    // store boundary image in gpu
    cudaMalloc(&gpu_bound_normalvector,                        grid_dim_x * grid_dim_y * sizeof(float2));
    // cudaMemcpy(gpu_bound_normalvector,  bound_normalvector, grid_dim_x * grid_dim_y * sizeof(float2), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    // calculate normal vector
    LineNormal2D <<< grid_dim_x, grid_dim_y >>> (gpu_bound_normalvector,  gpu_bound_next_id, gpu_bound_seg, grid_dim_x, grid_dim_y);



    
    /** 04. sdf of boundary and normal vector  **/

    // Load variables to GPU memory
    float * gpu_voxel_grid_TSDF;               // store distance
    cudaMalloc(&gpu_voxel_grid_TSDF,         grid_dim_x * grid_dim_y * sizeof(float));
    // cudaMemcpy(gpu_voxel_grid_TSDF, voxel_grid_TSDF, grid_dim_x * grid_dim_y * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());
    float * gpu_voxel_grid_normalvector;       // store theta of normal vector
    cudaMalloc(&gpu_voxel_grid_normalvector, grid_dim_x * grid_dim_y * sizeof(float));
    // cudaMemcpy(gpu_voxel_grid_normalvector, voxel_grid_normalvector, voxel_grid_dim_x * voxel_grid_dim_y * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());
    

    // // debug
    int * gpu_voxel_grid_obserID;          // store observation id
    cudaMalloc(&gpu_voxel_grid_obserID,    grid_dim_x * grid_dim_y * sizeof(int));
    checkCUDA(__LINE__, cudaGetLastError());


    // // Calculate TSDF in GPU
    CalculateTSDF <<< grid_dim_x, grid_dim_y >>> (
        gpu_voxel_grid_TSDF, gpu_voxel_grid_normalvector,    /*debug*/ gpu_voxel_grid_obserID,             // output
        voxel_grid_origin_x, voxel_grid_origin_y, voxel_size, trunc_margin, theta_threshold_innerproduct,  // tsdf config
        gpu_bound_id_stack, gpu_bound_pixel_stack, gpu_bound_normalvector, num_bound, grid_dim_x, grid_dim_y);


    /** 05. sdf gradient  **/
    // Initialize sdf gradient
    float2 * gpu_voxel_grid_gradient;                                    // store boundary image in gpu
    cudaMalloc(&gpu_voxel_grid_gradient,                        grid_dim_x * grid_dim_y * sizeof(float2));
    checkCUDA(__LINE__, cudaGetLastError());

    Gradient2D <<< grid_dim_x, grid_dim_y >>> (gpu_voxel_grid_gradient,  // output
                                               gpu_voxel_grid_TSDF, voxel_grid_origin_x, voxel_grid_origin_y, voxel_size, grid_dim_x, grid_dim_y);


    // output
    float * voxel_grid_TSDF = new float[grid_dim_x * grid_dim_y];           // store sdf
    float * voxel_grid_normalvector = new float[grid_dim_x * grid_dim_y];   // store cos(normalvector)
    float2 * voxel_grid_gradient = new float2[grid_dim_x * grid_dim_y];     // store sdf gradient
    cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, grid_dim_x * grid_dim_y * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(voxel_grid_normalvector, gpu_voxel_grid_normalvector, grid_dim_x * grid_dim_y * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(voxel_grid_gradient, gpu_voxel_grid_gradient, grid_dim_x * grid_dim_y * sizeof(float2), cudaMemcpyDeviceToHost);
    checkCUDA(__LINE__, cudaGetLastError());

    for (int i = 0; i < grid_dim_x * grid_dim_y; ++i){
        SDF[i] = voxel_grid_TSDF[i];
        FX[i] =  voxel_grid_gradient[i].y;        // FX, FY is inverse           
        FY[i] =  voxel_grid_gradient[i].x;
        NV_theta[i] = voxel_grid_normalvector[i];
    }

    
    // output corresponding id for debug
    unsigned int * voxel_grid_obserID = new unsigned int[grid_dim_x * grid_dim_y];    // store boundary as a binarized image in cpu
    cudaMemcpy(voxel_grid_obserID, gpu_voxel_grid_obserID, grid_dim_x * grid_dim_y * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid_dim_x * grid_dim_y; ++i){
        obserID[i] = (float) voxel_grid_obserID[i];
    }

    // output following for debug
    for (int i = 0; i < num_bound; ++i){
        BoundID[i] = bound_id_stack[i];
        BoundPixelx[i] = bound_pixel_stack[i].x;
        BoundPixely[i] = bound_pixel_stack[i].y;
    }
    


    // free mem
    cudaFree(gpu_seg);
    cudaFree(gpu_bound_seg);
    cudaFree(gpu_bound_next_id);
    cudaFree(gpu_bound_id_stack);
    
    cudaFree(gpu_bound_pixel_stack);
    cudaFree(gpu_bound_normalvector);
    cudaFree(gpu_voxel_grid_TSDF);
    cudaFree(gpu_voxel_grid_normalvector);
    cudaFree(gpu_voxel_grid_obserID);
    cudaFree(gpu_voxel_grid_gradient);
}