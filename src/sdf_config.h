#ifndef CU_MEX_SDF_CONTOUR
#define CU_MEX_SDF_CONTOUR


#include <math.h> 



/*  configuration for SDF  */
// Voxel grid parameters (change these to change voxel grid resolution, etc.)
#define SDF_ORIGIN_X  1.0f      // Location of voxel grid origin in base frame camera coordinates
#define SDF_ORIGIN_Y  1.0f
#define VOXEL_SIZE  1.0f        //todo: currently, we only deal with voxel_size as 1
#define TRUNC_MARGIN  500.0f    // actually use SDF, not tsdf



// also store theta threshold, this is for bring more robust
#define THETA_THRESHOLD  M_PI/3.0f
#define THETA_THRESHOLD_INNER  std::cos(THETA_THRESHOLD)

  


#endif