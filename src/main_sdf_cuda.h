
#ifndef CU_MEX_SDF_CUDA
#define CU_MEX_SDF_CUDA







/*  main function to calculate sdf  */
// this is called in mexFunction
// output: SDF: (500*500); FX/FY: gradient image of X/Y direction; NV_theta: theta
// input: seg_img: binary image (500*500) from unet
void main_sdf_cuda(float *SDF, float *FX, float *FY, float *NV_theta,     /* pure debug */  float *obserID, float *BoundID, float *BoundPixelx, float *BoundPixely,
                bool *seg_img, unsigned int grid_dim_x, unsigned int grid_dim_y);

#endif
