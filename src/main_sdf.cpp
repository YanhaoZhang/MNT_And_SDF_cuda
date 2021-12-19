// ---------------------------------------------------------
// Author: Yanhao Zhang, University of Technoligy Sydney, 2021.03.21
// main function to calculate sdf from unet segmentation
// ---------------------------------------------------------


// #pragma once
#include "mex.h"
#include "main_sdf_cuda.h"


/*  link function for matlab  */
// input: binary image (500*500) from unet. binarization is easily done by matlab. 
void mexFunction( int nout, mxArray *pout[], int nin, const mxArray *pin[])
{
    // Get the pointers to the data. 
    // All of the memory allocation stuff is done in 
    // mexFunction including the data checking.
    if (!mxIsLogical(pin[0])){
        mexErrMsgIdAndTxt( "MATLAB:matrix_mult:inputerror",
                "Input matrices must be boolean precision.");
    }
    
    // binarized image: 1 for object, 0 for background
    bool *seg_img = (bool*)mxGetData(pin[0]);
    
    // image height and width
    unsigned int img_h = mxGetM(pin[0]);
    unsigned int img_w = mxGetN(pin[0]);
    
       
    // set up outputs: pout[0] sdf, pout[1] FX, pout[2] FY, pout[3] NV_theta.
    // details in sdf_contour.m
    pout[0] = mxCreateNumericMatrix((mwSize) img_h, (mwSize) img_w, mxSINGLE_CLASS, mxREAL);
    pout[1] = mxCreateNumericMatrix((mwSize) img_h, (mwSize) img_w, mxSINGLE_CLASS, mxREAL);
    pout[2] = mxCreateNumericMatrix((mwSize) img_h, (mwSize) img_w, mxSINGLE_CLASS, mxREAL);
    pout[3] = mxCreateNumericMatrix((mwSize) img_h, (mwSize) img_w, mxSINGLE_CLASS, mxREAL);
    

    // to store the output by cuda function
    float *SDF = (float*)mxGetData(pout[0]);
    float *FY = (float*)mxGetData(pout[1]);          // voxel_grid_gradient_y is gradient along col axis (left2right) 
    float *FX = (float*)mxGetData(pout[2]);          // voxel_grid_gradient_x is gradient along row axis (up2down)
    float *NV_theta = (float*)mxGetData(pout[3]);    // cos(normal vector)



 // for pure debug

    pout[4] = mxCreateNumericMatrix((mwSize) img_h, (mwSize) img_w, mxSINGLE_CLASS, mxREAL);
    pout[5] = mxCreateNumericMatrix((mwSize) 5000, (mwSize) 1, mxSINGLE_CLASS, mxREAL); 
    pout[6] = mxCreateNumericMatrix((mwSize) 5000, (mwSize) 1, mxSINGLE_CLASS, mxREAL);  
    pout[7] = mxCreateNumericMatrix((mwSize) 5000, (mwSize) 1, mxSINGLE_CLASS, mxREAL);

    float *obserID = (float*)mxGetData(pout[4]);
    float *BoundID = (float*)mxGetData(pout[5]); 
    float *BoundPixelx = (float*)mxGetData(pout[6]);
    float *BoundPixely = (float*)mxGetData(pout[7]);

    main_sdf_cuda(SDF, FX, FY, NV_theta, /* just for debug */ obserID, BoundID, BoundPixelx, BoundPixely, // output
                seg_img, img_h, img_w);   // input
}


