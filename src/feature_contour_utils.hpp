// ---------------------------------------------------------
// Author: Yanhao Zhang, University of Technoligy Sydney, 2021.03.21
// boundary tracing from a binarize image
// ---------------------------------------------------------

// #pragma once

#ifndef CU_MEX_FEATURE_CONTOUR_UTILS
#define CU_MEX_FEATURE_CONTOUR_UTILS


#include <vector>


/*  struct to store id  */
// not used
struct grid_id {
  int x;
  int y;
} ;
  
 /*  struct to store id  */ 
 struct NextID2D {
  int idx_ccw = -1;
  int idy_ccw = -1;
  int idx_cw = -1;
  int idy_cw = -1;
};

#endif
