#ifndef __BILINEARINTERPH
#define __BILINEARINTERPH

#include <cuComplex.h>

__device__ cuDoubleComplex get_pixel
(   
    cuDoubleComplex beam[], 
    unsigned int x, 
    unsigned int y, 
    unsigned int grid_size 
)
{
    return beam[ y*grid_size + x ];
}
__device__ cuDoubleComplex
cbilinear_interpolation( cuDoubleComplex beam[], double grid_side, int grid_size, double x, double y )
{
    // Check that x and y don't fall outside the evaluation grid
    cuDoubleComplex zero = {0,0};
    if( fabs(x) >= grid_side/2.0 )
        return zero;
    if( fabs(y) >= grid_side/2.0 )
        return zero;
     
    double grid_dx   = grid_size/(grid_side);
    double grid_rmax = grid_side/2.0;
  
    // Compute the indexes of x,y 
    int idx  = (int)( ( grid_dx * (x + grid_rmax) ) ); 
    int idy  = (int)( ( grid_dx * (y + grid_rmax) ) ); 

    // Compute corner coordinates
    double x1 = -grid_rmax + (idx+0) / grid_dx;
    double x2 = -grid_rmax + (idx+1) / grid_dx;
    double y1 = -grid_rmax + (idy+0) / grid_dx;
    double y2 = -grid_rmax + (idy+1) / grid_dx;
   
    // Compute coefficients for linear interpolation in x
    cuDoubleComplex a  = { (x2-x )/(x2-x1), 0.0 };
    cuDoubleComplex b  = { ( x-x1)/(x2-x1), 0.0 };
    
    // Compute the values of the beam at the corners
    cuDoubleComplex Q11 = get_pixel( beam, idx+0, idy+0, grid_size ); 
    cuDoubleComplex Q21 = get_pixel( beam, idx+1, idy+0, grid_size );
    cuDoubleComplex Q22 = get_pixel( beam, idx+1, idy+1, grid_size );
    cuDoubleComplex Q12 = get_pixel( beam, idx+0, idy+1, grid_size );
    
    // Assemble linear interpolated values in x
    cuDoubleComplex fxy1 = cuCadd( cuCmul( a, Q11 ) , cuCmul( b, Q21 ) );
    cuDoubleComplex fxy2 = cuCadd( cuCmul( a, Q12 ) , cuCmul( b, Q22 ) );

    // Assemble interpolated value on y
    cuDoubleComplex   c  = { (y2 -  y)/(y2 - y1), 0.0 };
    cuDoubleComplex   d  = { ( y - y1)/(y2 - y1), 0.0 };

    cuDoubleComplex fxy  = cuCadd( cuCmul( c, fxy1) , cuCmul( d, fxy2) );
    
    return fxy;
    //return Q11;
}   


#endif
