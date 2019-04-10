#ifndef __BILINEARINTERPH
#define __BILINEARINTERPH

#include <cuComplex.h>

__device__ cuDoubleComplex
binterp_cbilinear_interpolation( cuDoubleComplex beam[], int input_beam_size, double grid_size, int grid_nx, double dx, double dy )
{
    cuDoubleComplex zero = {0,0};
    double RMAX   = grid_size/2.0;
    double RDELTA = (RMAX / grid_nx/2 );
 
    int ipix, jpix;
    double dxinc, dyinc;
    cuDoubleComplex bym, byp, bxm, bxp, b;

    if(fabs(dx) >= RMAX)
        return zero;

    else if (fabs(dy) >= RMAX)
        return zero;

    // Compute neighbourd pixels
    
    ipix = (int)( ( dx + grid_size/2.0 ) * ( grid_nx/grid_size ) );
    jpix = (int)( ( dy + grid_size/2.0 ) * ( grid_nx/grid_size ) );

    dyinc = fmod(dy, RDELTA) / RDELTA;
    dxinc = fmod(dx, RDELTA) / RDELTA;
    
    bym = beam[ ipix * grid_nx + jpix ];  
    byp = beam[ ipix * grid_nx + jpix + 1];  
    bxm = cuCadd( cuCmul( cuCsub(byp , bym) , make_cuDoubleComplex(dyinc,0) ) , bym );
    
    bym = beam[ (ipix+1)*grid_nx + jpix   ];  
    byp = beam[ (ipix+1)*grid_nx + jpix+1 ];  
    bxp = cuCadd( cuCmul( cuCsub(byp , bym) , make_cuDoubleComplex(dyinc,0) ) , bym );
    
    b   = cuCadd( cuCmul( cuCsub(bxp , bxm) , make_cuDoubleComplex(dxinc,0) ) , bxm );
    
    return b;
}   

#endif
