#ifndef __COMPLEXLA_H__
#define __COMPLEXLA_H__

#include <cuComplex.h>

__device__ void
complexla_matrix_times_matrix( cuDoubleComplex A[4][4], cuDoubleComplex B[4][4], cuDoubleComplex R[4][4] )
{
    cuDoubleComplex sum = make_cuDoubleComplex( 0,0 );
    int i,j,k;
    for (i = 0; i <= 2; i++) {
        for (j = 0; j <= 2; j++) {
            sum = make_cuDoubleComplex( 0,0 );
            for (k = 0; k <= 2; k++) {
                sum = cuCadd( sum,  cuCmul( A[i][k] , B[k][j] ) );
            }

        R[i][j] = sum;

        }
    }
}

__device__ void
complexla_matrix_times_vector ( cuDoubleComplex A[4][4], cuDoubleComplex S[4], cuDoubleComplex R[4] )
{
    
    cuDoubleComplex x = cuCadd( cuCadd( cuCmul( A[0][0], S[0] ), cuCmul( A[0][1], S[1] ) ), 
                                cuCadd( cuCmul( A[0][2], S[2] ), cuCmul( A[0][3], S[3] ) ) );
    
    cuDoubleComplex y = cuCadd( cuCadd( cuCmul( A[1][0], S[0] ), cuCmul( A[1][1], S[1] ) ), 
                                cuCadd( cuCmul( A[1][2], S[2] ), cuCmul( A[1][3], S[3] ) ) );

    cuDoubleComplex z = cuCadd( cuCadd( cuCmul( A[2][0], S[0] ), cuCmul( A[2][1], S[1] ) ), 
                                cuCadd( cuCmul( A[2][2], S[2] ), cuCmul( A[2][3], S[3] ) ) );
    
    cuDoubleComplex w = cuCadd( cuCadd( cuCmul( A[3][0], S[0] ), cuCmul( A[3][1], S[1] ) ), 
                                cuCadd( cuCmul( A[3][2], S[2] ), cuCmul( A[3][3], S[3] ) ) );

    R[0] = x;
    R[1] = y;
    R[2] = z;
    R[3] = w;
}

#endif
