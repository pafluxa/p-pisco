#ifndef __DETMUELLERH__
#define __DETMUELLERH__

#include <cuComplex.h>

__device__ void
det_conjugate_jones_matrix( cuFloatComplex J_det[2][2], cuFloatComplex J_det_conj[2][2] )
{

    J_det_conj[0][0] = cuConjf(J_det[0][0]);
    J_det_conj[0][1] = cuConjf(J_det[0][1]);
    J_det_conj[1][0] = cuConjf(J_det[1][0]);
    J_det_conj[1][1] = cuConjf(J_det[1][1]);
}

__device__ void
det_compute_jones_matrix( float alpha, cuFloatComplex J_det[2][2] )
{ 
    float sa = sinf(alpha);
    float ca = cosf(alpha);
    float c2 = ca*ca;
    float s2 = sa*sa;
    float sc = sa*ca;

    J_det[0][0] = make_cuFloatComplex( c2, 0.0f );
    J_det[0][1] = make_cuFloatComplex( sc, 0.0f );
    J_det[1][0] = make_cuFloatComplex( sc, 0.0f );
    J_det[1][1] = make_cuFloatComplex( s2, 0.0f );
}

__device__ void
det_compute_mueller_matrix( float alpha, float M_det[4][4] )
{
    // First, compute Jones Matrix for given arguments
    cuFloatComplex J_det[2][2], J_det_conj[2][2];
    det_compute_jones_matrix( alpha, J_det );
    // Compute the conjugate of Jones Matrix
    det_conjugate_jones_matrix( J_det, J_det_conj );

    // Compute kron product J_det (x) J_det^*
    cuFloatComplex J_kron[4][4];
    
    // This is a really ugly implementation, but saves a lot of for loops!
    // First block
    J_kron[0][0] = cuCmulf( J_det[0][0] , J_det_conj[0][0]);
    J_kron[0][1] = cuCmulf( J_det[0][0] , J_det_conj[0][1]); 
    J_kron[1][0] = cuCmulf( J_det[0][0] , J_det_conj[1][0]);
    J_kron[1][1] = cuCmulf( J_det[0][0] , J_det_conj[1][1]);
    // Second block
    J_kron[0][2] = cuCmulf( J_det[0][1] , J_det_conj[0][0]);
    J_kron[0][3] = cuCmulf( J_det[0][1] , J_det_conj[0][1]);
    J_kron[1][2] = cuCmulf( J_det[0][1] , J_det_conj[1][0]);
    J_kron[1][3] = cuCmulf( J_det[0][1] , J_det_conj[1][1]);
    // Third block
    J_kron[2][0] = cuCmulf( J_det[1][0] , J_det_conj[0][0]);
    J_kron[2][1] = cuCmulf( J_det[1][0] , J_det_conj[0][1]);
    J_kron[3][0] = cuCmulf( J_det[1][0] , J_det_conj[1][0]);
    J_kron[3][1] = cuCmulf( J_det[1][0] , J_det_conj[1][1]);
    // Fourth block
    J_kron[2][2] = cuCmulf( J_det[1][1] , J_det_conj[0][0]);
    J_kron[2][3] = cuCmulf( J_det[1][1] , J_det_conj[0][1]);
    J_kron[3][2] = cuCmulf( J_det[1][1] , J_det_conj[1][0]);
    J_kron[3][3] = cuCmulf( J_det[1][1] , J_det_conj[1][1]);
    
    // Define Matrix A 
    cuFloatComplex A    [4][4];
    A[0][0] = make_cuFloatComplex( 1.0f, 0.0f );
    A[0][1] = make_cuFloatComplex( 0.0f, 0.0f );
    A[0][2] = make_cuFloatComplex( 0.0f, 0.0f );
    A[0][3] = make_cuFloatComplex( 1.0f, 0.0f );
    
    A[1][0] = make_cuFloatComplex( 1.0f, 0.0f );
    A[1][1] = make_cuFloatComplex( 0.0f, 0.0f );
    A[1][2] = make_cuFloatComplex( 0.0f, 0.0f );
    A[1][3] = make_cuFloatComplex(-1.0f, 0.0f );
    
    A[2][0] = make_cuFloatComplex( 0.0f, 0.0f );
    A[2][1] = make_cuFloatComplex( 1.0f, 0.0f );
    A[2][2] = make_cuFloatComplex( 1.0f, 0.0f );
    A[2][3] = make_cuFloatComplex( 0.0f, 0.0f );
    
    A[3][0] = make_cuFloatComplex( 0.0f, 0.0f );
    A[3][1] = make_cuFloatComplex( 0.0f, 1.0f );
    A[3][2] = make_cuFloatComplex( 0.0f,-1.0f );
    A[3][3] = make_cuFloatComplex( 0.0f, 0.0f );

    // Define inverse of matrix A
    cuFloatComplex A_inv[4][4];
    A_inv[0][0] = make_cuFloatComplex( 0.5f, 0.0f );
    A_inv[0][1] = make_cuFloatComplex( 0.5f, 0.0f );
    A_inv[0][2] = make_cuFloatComplex( 0.0f, 0.0f );
    A_inv[0][3] = make_cuFloatComplex( 0.0f, 0.0f );
    
    A_inv[1][0] = make_cuFloatComplex( 0.0f, 0.0f );
    A_inv[1][1] = make_cuFloatComplex( 0.0f, 0.0f );
    A_inv[1][2] = make_cuFloatComplex( 0.5f, 0.0f );
    A_inv[1][3] = make_cuFloatComplex( 0.0f,-0.5f );
    
    A_inv[2][0] = make_cuFloatComplex( 0.0f, 0.0f );
    A_inv[2][1] = make_cuFloatComplex( 0.0f, 0.0f );
    A_inv[2][2] = make_cuFloatComplex( 0.5f, 0.0f );
    A_inv[2][3] = make_cuFloatComplex( 0.0f, 0.5f );
    
    A_inv[3][0] = make_cuFloatComplex( 0.5f, 0.0f );
    A_inv[3][1] = make_cuFloatComplex(-0.5f, 0.0f );
    A_inv[3][2] = make_cuFloatComplex( 0.0f, 0.0f );
    A_inv[3][3] = make_cuFloatComplex( 0.0f, 0.0f );
    
    // Temporal matrix to store J_kron x A_inv
    cuFloatComplex tmp[4][4];
    // Compute J_kron x A_inv; for only this time, I'll use for loops to do the matrix-matrix multiplication.
    int c,d,k;
    cuFloatComplex zero = make_cuFloatComplex( 0.0f, 0.0f ); 
    cuFloatComplex sum  = zero;
    for( c=0; c < 4; c++ ){
        for( d=0; d < 4; d++ ){
            for( k=0; k < 4; k++ ){
                sum = cuCaddf( sum, cuCmulf( J_kron[c][k], A_inv[k][d] ) );
            }
            
            tmp[c][d] = sum;
            sum = zero;
        }
    }
    // Compute M as A x tmp
    cuFloatComplex M[4][4];
    sum = zero;
    for( c=0; c < 4; c++ ){
        for( d=0; d < 4; d++ ){
            for( k=0; k < 4; k++ ){
                sum = cuCaddf( sum, cuCmulf( A[c][k], tmp[k][d] ) );
            }
            
            M[c][d] = sum;
            sum = zero;
        }
    }

    for( c=0; c < 4; c++ )
    for( d=0; d < 4; d++ )
        M_det[c][d] = cuCrealf(M[c][d]);
}

#endif
