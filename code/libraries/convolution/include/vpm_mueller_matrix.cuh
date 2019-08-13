#ifndef __VPMMUELLERH__
#define __VPMMUELLERH__

#include <cuComplex.h>

__device__ void
vpm_conjugate_jones_matrix( cuFloatComplex J_vpm[2][2], cuFloatComplex J_vpm_conj[2][2] )
{

    J_vpm_conj[0][0] = cuConjf(J_vpm[0][0]);
    J_vpm_conj[0][1] = cuConjf(J_vpm[0][1]);
    J_vpm_conj[1][0] = cuConjf(J_vpm[1][0]);
    J_vpm_conj[1][1] = cuConjf(J_vpm[1][1]);
}

__device__ void
vpm_compute_jones_matrix( float d, float k, float theta, float phi, cuFloatComplex J_vpm[2][2] )
{
    cuFloatComplex G_TM_para = make_cuFloatComplex( -1.0f, 0.0f );
    cuFloatComplex G_TE_para = make_cuFloatComplex( -1.0f, 0.0f );
    cuFloatComplex G_TM_perp = make_cuFloatComplex(-cosf(2*d*k*cos(theta)),-sinf(2*d*k*cos(theta)) );
    cuFloatComplex G_TE_perp = make_cuFloatComplex(-cosf(2*d*k*cos(theta)),-sinf(2*d*k*cos(theta)) );

    cuFloatComplex c2 = make_cuFloatComplex( cosf(phi)*cosf(phi), 0.0f );
    cuFloatComplex s2 = make_cuFloatComplex( sinf(phi)*sinf(phi), 0.0f );
    cuFloatComplex sc = make_cuFloatComplex( sinf(phi)*cosf(phi), 0.0f );
   
    J_vpm[0][0] = cuCsubf( cuCmulf(G_TM_para, c2 ), cuCmulf( G_TM_perp, s2 ) );
    J_vpm[0][1] = cuCmulf( cuCaddf(G_TE_para,G_TE_perp), sc ); 
    J_vpm[1][0] = cuCmulf( cuCaddf(G_TM_para,G_TM_perp), sc ); 
    J_vpm[1][1] = cuCsubf( cuCmulf(G_TE_para, s2 ) , cuCmulf( G_TE_perp, c2 ) ); 
}

__device__ void
vpm_compute_mueller_matrix( float distance, float wave_number, float theta, float phi, float M_vpm[4][4] )
{
    // First, compute Jones Matrix for given arguments
    cuFloatComplex J_vpm[2][2], J_vpm_conj[2][2];
    vpm_compute_jones_matrix( distance, wave_number, theta, phi, J_vpm );
    // Compute the conjugate of Jones Matrix
    vpm_conjugate_jones_matrix( J_vpm, J_vpm_conj );

    // Compute kron product J_vpm (x) J_vpm^*
    cuFloatComplex J_kron[4][4];
    
    // This is a really ugly implementation, but saves a lot of for loops!
    // First block
    J_kron[0][0] = cuCmulf( J_vpm[0][0] , J_vpm_conj[0][0]);
    J_kron[0][1] = cuCmulf( J_vpm[0][0] , J_vpm_conj[0][1]); 
    J_kron[1][0] = cuCmulf( J_vpm[0][0] , J_vpm_conj[1][0]);
    J_kron[1][1] = cuCmulf( J_vpm[0][0] , J_vpm_conj[1][1]);
    // Second block
    J_kron[0][2] = cuCmulf( J_vpm[0][1] , J_vpm_conj[0][0]);
    J_kron[0][3] = cuCmulf( J_vpm[0][1] , J_vpm_conj[0][1]);
    J_kron[1][2] = cuCmulf( J_vpm[0][1] , J_vpm_conj[1][0]);
    J_kron[1][3] = cuCmulf( J_vpm[0][1] , J_vpm_conj[1][1]);
    // Third block
    J_kron[2][0] = cuCmulf( J_vpm[1][0] , J_vpm_conj[0][0]);
    J_kron[2][1] = cuCmulf( J_vpm[1][0] , J_vpm_conj[0][1]);
    J_kron[3][0] = cuCmulf( J_vpm[1][0] , J_vpm_conj[1][0]);
    J_kron[3][1] = cuCmulf( J_vpm[1][0] , J_vpm_conj[1][1]);
    // Fourth block
    J_kron[2][2] = cuCmulf( J_vpm[1][1] , J_vpm_conj[0][0]);
    J_kron[2][3] = cuCmulf( J_vpm[1][1] , J_vpm_conj[0][1]);
    J_kron[3][2] = cuCmulf( J_vpm[1][1] , J_vpm_conj[1][0]);
    J_kron[3][3] = cuCmulf( J_vpm[1][1] , J_vpm_conj[1][1]);
    
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
        M_vpm[c][d] = cuCrealf(M[c][d]);
}

#endif
