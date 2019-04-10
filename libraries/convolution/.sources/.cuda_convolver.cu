#include <healpix_utils.cuh>
#include <sky_coords.cuh>
#include <bilinear_interpolation.cuh>
#include <complex_la.h>

#include <cuda_utils.h>

#include <chealpix.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <omp.h>
#include <cuda.h>

#define CUDA_BLOCK_SIZE   128
#define CUDA_NUM_BLOCKS   256

// Instantiate texture space for sky maps
texture<float4, 1, cudaReadModeElementType> tex_IQUV;


//######################################################################################################
// Allocate space in GPU
//######################################################################################################
// Beam pointing bufferse
double* gpu_beam_grid_dx;
double* gpu_beam_grid_dy;

int*    gpu_num_pixels_in_grid;
int*    gpu_eval_grid_pixels;
double* gpu_eval_grid_dx;
double* gpu_eval_grid_dy;

double *gpu_ra;
double *gpu_dec;
double *gpu_pa;

double *gpu_tod;

float4* gpu_IQUV;
float4* host_IQUV;

//######################################################################################################
// Mueller Matrices in complex form
//######################################################################################################
cuDoubleComplex *gpu_M_TT,  *gpu_M_TP,  *gpu_M_TPs,  *gpu_M_TV,
                *gpu_M_PT,  *gpu_M_PP,  *gpu_M_PPs,  *gpu_M_PV,
                *gpu_M_PsT, *gpu_M_PsP, *gpu_M_PsPs, *gpu_M_PsV,
                *gpu_M_VT , *gpu_M_VP , *gpu_M_VPs , *gpu_M_VV;

//######################################################################################################
// Copy in the host to make the copy easier
//######################################################################################################
cuDoubleComplex *h_M_TT,  *h_M_TP,  *h_M_TPs,  *h_M_TV,
                *h_M_PT,  *h_M_PP,  *h_M_PPs,  *h_M_PV,
                *h_M_PsT, *h_M_PsP, *h_M_PsPs, *h_M_PsV,
                *h_M_VT , *h_M_VP , *h_M_VPs , *h_M_VV;

void texturize_maps
(
    int input_map_size, 
    float I[], float Q[], float U[], float V[]
)
{
    gpu_error_check( cudaMallocHost( (void**)&host_IQUV, sizeof( float4 ) * input_map_size ) );               

    gpu_error_check( cudaMalloc( (void**)&gpu_IQUV, sizeof( float4 ) * input_map_size ) );               
    
    #pragma omp parallel for
    for( int pix=0; pix < input_map_size; pix++ )                                                             
    {                                                                                                         
        host_IQUV[pix].x = I[ pix ];                                                                          
        host_IQUV[pix].y = Q[ pix ];                                                                          
        host_IQUV[pix].z = U[ pix ];                                                                          
        host_IQUV[pix].w = V[ pix ];                                                                          
                                                                                                              
    }                                                                                                         
                                                                                                              
    //######################################################################################################  
    // Transfer maps to GPU                                                                                   
    //######################################################################################################  
    gpu_error_check(                                                                                          
        cudaMemcpy(                                                                                           
        gpu_IQUV,            // destination                                                                   
        host_IQUV,           // source                                                                        
        input_map_size * sizeof(float4),   // amount of bytes                                                 
        cudaMemcpyHostToDevice ) );        // specify direction of transfer                                   
    //######################################################################################################  
    // Bind Texture                                                                                           
    gpu_error_check( cudaBindTexture(0, tex_IQUV, gpu_IQUV, input_map_size * sizeof(float4) ) );              
    //###################################################################################################### 
}

void allocate_tod
(
    int nsamples 
)
{
    gpu_error_check( cudaMalloc( (void**)&gpu_tod , sizeof( double ) * nsamples ) );
}
    

void transfer_pointing_streams
(
    int nsamples,
    double ra[], double dec[], double pa[]
)
{
    // Allocate GPU buffers
    gpu_error_check( cudaMalloc( (void**)&gpu_ra , sizeof( double ) * nsamples ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_dec, sizeof( double ) * nsamples ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_pa , sizeof( double ) * nsamples ) );

    // Transfer beam evaluation grid
    gpu_error_check( cudaMemcpy( gpu_ra ,  ra, sizeof( double ) * nsamples, cudaMemcpyHostToDevice) );
    gpu_error_check( cudaMemcpy( gpu_dec, dec, sizeof( double ) * nsamples, cudaMemcpyHostToDevice) );
    gpu_error_check( cudaMemcpy( gpu_pa ,  pa, sizeof( double ) * nsamples, cudaMemcpyHostToDevice) );

}

void build_and_transfer_eval_grid
(
   int nsamples, int npix_max, int num_pixels[], int evalgrid_pixels[]
)
{
    // Allocate GPU buffers
    gpu_error_check( cudaMalloc( (void**)&gpu_num_pixels_in_grid, sizeof( double ) * nsamples ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_eval_grid_pixels, sizeof( double ) * nsamples * npix_max) );
    
    // Transfer eval evaluation grid
    gpu_error_check( 
        cudaMemcpy( gpu_num_pixels_in_grid, num_pixels, sizeof( double ) * nsamples, cudaMemcpyHostToDevice) );
    gpu_error_check( 
        cudaMemcpy( gpu_eval_grid_pixels, evalgrid_pixels, sizeof( double ) * nsamples * npix_max, cudaMemcpyHostToDevice) );
}

void allocate_and_transfer_mueller_beams
(
    int nx,

    double reM_TT[],  double imM_TT[], 
    double reM_TP[],  double imM_TP[], 
    double reM_TPs[], double imM_TPs[], 
    double reM_TV[],  double imM_TV[], 
    
    double reM_PT[],  double imM_PT[], 
    double reM_PP[],  double imM_PP[], 
    double reM_PPs[], double imM_PPs[], 
    double reM_PV[],  double imM_PV[], 
    
    double reM_PsT[],  double imM_PsT[], 
    double reM_PsP[],  double imM_PsP[], 
    double reM_PsPs[], double imM_PsPs[], 
    double reM_PsV[],  double imM_PsV[], 
    
    double reM_VT[],  double imM_VT[], 
    double reM_VP[],  double imM_VP[], 
    double reM_VPs[], double imM_VPs[], 
    double reM_VV[],  double imM_VV[]
)
{
    
    int input_beam_size = nx*nx;

    //######################################################################################################
    // Allocate space on the GPU
    //######################################################################################################
    // First Row
    gpu_error_check( cudaMallocHost( (void**)&h_M_TT,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_TP,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_TPs,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_TV,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    // Second Row
    gpu_error_check( cudaMallocHost( (void**)&h_M_PT,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_PP,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_PPs,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_PV,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    // Third Row
    gpu_error_check( cudaMallocHost( (void**)&h_M_PsT,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_PsP,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_PsPs, sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_PsV,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    // Fourth Row
    gpu_error_check( cudaMallocHost( (void**)&h_M_VT,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_VP,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_VPs,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMallocHost( (void**)&h_M_VV,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    
    for( int pix=0; pix < nx*nx; pix++ )
    {
        h_M_TT  [ pix ].x = reM_TT  [ pix ];
        h_M_TP  [ pix ].x = reM_TP  [ pix ];
        h_M_TPs [ pix ].x = reM_TPs [ pix ];
        h_M_TV  [ pix ].x = reM_TV  [ pix ];

        h_M_PT  [ pix ].x = reM_PT  [ pix ];
        h_M_PP  [ pix ].x = reM_PP  [ pix ];
        h_M_PPs [ pix ].x = reM_PPs [ pix ];
        h_M_PV  [ pix ].x = reM_PV  [ pix ];

        h_M_PsT [ pix ].x = reM_PsT [ pix ];
        h_M_PsP [ pix ].x = reM_PsP [ pix ];
        h_M_PsPs[ pix ].x = reM_PsPs[ pix ];
        h_M_PsV [ pix ].x = reM_PsV [ pix ];

        h_M_VT  [ pix ].x = reM_VT  [ pix ];
        h_M_VP  [ pix ].x = reM_VP  [ pix ];
        h_M_VPs [ pix ].x = reM_VPs [ pix ];
        h_M_VV  [ pix ].x = reM_VV  [ pix ];

        h_M_TT  [ pix ].y = imM_TT  [ pix ];
        h_M_TP  [ pix ].y = imM_TP  [ pix ];
        h_M_TPs [ pix ].y = imM_TPs [ pix ];
        h_M_TV  [ pix ].y = imM_TV  [ pix ];

        h_M_PT  [ pix ].y = imM_PT  [ pix ];
        h_M_PP  [ pix ].y = imM_PP  [ pix ];
        h_M_PPs [ pix ].y = imM_PPs [ pix ];
        h_M_PV  [ pix ].y = imM_PV  [ pix ];

        h_M_PsT [ pix ].y = imM_PsT [ pix ];
        h_M_PsP [ pix ].y = imM_PsP [ pix ];
        h_M_PsPs[ pix ].y = imM_PsPs[ pix ];
        h_M_PsV [ pix ].y = imM_PsV [ pix ];

        h_M_VT  [ pix ].y = imM_VT  [ pix ];
        h_M_VP  [ pix ].y = imM_VP  [ pix ];
        h_M_VPs [ pix ].y = imM_VPs [ pix ];
        h_M_VV  [ pix ].y = imM_VV  [ pix ];
    }
    //######################################################################################################
    // Allocate space on the GPU
    //######################################################################################################
    // First Row
    gpu_error_check( cudaMalloc( (void**)&gpu_M_TT,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_TP,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_TPs,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_TV,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    // Second Row
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PT,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PP,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PPs,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PV,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    // Third Row
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PsT,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PsP,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PsPs, sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PsV,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    // Fourth Row
    gpu_error_check( cudaMalloc( (void**)&gpu_M_VT,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_VP,   sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_VPs,  sizeof( cuDoubleComplex ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_VV,   sizeof( cuDoubleComplex ) * input_beam_size ) );

    gpu_error_check( cudaMemcpy( gpu_M_TT,   h_M_TT,   sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_TP,   h_M_TP,   sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_TPs,  h_M_TPs,  sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_TV,   h_M_TV,   sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    // Second Row
    gpu_error_check( cudaMemcpy( gpu_M_PT,   h_M_PT,   sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PP,   h_M_PP,   sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PPs,  h_M_PPs,  sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PV,   h_M_PV,   sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    // Third Row
    gpu_error_check( cudaMemcpy( gpu_M_PsT,  h_M_PsT,  sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PsP,  h_M_PsP,  sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PsPs, h_M_PsPs, sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PsV,  h_M_PsV,  sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    // Fourth Row
    gpu_error_check( cudaMemcpy( gpu_M_VT,   h_M_VT ,  sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_VP,   h_M_VP ,  sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_VPs,  h_M_VPs,  sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_VV,   h_M_VV ,  sizeof( cuDoubleComplex ) * input_beam_size  , cudaMemcpyHostToDevice));
}

void free_everything()
{
    gpu_error_check( cudaFree( gpu_tod  ) );

    // Free pointing buffers
    gpu_error_check( cudaFree( gpu_ra   ) );
    gpu_error_check( cudaFree( gpu_dec  ) );
    gpu_error_check( cudaFree( gpu_pa   ) );

    cudaFree( gpu_eval_grid_dx );
    cudaFree( gpu_eval_grid_dy );
    
    cudaFree( gpu_beam_grid_dx );
    cudaFree( gpu_beam_grid_dy );
   
    cudaFree(gpu_IQUV);
    cudaUnbindTexture(tex_IQUV);
    cudaFreeHost( host_IQUV );

    cudaFreeHost( h_M_TT );
    cudaFreeHost( h_M_TP );
    cudaFreeHost( h_M_TPs );
    cudaFreeHost( h_M_TV );

    cudaFreeHost( h_M_PT );
    cudaFreeHost( h_M_PP );
    cudaFreeHost( h_M_PPs );
    cudaFreeHost( h_M_PV );
    
    cudaFreeHost( h_M_PsT );
    cudaFreeHost( h_M_PsP );
    cudaFreeHost( h_M_PsPs );
    cudaFreeHost( h_M_PsV );
    
    cudaFreeHost( h_M_VT );
    cudaFreeHost( h_M_VP );
    cudaFreeHost( h_M_VPs );
    cudaFreeHost( h_M_VV );
    
    cudaFree( gpu_M_TT );
    cudaFree( gpu_M_TP );
    cudaFree( gpu_M_TPs );
    cudaFree( gpu_M_TV );

    cudaFree( gpu_M_PT );
    cudaFree( gpu_M_PP );
    cudaFree( gpu_M_PPs );
    cudaFree( gpu_M_PV );
    
    cudaFree( gpu_M_PsT );
    cudaFree( gpu_M_PsP );
    cudaFree( gpu_M_PsPs );
    cudaFree( gpu_M_PsV );
    
    cudaFree( gpu_M_VT );
    cudaFree( gpu_M_VP );
    cudaFree( gpu_M_VPs );
    cudaFree( gpu_M_VV );
}

__global__ void
convolve_mueller_beams_with_map
(
    // Input: number of samples in the pointing stream
    int nsamples,

    // Pointing of the detector in Equatorial Coordinates coordinates
    double ra[], double dec[], double pa[], double pol_angle,

    int npix_max, int *num_pixels_in_grid, int *eval_grid_pixels,
    
    int beam_grid_size, double beam_grid_side,
    cuDoubleComplex M_TT[],  cuDoubleComplex M_TP[],  cuDoubleComplex M_TPs[],  cuDoubleComplex M_TV[],
    cuDoubleComplex M_PT[],  cuDoubleComplex M_PP[],  cuDoubleComplex M_PPs[],  cuDoubleComplex M_PV[],
    cuDoubleComplex M_PsT[], cuDoubleComplex M_PsP[], cuDoubleComplex M_PsPs[], cuDoubleComplex M_PsV[],
    cuDoubleComplex M_VT[] , cuDoubleComplex M_VP[] , cuDoubleComplex M_VPs[] , cuDoubleComplex M_VV[],

    // Input map nside parameter. Maps are held in texture memory.
    int input_map_nside,

    // Detector data to be synthethized
    double *data
)
{
    // Sample index
    int sample  = blockIdx.x;
    int pixel   = threadIdx.x;

    // Shared memory buffers to store block-wise computations
    __shared__ cuDoubleComplex T_obs [ CUDA_BLOCK_SIZE ];
    __shared__ cuDoubleComplex P_obs [ CUDA_BLOCK_SIZE ];
    __shared__ cuDoubleComplex Ps_obs[ CUDA_BLOCK_SIZE];
    

    // Complex zero
    cuDoubleComplex c_zero = make_cuDoubleComplex( 0.0, 0.0 );
    // Complex one
    cuDoubleComplex c_one  = make_cuDoubleComplex( 1.0, 0.0 );
    
    // Buffer to store Complex Stokes parameters from the sky
    cuDoubleComplex S[4];
    
    // Buffer to store Complex Rotation Matrix taking Stokes parameters from the sky to the beam frame
    cuDoubleComplex to_ins[4][4];
    
    // Buffer to store pixel Mueller Matrix
    cuDoubleComplex M  [4][4];
    
    cuDoubleComplex M_norm[4][4];

    cuDoubleComplex r1 = make_cuDoubleComplex( cos(2*pol_angle), -sin(2*pol_angle) );
    cuDoubleComplex r2 = make_cuDoubleComplex( cos(2*pol_angle),  sin(2*pol_angle) );
            
    for( int s = sample; s < nsamples; s+= gridDim.x )
    {
        //printf( "%d\n", s );
        // Get pointing of the detector
        double ra_bc  = ra[s];
        double dec_bc = dec[s];
        double pa_bc  = pa[s];
        
        int pixels_in_grid = num_pixels_in_grid[s];

        T_obs [threadIdx.x] = c_zero;
        P_obs [threadIdx.x] = c_zero;
        Ps_obs[threadIdx.x] = c_zero;
        
        for( int k=0; k<4; k++ )
        for( int l=0; l<4; l++ )
            M_norm[k][l] = c_zero;

        for( int pp=pixel; pp < pixels_in_grid; pp+=blockDim.x )
        {
            int sky_pixel = eval_grid_pixels[ s*npix_max + pp ];
            // Get sky value at beam position
            float4 IQUV;
            float I_pix = 0, Q_pix = 0, U_pix = 0, V_pix = 0;
            //int sky_pixel = cudachealpix_ang2pix( input_map_nside, ra_pix, HALF_PI - dec_pix );
            IQUV = tex1Dfetch(tex_IQUV, sky_pixel );
            I_pix += IQUV.x; Q_pix += IQUV.y; U_pix += IQUV.z; V_pix += IQUV.w;
            
            double tht_pix, ra_pix;
            cudachealpix_pix2ang( input_map_nside, sky_pixel, &tht_pix, &ra_pix );

            double dx;
            double dy;
            dx_dy_pix( ra_bc, dec_bc, pa_bc, ra_pix, HALF_PI - tht_pix, &dx, &dy );
            
            double p = pa_pix( dx, dy, dec_bc, pa_bc );
            double psi2 = 2 * pa_bc;
            
            // Assemble rotation matrix to instrument basis
            cuDoubleComplex f1  = make_cuDoubleComplex( cos(psi2), -sin(psi2) );
            cuDoubleComplex f2  = make_cuDoubleComplex( cos(psi2),  sin(psi2) );

            to_ins[0][0] = c_one;  to_ins[1][0] = c_zero; to_ins[2][0] = c_zero; to_ins[3][0] = c_zero;
            to_ins[0][1] = c_zero; to_ins[1][1] = f1;     to_ins[2][1] = c_zero; to_ins[3][1] = c_zero;
            to_ins[0][2] = c_zero; to_ins[1][2] = c_zero; to_ins[2][2] = f2    ; to_ins[3][2] = c_zero;
            to_ins[0][3] = c_zero; to_ins[1][3] = c_zero; to_ins[2][3] = c_zero; to_ins[3][3] = c_one ;
            
            /*
            // Compute analytical beam
            double g = 1.0;
            if( acos( cos(dx)*cos(dy) ) > 3.14159265/180.0 * 1.0 )
                g = 0.0;
            */           
            // Assemble Mueller Matrix
            M[0][0] = cbilinear_interpolation( M_TT  , beam_grid_side, beam_grid_size, dx, dy );  
            M[0][1] = cbilinear_interpolation( M_TP  , beam_grid_side, beam_grid_size, dx, dy );  
            M[0][2] = cbilinear_interpolation( M_TPs , beam_grid_side, beam_grid_size, dx, dy );  
            M[0][3] = cbilinear_interpolation( M_TV  , beam_grid_side, beam_grid_size, dx, dy );  
            
            M[1][0] = cbilinear_interpolation( M_PT  , beam_grid_side, beam_grid_size, dx, dy );  
            M[1][1] = cbilinear_interpolation( M_PP  , beam_grid_side, beam_grid_size, dx, dy );  
            M[1][2] = cbilinear_interpolation( M_PPs , beam_grid_side, beam_grid_size, dx, dy );  
            M[1][3] = cbilinear_interpolation( M_PV  , beam_grid_side, beam_grid_size, dx, dy );  
            
            M[2][0] = cbilinear_interpolation( M_PsT , beam_grid_side, beam_grid_size, dx, dy );  
            M[2][1] = cbilinear_interpolation( M_PsP , beam_grid_side, beam_grid_size, dx, dy );  
            M[2][2] = cbilinear_interpolation( M_PsPs, beam_grid_side, beam_grid_size, dx, dy );  
            M[2][3] = cbilinear_interpolation( M_PsV , beam_grid_side, beam_grid_size, dx, dy );  
            
            M[3][0] = cbilinear_interpolation( M_VT  , beam_grid_side, beam_grid_size, dx, dy );  
            M[3][1] = cbilinear_interpolation( M_VP  , beam_grid_side, beam_grid_size, dx, dy );  
            M[3][2] = cbilinear_interpolation( M_VPs , beam_grid_side, beam_grid_size, dx, dy );  
            M[3][3] = cbilinear_interpolation( M_VV  , beam_grid_side, beam_grid_size, dx, dy );  
            
            /*
            // Assemble M_norm
            M_norm[0][0] = cuCadd( M_norm[0][0], M[0][0] );
            M_norm[0][1] = cuCadd( M_norm[0][1], M[0][1] );
            M_norm[0][2] = cuCadd( M_norm[0][2], M[0][2] );
            M_norm[0][3] = cuCadd( M_norm[0][3], M[0][3] );
            
            M_norm[1][0] = cuCadd( M_norm[1][0], M[1][0] );
            M_norm[1][1] = cuCadd( M_norm[1][1], M[1][1] );
            M_norm[1][2] = cuCadd( M_norm[1][2], M[1][2] );
            M_norm[1][3] = cuCadd( M_norm[1][3], M[1][3] );
            
            M_norm[2][0] = cuCadd( M_norm[2][0], M[2][0] );
            M_norm[2][1] = cuCadd( M_norm[2][1], M[2][1] );
            M_norm[2][2] = cuCadd( M_norm[2][2], M[2][2] );
            M_norm[2][3] = cuCadd( M_norm[2][3], M[2][3] );
            
            M_norm[3][0] = cuCadd( M_norm[3][0], M[3][0] );
            M_norm[3][1] = cuCadd( M_norm[3][1], M[3][1] );
            M_norm[3][2] = cuCadd( M_norm[3][2], M[3][2] );
            M_norm[3][3] = cuCadd( M_norm[3][3], M[3][3] );
            */
            
            
            // Assemble Complex Stokes Vector
            S[0] = make_cuDoubleComplex( I_pix, 0.0   );
            S[1] = make_cuDoubleComplex( Q_pix,+U_pix );
            S[2] = make_cuDoubleComplex( Q_pix,-U_pix );
            S[3] = make_cuDoubleComplex( V_pix, 0.0   );
            
            
            // Rotate Stokes Vector to the beam frame
            complexla_matrix_times_vector( to_ins, S, S );            
            
            // Apply Mueller Matrix to Stokes vector in the beam frame
            complexla_matrix_times_vector( M, S, S );

             T_obs[threadIdx.x] = cuCadd( T_obs[threadIdx.x], S[0] );
             P_obs[threadIdx.x] = cuCadd( P_obs[threadIdx.x], S[1] );
            Ps_obs[threadIdx.x] = cuCadd(Ps_obs[threadIdx.x], S[2] );
        }

        __syncthreads();

        //This uses a tree structure to do the addtions
        for (int stride = blockDim.x/2; stride >  0; stride /= 2)
        {
            if ( threadIdx.x < stride)
            {
                T_obs[threadIdx.x] = cuCadd( T_obs[threadIdx.x], T_obs[threadIdx.x + stride] );
                P_obs[threadIdx.x] = cuCadd( P_obs[threadIdx.x], P_obs[threadIdx.x + stride] );
               Ps_obs[threadIdx.x] = cuCadd(Ps_obs[threadIdx.x],Ps_obs[threadIdx.x + stride] );
            }
            __syncthreads();
        } 
        
        /*
        // Multiply by normalizatoin Matrix
        S[0] = T_obs[0];
        S[1] = P_obs[0];
        S[2] = Ps_obs[0];
        S[3] = c_zero;
        complexla_matrix_times_vector( M_norm, S, S );                                                                  
        */
        if( threadIdx.x == 0 )
        {

            data[s] =     cuCreal(  T_obs[0]     ) + 
                      0.5*cuCreal( cuCadd( cuCmul(  P_obs[1], r1 ) , 
                                           cuCmul(  Ps_obs[2], r2 ) ) );
        }

        __syncthreads();

        
    }

}

extern "C" void
libconvolve_cuda_deproject_detector
(
    // Number of samples in the pointing stream
    int nsamples,

    // Pointing of the feedhorn
    double ra[], double dec[], double pa[],

    // Detector orientation angles inside the feedhorn
    double det_pol_angle,

    double beam_grid_side, int beam_grid_size,

    int npix_max, int num_pixels[], int evalgrid_pixels[], 

    // first row
    double* reM_TT , double* imM_TT, double* reM_TP , double* imM_TP, double* reM_TPs, double* imM_TPs, double* reM_TV , double* imM_TV,
    // second row
    double* reM_PT , double* imM_PT, double* reM_PP , double* imM_PP, double* reM_PPs, double* imM_PPs, double* reM_PV , double* imM_PV,
    // third row
    double* reM_PsT , double* imM_PsT, double* reM_PsP , double* imM_PsP, double* reM_PsPs, double* imM_PsPs, double* reM_PsV , double* imM_PsV,
    // fourth row
    double* reM_VT , double* imM_VT, double* reM_VP , double* imM_VP, double* reM_VPs, double* imM_VPs, double* reM_VV , double* imM_VV,

    // Maps with Stokes parameters
    int input_map_nside, int input_map_size, float I[], float Q[], float U[], float V[],

    // Device to use
    int gpu_id,

    // Output: ndets x nsamples table with detector data.
    double tod[]
) {
    
    // Set computation device
    gpu_error_check ( cudaSetDevice( gpu_id ) );
        
    build_and_transfer_eval_grid( nsamples, npix_max, num_pixels, evalgrid_pixels );
    
    allocate_tod( nsamples );
    
    transfer_pointing_streams( nsamples, ra, dec, pa );
    
    allocate_and_transfer_mueller_beams
    (
         beam_grid_size,
         reM_TT  ,  imM_TT, 
         reM_TP  ,  imM_TP, 
         reM_TPs ,  imM_TPs, 
         reM_TV  ,  imM_TV, 
         reM_PT  ,  imM_PT, 
         reM_PP  ,  imM_PP, 
         reM_PPs ,  imM_PPs, 
         reM_PV  ,  imM_PV, 
         reM_PsT ,  imM_PsT, 
         reM_PsP ,  imM_PsP, 
         reM_PsPs,  imM_PsPs, 
         reM_PsV ,  imM_PsV, 
         reM_VT  ,  imM_VT, 
         reM_VP  ,  imM_VP, 
         reM_VPs ,  imM_VPs, 
         reM_VV  ,  imM_VV
    );
    
    texturize_maps( input_map_size, I, Q, U, V );
    
    convolve_mueller_beams_with_map<<< CUDA_NUM_BLOCKS, CUDA_BLOCK_SIZE >>>
    (
        // Input: number of samples in the pointing stream
        nsamples,

        // Pointing of the detector in Equatorial Coordinates coordinates
        gpu_ra,  gpu_dec,  gpu_pa,  det_pol_angle,

        npix_max, gpu_num_pixels_in_grid, gpu_eval_grid_pixels,

        beam_grid_size,  beam_grid_side,
        gpu_M_TT,   gpu_M_TP,   gpu_M_TPs,   gpu_M_TV,
        gpu_M_PT,   gpu_M_PP,   gpu_M_PPs,   gpu_M_PV,
        gpu_M_PsT,  gpu_M_PsP,  gpu_M_PsPs,  gpu_M_PsV,
        gpu_M_VT ,  gpu_M_VP ,  gpu_M_VPs ,  gpu_M_VV,

        // Input map nside parameter. Maps are held in texture memory.
        input_map_nside,

        // Detector data to be synthethized
        gpu_tod
    );
    
    gpu_error_check( cudaDeviceSynchronize() );

    // Transfer detector data back to host
    gpu_error_check(
        cudaMemcpy(
        tod,
        gpu_tod,
        nsamples * sizeof(double),
        cudaMemcpyDeviceToHost ) );
    
    free_everything();
}

