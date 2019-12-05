#include <healpix_utils.cuh>
#include <sky_coords.cuh>
#include <complex_la.h>

//#include <rotations.cuh>

#include <cuda_utils.h>

#include <chealpix.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <omp.h>
#include <cuda.h>

#define CUDA_BLOCK_SIZE     32
#define CUDA_NUM_BLOCKS    128

// Instantiate texture space for sky maps
texture<float4, 1, cudaReadModeElementType> tex_IQUV;


//######################################################################################################
// Allocate space in GPU
//######################################################################################################
int*    gpu_num_pixels_in_grid;
int*    gpu_eval_grid_pixels;

double *gpu_ra;
double *gpu_dec;
double *gpu_pa;

double *gpu_tod;

float4* gpu_IQUV;
float4* host_IQUV;

//######################################################################################################
// Mueller Matrices in real form
//######################################################################################################
double          *gpu_M_TT,  *gpu_M_TP,  *gpu_M_TPs,  *gpu_M_TV,
                *gpu_M_PT,  *gpu_M_PP,  *gpu_M_PPs,  *gpu_M_PV,
                *gpu_M_PsT, *gpu_M_PsP, *gpu_M_PsPs, *gpu_M_PsV,
                *gpu_M_VT , *gpu_M_VP , *gpu_M_VPs , *gpu_M_VV;

void texturize_maps
(
    int input_map_size,
    float I[], float Q[], float U[], float V[]
)
{
    gpu_error_check( cudaMallocHost( (void**)&host_IQUV, sizeof( float4 ) * input_map_size ) );

    gpu_error_check( cudaMalloc( (void**)&gpu_IQUV, sizeof( float4 ) * input_map_size ) );

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
    gpu_error_check( cudaMalloc( (void**)&gpu_num_pixels_in_grid, sizeof( int ) * nsamples ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_eval_grid_pixels, sizeof( int ) * nsamples * npix_max) );

    // Transfer eval evaluation grid
    gpu_error_check(
        cudaMemcpy( gpu_num_pixels_in_grid, num_pixels, sizeof( int ) * nsamples, cudaMemcpyHostToDevice) );
    gpu_error_check(
        cudaMemcpy( gpu_eval_grid_pixels, evalgrid_pixels, sizeof( int ) * nsamples * npix_max, cudaMemcpyHostToDevice) );
}

void allocate_and_transfer_mueller_beams
(
    int input_beam_nside,
    
    int maxPix,

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
    int input_beam_size = maxPix;
    
    printf( "%d\n", input_beam_size );

    //######################################################################################################
    // Allocate space on the GPU
    //######################################################################################################
    // First Row
    gpu_error_check( cudaMalloc( (void**)&gpu_M_TT,   sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_TP,   sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_TPs,  sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_TV,   sizeof( double ) * input_beam_size ) );
    // Second Row
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PT,   sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PP,   sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PPs,  sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PV,   sizeof( double ) * input_beam_size ) );
    // Third Row
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PsT,  sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PsP,  sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PsPs, sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_PsV,  sizeof( double ) * input_beam_size ) );
    // Fourth Row
    gpu_error_check( cudaMalloc( (void**)&gpu_M_VT,   sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_VP,   sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_VPs,  sizeof( double ) * input_beam_size ) );
    gpu_error_check( cudaMalloc( (void**)&gpu_M_VV,   sizeof( double ) * input_beam_size ) );
    
    //######################################################################################################
    // Move stuff to the GPU
    //######################################################################################################
    gpu_error_check( cudaMemcpy( gpu_M_TT,   reM_TT,   sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_TP,   reM_TP,   sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_TPs,  reM_TPs,  sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_TV,   reM_TV,   sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    // Second Row
    gpu_error_check( cudaMemcpy( gpu_M_PT,   reM_PT,   sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PP,   reM_PP,   sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PPs,  reM_PPs,  sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PV,   reM_PV,   sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    // Third Row
    gpu_error_check( cudaMemcpy( gpu_M_PsT,  reM_PsT,  sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PsP,  reM_PsP,  sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PsPs, reM_PsPs, sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_PsV,  reM_PsV,  sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    // Fourth Row
    gpu_error_check( cudaMemcpy( gpu_M_VT,   reM_VT ,  sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_VP,   reM_VP ,  sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_VPs,  reM_VPs,  sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
    gpu_error_check( cudaMemcpy( gpu_M_VV,   reM_VV ,  sizeof( double ) * input_beam_size  , cudaMemcpyHostToDevice));
}

void free_everything()
{
    gpu_error_check( cudaFree( gpu_tod  ) );

    // Free pointing buffers
    gpu_error_check( cudaFree( gpu_ra   ) );
    gpu_error_check( cudaFree( gpu_dec  ) );
    gpu_error_check( cudaFree( gpu_pa   ) );

    cudaFree( gpu_num_pixels_in_grid );
    cudaFree( gpu_eval_grid_pixels );

    cudaFree(gpu_IQUV);
    cudaUnbindTexture(tex_IQUV);
    cudaFreeHost( host_IQUV );
    
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

    int input_beam_nside, 
    double M_TT[],  double M_TP[],  double M_TPs[],  double M_TV[],
    double M_PT[],  double M_PP[],  double M_PPs[],  double M_PV[],
    double M_PsT[], double M_PsP[], double M_PsPs[], double M_PsV[],
    double M_VT[] , double M_VP[] , double M_VPs[] , double M_VV[],
    
    // Input map nside parameter. Maps are held in texture memory.
    int input_map_nside,

    // Detector data to be synthethized
    double *data
)
{
    // Sample index
    int sample  = blockIdx.x ;
    int pixel   = threadIdx.x;

    // Shared memory buffers to store block-wise computations
    __shared__ double I_obs[ CUDA_BLOCK_SIZE ];
    __shared__ double Q_obs[ CUDA_BLOCK_SIZE ];
    __shared__ double U_obs[ CUDA_BLOCK_SIZE ];

    // Buffer to store pixel Mueller Matrix
    double M  [4][4];
    
    for( int s = sample; s < nsamples; s+=gridDim.x )
    {
        // Get pointing of the detector
        double ra_bc  = ra[s];
        double dec_bc = dec[s];
        // Passed arguments are counterclockwise on the sky
        // CMB requires clockwise
        double psi_bc  = -pa[s];
        double cmb_pol_angle = -pol_angle;
        
        int pixels_in_grid = num_pixels_in_grid[s];

        I_obs[threadIdx.x] = 0.0;
        Q_obs[threadIdx.x] = 0.0;
        U_obs[threadIdx.x] = 0.0;
        
        for( int pp=pixel; pp < pixels_in_grid; pp+=blockDim.x )
        {
            // Get pixel of the evaluation grid (in sky coordinates)
            int eval_pixel = eval_grid_pixels[ s*npix_max + pp ];
            
            // Get value at sky_pixel
            float4 IQUV;
            double I_pix, Q_pix, U_pix;
            IQUV  = tex1Dfetch(tex_IQUV, eval_pixel );
            I_pix = IQUV.x; 
            Q_pix = IQUV.y; 
            U_pix = IQUV.z; 
            //V_pix = IQUV.w;

            // Get sky coordinates of eval_pixel
            double dec_eval, ra_eval;
            cuHealpix_pix2ang( input_map_nside, eval_pixel, &dec_eval, &ra_eval );

            // pix2agn returns co-latitude.
            dec_eval = pi/2.0 - dec_eval;
                        
            // compute dx/dy offsets from beam center to eval_pixel. Also computes pa at eval_pixel.
            double tht_at_pix, phi_at_pix, psi_at_pix;
            theta_phi_psi_pix( 
                             &tht_at_pix, &phi_at_pix, &psi_at_pix, 
                              ra_bc  , dec_bc, psi_bc,
                              ra_eval, dec_eval );
            
            /*
             * Uncomment below to make use of Ludwig's 3rd definition.
             */
            rho_sigma_chi_pix( &tht_at_pix, &phi_at_pix, &psi_at_pix,
                                ra_bc, dec_bc, psi_bc,
                                ra_eval, dec_eval );

            /*
             * Uncomment below to override polarization angle correction.
             */
            // psi_at_pix = psi_bc;
            
            /*
             * Uncomment for debugging.
            if( s == 0 && pp == 0 )
            {
                // Comment the above and uncomment this to use 3rd definition 
                double rho_at_pix, sigma_at_pix, chi_at_pix;
                rho_sigma_chi_pix( &rho_at_pix, &sigma_at_pix, &chi_at_pix,
                                   ra_bc, dec_bc, psi_bc,
                                   ra_eval, dec_eval );

                double r2d = 180.0/pi;
                printf( "ra_bc: %+2.4f dec_bc: %+2.4f psi_bc: %+2.4f\n",
                         r2d * ra_bc, r2d * dec_bc, r2d * psi_bc );
                printf( "delta rho: %+2.6f delta sigma: %+2.6f delta chi: %+2.6f\n", 
                    r2d*(tht_at_pix - rho_at_pix), 
                    r2d*(phi_at_pix - sigma_at_pix), 
                    r2d*(psi_at_pix - chi_at_pix ) );
            }
            */

            int     neigh_pixels[4];
            double           wgt[4]; 
            cuHealpix_interpolate( input_beam_nside, tht_at_pix, phi_at_pix, neigh_pixels, wgt ); 
            double scaled_M_XX = 0;
            
            M[0][0] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_TT[ neigh_pixels[i] ] * wgt[i];
                M[0][0] += scaled_M_XX;
            }
            /*
            M[0][1] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_TP[ neigh_pixels[i] ] * wgt[i];
                M[0][1] += scaled_M_XX;
            }
            
            M[0][2] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_TPs[ neigh_pixels[i] ] * wgt[i];
                M[0][2] += scaled_M_XX;
            }
            
            M[0][3] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_TV[ neigh_pixels[i] ] * wgt[i];
                M[0][3] += scaled_M_XX;
            }
            
            M[1][0] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_PT[ neigh_pixels[i] ] * wgt[i];
                M[1][0] += scaled_M_XX;
            }
            */
            M[1][1] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_PP[ neigh_pixels[i] ] * wgt[i];
                M[1][1] += scaled_M_XX;
            }
            /*
            M[1][2] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_PPs[ neigh_pixels[i] ] * wgt[i];
                M[1][2] += scaled_M_XX;
            }
            
            M[1][3] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_PV[ neigh_pixels[i] ] * wgt[i];
                M[1][3] += scaled_M_XX;
            }
            
            M[2][0] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_PsT[ neigh_pixels[i] ] * wgt[i];
                M[2][0] += scaled_M_XX;
            }
            
            M[2][1] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_PsP[ neigh_pixels[i] ] * wgt[i];
                M[2][1] += scaled_M_XX;
            }
            */
            M[2][2] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_PsPs[ neigh_pixels[i] ] * wgt[i];
                M[2][2] += scaled_M_XX;
            }
            /* 
            M[2][3] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_PsV[ neigh_pixels[i] ] * wgt[i];
                M[2][3] += scaled_M_XX;
            }
            
            M[3][0] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_VT[ neigh_pixels[i] ].x * wgt[i];
                M[3][0] += scaled_M_XX;
            }
            
            M[3][1] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_VP[ neigh_pixels[i] ].x * wgt[i];
                M[3][1] += scaled_M_XX;
            }
            
            M[3][2] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_VPs[ neigh_pixels[i] ].x * wgt[i];
                M[3][2] += scaled_M_XX;
            }
            
            M[3][3] = 0.0;
            for( int i=0; i < 4; i++ )
            {   
                scaled_M_XX = M_VV[ neigh_pixels[i] ].x * wgt[i];
                M[3][3] += scaled_M_XX;
            }
            */
            
            double  q =  2*psi_at_pix;
            double cq =        cos(q);
            double sq =        sin(q);

            double temp1 = Q_pix;
            double temp2 = U_pix;

            Q_pix =  temp1*cq + temp2*sq;
            U_pix = -temp1*sq + temp2*cq;
            
            // Apply Mueller Matrix to Stokes vector in the beam frame
            I_pix *= M[0][0];// + _Q*M[0][1] + _U*M[0][2] + _V*M[0][3];
            Q_pix *= M[1][1];// + _Q*M[1][1] + _U*M[1][2] + _V*M[1][3];
            U_pix *= M[2][2];// + _Q*M[2][1] + _U*M[2][2] + _V*M[2][3];
            
            I_obs[threadIdx.x] += I_pix;
            Q_obs[threadIdx.x] += Q_pix;
            U_obs[threadIdx.x] += U_pix;
        }

        __syncthreads();

        //This uses a tree structure to do the addtions
        for (int stride = blockDim.x/2; stride >  0; stride /= 2)
        {
            if ( threadIdx.x < stride)
            {
               I_obs[threadIdx.x] += I_obs[threadIdx.x + stride];
               Q_obs[threadIdx.x] += Q_obs[threadIdx.x + stride];
               U_obs[threadIdx.x] += U_obs[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if( threadIdx.x == 0 )
        {
            double I = I_obs[0];
            double Q = Q_obs[0];
            double U = U_obs[0];
            
            data[s] = I + Q*cos(2*cmb_pol_angle) + U*sin(2*cmb_pol_angle);
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

    // Evaluation grid specs
    int npix_max, int num_pixels[], int evalgrid_pixels[],

    // Beam specs
    int input_beam_nside,
    // first row
    double* reM_TT , double* imM_TT,
    double* reM_TP , double* imM_TP,
    double* reM_TPs, double* imM_TPs,
    double* reM_TV , double* imM_TV,
    // second row
    double* reM_PT , double* imM_PT,
    double* reM_PP , double* imM_PP,
    double* reM_PPs, double* imM_PPs,
    double* reM_PV , double* imM_PV,
    // third row
    double* reM_PsT, double* imM_PsT,
    double* reM_PsP, double* imM_PsP,
    double* reM_PsPs,double* imM_PsPs,
    double* reM_PsV ,double* imM_PsV,
    // fourth row
    double* reM_VT , double* imM_VT,
    double* reM_VP , double* imM_VP,
    double* reM_VPs, double* imM_VPs,
    double* reM_VV , double* imM_VV,
    
    // max pixel
    int maxPix,

    // Maps with Stokes parameters
    int input_map_nside, int input_map_size, float I[], float Q[], float U[], float V[],

    // Device to use
    int gpu_id,

    // Output: ndets x nsamples table with detector data.
    double tod[]
)
{

    // Time stuff
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // Set computation device
    gpu_error_check ( cudaSetDevice( gpu_id ) );

    allocate_tod( nsamples );

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); 
    //
    transfer_pointing_streams( nsamples, ra, dec, pa );
    //
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf( "transfer_pointing_streams: %f ms\n", milliseconds );
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); 
    //
    build_and_transfer_eval_grid( nsamples, npix_max, num_pixels, evalgrid_pixels );
    //
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf( "build_and_transfer_eval_grid: %f ms\n", milliseconds );

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); 
    //
    allocate_and_transfer_mueller_beams
    (
         input_beam_nside, maxPix,
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
    //
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf( "allocate_and_transfer_mueller_beams: %f ms\n", milliseconds );
    texturize_maps( input_map_size, I, Q, U, V );
    
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); 
    //
    convolve_mueller_beams_with_map<<< CUDA_NUM_BLOCKS, CUDA_BLOCK_SIZE >>>
    (
        // Input: number of samples in the pointing stream
        nsamples,

        // Pointing of the detector in Equatorial Coordinates coordinates
        gpu_ra,  gpu_dec,  gpu_pa,  det_pol_angle,

        npix_max, gpu_num_pixels_in_grid, gpu_eval_grid_pixels,

        input_beam_nside,
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
    //
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf( "convolve_mueller_beams_with_map: %f ms\n", milliseconds );
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    
    // Transfer detector data back to host
    gpu_error_check(
        cudaMemcpy(
        tod,
        gpu_tod,
        nsamples * sizeof(double),
        cudaMemcpyDeviceToHost ) );

    free_everything();
}

