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

#define CUDA_BLOCK_SIZE  256
#define CUDA_NUM_BLOCKS  256

// Instantiate texture space for sky maps
texture<float4, 1, cudaReadModeElementType> tex_IQUV;


//######################################################################################################
// Allocate space in GPU
//######################################################################################################
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
    int input_beam_size = nside2npix( input_beam_nside );

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
    
    for( int pix=0; pix < input_beam_size; pix++ )
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
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //######################################################################################################
    // Move stuff to the GPU
    //######################################################################################################
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
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    fprintf( stderr, "Took %f ms to move beams to GPU\n", milliseconds );

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

    cudaFree( gpu_num_pixels_in_grid );
    cudaFree( gpu_eval_grid_pixels );

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

    int input_beam_nside,
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
    int eval_pixel = blockIdx.x * blockDim.x + threadIdx.x ;
    if( eval_pixel > nsamples )
        return;

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

    cuDoubleComplex r1 = make_cuDoubleComplex( cos(2*pol_angle), -sin(2*pol_angle) );
    cuDoubleComplex r2 = make_cuDoubleComplex( cos(2*pol_angle),  sin(2*pol_angle) );
    
    // Small buffer to store values of the maps
    float4 IQUV;

    // Shared memor buffers
    double shared_data            [ CUDA_BLOCK_SIZE ];
    double shared_phi_eval_pixel  [ CUDA_BLOCK_SIZE ];
    double shared_tht_eval_pixel  [ CUDA_BLOCK_SIZE ];
    int    shared_eval_pixels     [ CUDA_BLOCK_SIZE ];

    int pixels_in_grid = num_pixels_in_grid[ eval_pixel ];
    for( int tile=0; tile < gridDim.x ; tile++ )
    {
        // Get coordinates of the beam center
        double  ra_bc  =  ra[ tile * blockDim.x + threadIdx.x ];
        double dec_bc  = dec[ tile * blockDim.x + threadIdx.x ];
        double  pa_bc  =  pa[ tile * blockDim.x + threadIdx.x ];

        int eval_grid_pixel = eval_grid_pixels[ tile * blockDim.x + threadIdx.x ];
        double tht_pixel, ra_pixel;
        cudachealpix_pix2ang( 
            input_map_nside, 
            eval_grid_pixel, 
            &tht_pixel, &ra_pixel );

        shared_data            [ threadIdx.x ] =  0.0;
        shared_phi_eval_pixel  [ threadIdx.x ] =  ra_pixel;
        shared_tht_eval_pixel  [ threadIdx.x ] = tht_pixel;
        shared_eval_pixels     [ threadIdx.x ] = eval_grid_pixel;
        
        __syncthreads();

        for( unsigned int counter=0; counter < blockDim.x; counter++ )
        {
            // Get the sky values of where the beam center is pointing at
            int eval_pixel = eval_grid_pixels[ counter ];
            IQUV = tex1Dfetch(tex_IQUV, eval_pixel );
            double I_sky = IQUV.x; 
            double Q_sky = IQUV.y; 
            double U_sky = IQUV.z; 
            double V_sky = IQUV.w;

            // Transform coordinates of pixel to beam coordinates
            double dx, dy;
            dx_dy_pix( ra_bc, dec_bc, pa_bc, 
                       shared_phi_eval_pixel[ counter ],
                       shared_tht_eval_pixel[ counter ],
                       &dx, &dy );

            // Compute tht and phi from beam coordinates dx and dy
            double cdx = cos(dx);
            double sdx = sin(dx);
            double cdy = cos(dy);
            double sdy = sin(dy);
            double cr  = cdx * cdy;
            double tht_pix = acos(cr);
            double phi_pix = atan2(sdy, sdx * cdy );
            // Get parallactic angle at the evaluation grid position
            double pa_pixel = pa_pix( dx, dy, dec_bc, pa_bc );
            
            // Compute which pixel in the beam falls in the evaluation grid pixel
            int beam_pix = cudachealpix_ang2pix( input_beam_nside, phi_pix, tht_pix );

            // Assemble rotation matrix to beam basis
            double psi2         = 2 * pa_pixel;
            cuDoubleComplex f1  = make_cuDoubleComplex( cos(psi2), -sin(psi2) );
            cuDoubleComplex f2  = make_cuDoubleComplex( cos(psi2),  sin(psi2) );
            to_ins[0][0] = c_one;  to_ins[1][0] = c_zero; to_ins[2][0] = c_zero; to_ins[3][0] = c_zero;
            to_ins[0][1] = c_zero; to_ins[1][1] = f1;     to_ins[2][1] = c_zero; to_ins[3][1] = c_zero;
            to_ins[0][2] = c_zero; to_ins[1][2] = c_zero; to_ins[2][2] = f2    ; to_ins[3][2] = c_zero;
            to_ins[0][3] = c_zero; to_ins[1][3] = c_zero; to_ins[2][3] = c_zero; to_ins[3][3] = c_one ; 

            // Assemble pixel of Mueller Matrix Field
            M[0][0] = M_TT  [ beam_pix ];
            M[0][1] = M_TP  [ beam_pix ];
            M[0][2] = M_TPs [ beam_pix ];
            M[0][3] = M_TV  [ beam_pix ];
            
            M[1][0] = M_PT  [ beam_pix ];
            M[1][1] = M_PP  [ beam_pix ];
            M[1][2] = M_PPs [ beam_pix ];
            M[1][3] = M_PV  [ beam_pix ];
            
            M[2][0] = M_PsT [ beam_pix ];
            M[2][1] = M_PsP [ beam_pix ];
            M[2][2] = M_PsPs[ beam_pix ];
            M[2][3] = M_PsV [ beam_pix ];
            
            M[3][0] = M_VT  [ beam_pix ];
            M[3][1] = M_VP  [ beam_pix ];
            M[3][2] = M_VPs [ beam_pix ];
            M[3][3] = M_VV  [ beam_pix ];
             
            // Assemble Complex Stokes Vector
            S[0] = make_cuDoubleComplex( I_sky, 0.0   );
            S[1] = make_cuDoubleComplex( Q_sky, U_sky );
            S[2] = make_cuDoubleComplex( Q_sky,-U_sky );
            S[3] = make_cuDoubleComplex( V_sky, 0.0   );

            // Rotate Stokes Vector to the beam frame
            complexla_matrix_times_vector( to_ins, S, S );

            // Apply Mueller Matrix to Stokes vector in the beam frame
            complexla_matrix_times_vector( M, S, S );

            data[ counter ] = ( cuCreal( S[0] ) + 0.5*cuCreal( cuCadd( cuCmul(S[1],r1 ),cuCmul(S[2],r2) ) ) );
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

    // Maps with Stokes parameters
    int input_map_nside, int input_map_size, float I[], float Q[], float U[], float V[],

    // Device to use
    int gpu_id,

    // Output: ndets x nsamples table with detector data.
    double tod[]
)
{

    // Set computation device
    gpu_error_check ( cudaSetDevice( gpu_id ) );

    allocate_tod( nsamples );

    transfer_pointing_streams( nsamples, ra, dec, pa );
    
    build_and_transfer_eval_grid( nsamples, npix_max, num_pixels, evalgrid_pixels );

    allocate_and_transfer_mueller_beams
    (
         input_beam_nside,
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

    // Transfer detector data back to host
    gpu_error_check(
        cudaMemcpy(
        tod,
        gpu_tod,
        nsamples * sizeof(double),
        cudaMemcpyDeviceToHost ) );

    free_everything();
}

