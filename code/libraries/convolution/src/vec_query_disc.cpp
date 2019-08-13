#include <stdio.h>
#include <string.h>

#include <math.h>
#include <omp.h>

#include <vector>

#include <healpix_map.h>
#include <rangeset.h>

#define PI (3.141592653589793238462643383279)

extern "C"
void libconvolve_vectorized_query_disc
(
    // input
    int nsamples, double ra[], double dec[],
    int pixel_buffer_nside, int pixel_buffer_size, double disc_size, 
    // output
    int pixel_buffer[], int pixel_count_buffer[]
)
{   
    #pragma omp parallel default( shared )
    {
    
    // Prepare parallelization scheme
    int thread_num;
    int num_threads;
    int samples_per_thread;
    int start, end;
    
    thread_num  = omp_get_thread_num();
    num_threads = omp_get_num_threads();
    
    samples_per_thread = nsamples / num_threads;
    
    start = thread_num * samples_per_thread;
    
    if( thread_num + 1 == num_threads )
        samples_per_thread += nsamples % num_threads ;
    
    end   = start + samples_per_thread;
    
    //printf( "thread: %d start=%d end=%d\n", thread_num, start, end );

    int k;
    int s;
    int *pixels;
    int n_pixels;
    int order;
    
    order = (int)( log(pixel_buffer_nside)/log(2) );
    Healpix_Map<int> H ( order, RING );
    rangeset< int > R;
    std::vector<int> v;
    
    for( s=start; s < end; s++ )
    {
        pointing p ( PI/2.0 - dec[s], ra[s] );
        H.query_disc( p, disc_size, R );
        
        R.toVector( v );     
        
        if( v.size() > pixel_buffer_size )
        {
            fprintf( stderr, "Pixel buffer size is smaller than query_disc() output. Aborting.\n" );
        }
        
        pixels = v.data();
        n_pixels = v.size();

        for( k=0; k < v.size(); k++ )
            pixel_buffer[ s * pixel_buffer_size + k ] = pixels[k];
        
        pixel_count_buffer[s] = n_pixels;
    }
    
    }
} 
