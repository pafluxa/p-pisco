#ifndef VQUERYDISCH_
#define VQUERYDISCH_


extern void
libconvolve_vectorized_query_disc
(
    // input
    int nsamples, double ra[], double dec[],
    int pixel_buffer_nside, int pixel_buffer_size, double disc_size,
    // output
    int pixel_buffer[], int pixel_count_buffer[]
);

extern void 
libconvolve_vectorized_query_ranges
(                                                                                                             
    // input                                                                                                  
    int nsamples, double ra[], double dec[],                                                                  
    int sky_nside, double disc_size,                                                                          
    int max_ranges_in_buffer,                                                                                 
    // output                                                                                                 
    int ranges_per_pointing[], int n_ranges_per_pointing[]                                                    
);

#endif
