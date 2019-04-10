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

#endif
