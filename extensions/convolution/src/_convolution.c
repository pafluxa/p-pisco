#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#include <omp.h>

#include "cuda_convolver.h"

#include "vec_query_disc.h"

static char module_docstring[] = "";

static PyObject *convolution_deproject_detector(PyObject *self, PyObject *args);
static char      convolution_deproject_detector_docstring[] ="";

static PyObject *convolution_vectorized_query_disc(PyObject *self, PyObject *args);
static char      convolution_vectorized_query_disc_docstring[] ="";

static PyObject *convolution_vectorized_query_ranges(PyObject *self, PyObject *args);
static char      convolution_vectorized_query_ranges_docstring[] ="";

static PyMethodDef module_methods[] = {

    {"deproject_detector",
        convolution_deproject_detector,
        METH_VARARGS,
        convolution_deproject_detector_docstring },
    {"vectorized_query_disc",
        convolution_vectorized_query_disc,
        METH_VARARGS,
        convolution_vectorized_query_disc_docstring },
    {"vectorized_query_ranges",
        convolution_vectorized_query_ranges,
        METH_VARARGS,
        convolution_vectorized_query_ranges_docstring },
    // Centinel
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_convolution(void)
{
    PyObject *m = Py_InitModule3("_convolution", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}


static PyObject *convolution_vectorized_query_disc( PyObject *self, PyObject *args )
{

    PyObject *pyObj_ra, *pyObj_dec, *pyObj_pixel_buffer, *pyObj_pixel_count_buffer;
    int pixel_buffer_nside;
    double disc_size;

    if (!PyArg_ParseTuple(args, "OOidOO",
        &pyObj_ra, &pyObj_dec, &pixel_buffer_nside, &disc_size, &pyObj_pixel_buffer, &pyObj_pixel_count_buffer ) )
    {
        fprintf( stderr, "Bad arguments to function. RTFM!\n" );
        return NULL;
    }

    // Parse feedhorn coordinates to numpy arrays
    PyArrayObject *pyArr_ra =
         (PyArrayObject*)PyArray_FROM_OTF( pyObj_ra, NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_dec =
         (PyArrayObject*)PyArray_FROM_OTF( pyObj_dec, NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_pixel_buffer =
         (PyArrayObject*)PyArray_FROM_OTF( pyObj_pixel_buffer, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_pixel_count_buffer =
         (PyArrayObject*)PyArray_FROM_OTF( pyObj_pixel_count_buffer, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    
    int nsamples           = (int)PyArray_DIM( pyArr_pixel_buffer, 0 );
    int pixel_buffer_size  = (int)PyArray_DIM( pyArr_pixel_buffer, 1 );
    
    int *pixel_buffer = (int *)PyArray_DATA( pyArr_pixel_buffer );
    int *pixel_count_buffer = (int *)PyArray_DATA( pyArr_pixel_count_buffer );

    double *ra  = (double *)PyArray_DATA( pyArr_ra  );
    double *dec = (double *)PyArray_DATA( pyArr_dec );
    
    libconvolve_vectorized_query_disc                                                                             
    (                                                                                                             
    // input                                                                                                  
    nsamples, ra, dec,                                                                  
    pixel_buffer_nside, pixel_buffer_size, disc_size,                                          
    // output                                                                                                 
    pixel_buffer, pixel_count_buffer                                             
    ); 
    
    Py_DECREF( pyArr_ra );
    Py_DECREF( pyArr_dec );

    Py_DECREF( pyArr_pixel_buffer );
    Py_DECREF( pyArr_pixel_count_buffer );
    
    Py_INCREF( Py_None );

    return Py_None;
}

static PyObject *convolution_vectorized_query_ranges( PyObject *self, PyObject *args )
{

    PyObject *pyObj_ra, *pyObj_dec, *pyObj_ranges, *pyObj_range_count;
    int sky_nside;
    double disc_size;

    if (!PyArg_ParseTuple(args, "OOidOO",
        &pyObj_ra, &pyObj_dec, 
        
        &sky_nside, &disc_size, 
        
        &pyObj_ranges, &pyObj_range_count ) )
    {
        fprintf( stderr, "Bad arguments to function. RTFM!\n" );
        return NULL;
    }

    // Parse feedhorn coordinates to numpy arrays
    PyArrayObject *pyArr_ra =
         (PyArrayObject*)PyArray_FROM_OTF( pyObj_ra, NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    
    PyArrayObject *pyArr_dec =
         (PyArrayObject*)PyArray_FROM_OTF( pyObj_dec, NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    
    PyArrayObject *pyArr_ranges =
         (PyArrayObject*)PyArray_FROM_OTF( pyObj_ranges, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    PyArrayObject *pyArr_range_count =
         (PyArrayObject*)PyArray_FROM_OTF( pyObj_range_count, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    
    int nsamples   = (int)PyArray_DIM( pyArr_ranges, 0 );
    int max_ranges = (int)PyArray_DIM( pyArr_ranges, 1 );
    
    int *ranges      = (int *)PyArray_DATA( pyArr_ranges      );
    int *range_count = (int *)PyArray_DATA( pyArr_range_count );

    double *ra  = (double *)PyArray_DATA( pyArr_ra  );
    double *dec = (double *)PyArray_DATA( pyArr_dec );
   
    libconvolve_vectorized_query_ranges                                                                           
    (                                                                                                             
        // input                                                                                                  
        nsamples , ra, dec,                                                                  
        sky_nside, disc_size,                                                                          
        max_ranges,                                                                                 
        // output                                                                                                 
        ranges, range_count 
    );                                                   
    
    Py_DECREF( pyArr_ra );
    Py_DECREF( pyArr_dec );

    Py_DECREF( pyArr_ranges );
    Py_DECREF( pyArr_range_count );
    
    Py_INCREF( Py_None );

    return Py_None;
}

static PyObject *convolution_deproject_detector(PyObject *self, PyObject *args)
{
    // Feedhorn coordinates
    PyObject *pyObj_feed_ra, *pyObj_feed_dec, *pyObj_feed_pa;

    // Detector polarization angles in the feedhorn
    double det_pol_angle;

    // NSIDE parameter of Mueller Matrices beams
    int input_beam_nside;
    // Input co and cross polar beam
    PyObject *pyObj_re_M_beams;
    PyObject *pyObj_im_M_beams;
    // Evaluation grid
    PyObject *pyObj_num_pixels;
    PyObject *pyObj_evalgrid_pixels;

    // Input maps
    PyObject *pyObj_I, *pyObj_Q, *pyObj_U, *pyObj_V;

    // Output buffer for detector data
    PyObject *pyObj_det_stream;

    // GPU device to use in the computation
    int gpu_device;
    
    int maxPix;

    // Parse input parameters into Python Objects
    if (!PyArg_ParseTuple(args, "OOOdiOOiOOOOOOiO",

                  // Feedhorn coordinates
                  &pyObj_feed_ra, &pyObj_feed_dec, &pyObj_feed_pa,

                  // Detector angles at feedhorn
                  &det_pol_angle,

                  // Mueller Matrices
                  &input_beam_nside, //NSIDE parameter of the beam
                  &pyObj_re_M_beams, &pyObj_im_M_beams, &maxPix,
                  // Evaluation grid pixels
                  &pyObj_num_pixels, &pyObj_evalgrid_pixels,

                  // Input maps to deproject from
                  &pyObj_I, &pyObj_Q, &pyObj_U, &pyObj_V,

                  // GPU device to use
                  &gpu_device,

                  // Ouput detector streams
                  &pyObj_det_stream ) ) {

        return NULL;
    }

    // Parse feedhorn coordinates to numpy arrays
    PyArrayObject *pyArr_feed_ra =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_feed_ra  , NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_feed_dec =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_feed_dec  , NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_feed_pa =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_feed_pa  , NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);

    // Parse input maps to numpy arrays
    PyArrayObject *pyArr_I =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_I, NPY_FLOAT , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_Q =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_Q, NPY_FLOAT , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_U =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_U, NPY_FLOAT , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_V =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_V, NPY_FLOAT , NPY_ARRAY_IN_ARRAY);

    PyArrayObject *pyArr_re_M_beams =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_re_M_beams, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_im_M_beams =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_im_M_beams, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    PyArrayObject *pyArr_num_pixels =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_num_pixels, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_evalgrid_pixels =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_evalgrid_pixels, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    // Parse detector streams to numpy arrays (as output!!)
    PyArrayObject *pyArr_det_stream =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_det_stream, NPY_FLOAT64, NPY_ARRAY_OUT_ARRAY);

     // The input map size is the dimension of the I map
    int input_map_size  = (int)PyArray_DIM( pyArr_I, 0 );

    // The input map nside is sqrt( len(I)/12 )
    int input_map_nside = (int)( sqrt( input_map_size/12 ) );

    // Number of samples is the length of the boresight stream
    int nsamples = PyArray_DIM( pyArr_feed_ra, 0 );

    // Parse complex beam Mueller Matrices to C-arrays
    double* reM_TT   = PyArray_GETPTR2( pyArr_re_M_beams, 0, 0 );
    double* reM_TP   = PyArray_GETPTR2( pyArr_re_M_beams, 0, 1 );
    double* reM_TPs  = PyArray_GETPTR2( pyArr_re_M_beams, 0, 2 );
    double* reM_TV   = PyArray_GETPTR2( pyArr_re_M_beams, 0, 3 );

    double* reM_PT   = PyArray_GETPTR2( pyArr_re_M_beams, 1, 0 );
    double* reM_PP   = PyArray_GETPTR2( pyArr_re_M_beams, 1, 1 );
    double* reM_PPs  = PyArray_GETPTR2( pyArr_re_M_beams, 1, 2 );
    double* reM_PV   = PyArray_GETPTR2( pyArr_re_M_beams, 1, 3 );

    double* reM_PsT  = PyArray_GETPTR2( pyArr_re_M_beams, 2, 0 );
    double* reM_PsP  = PyArray_GETPTR2( pyArr_re_M_beams, 2, 1 );
    double* reM_PsPs = PyArray_GETPTR2( pyArr_re_M_beams, 2, 2 );
    double* reM_PsV  = PyArray_GETPTR2( pyArr_re_M_beams, 2, 3 );

    double* reM_VT  = PyArray_GETPTR2( pyArr_re_M_beams , 3, 0 );
    double* reM_VP  = PyArray_GETPTR2( pyArr_re_M_beams , 3, 1 );
    double* reM_VPs = PyArray_GETPTR2( pyArr_re_M_beams , 3, 2 );
    double* reM_VV  = PyArray_GETPTR2( pyArr_re_M_beams , 3, 3 );

    double* imM_TT   = PyArray_GETPTR2( pyArr_im_M_beams, 0, 0 );
    double* imM_TP   = PyArray_GETPTR2( pyArr_im_M_beams, 0, 1 );
    double* imM_TPs  = PyArray_GETPTR2( pyArr_im_M_beams, 0, 2 );
    double* imM_TV   = PyArray_GETPTR2( pyArr_im_M_beams, 0, 3 );

    double* imM_PT   = PyArray_GETPTR2( pyArr_im_M_beams, 1, 0 );
    double* imM_PP   = PyArray_GETPTR2( pyArr_im_M_beams, 1, 1 );
    double* imM_PPs  = PyArray_GETPTR2( pyArr_im_M_beams, 1, 2 );
    double* imM_PV   = PyArray_GETPTR2( pyArr_im_M_beams, 1, 3 );

    double* imM_PsT  = PyArray_GETPTR2( pyArr_im_M_beams, 2, 0 );
    double* imM_PsP  = PyArray_GETPTR2( pyArr_im_M_beams, 2, 1 );
    double* imM_PsPs = PyArray_GETPTR2( pyArr_im_M_beams, 2, 2 );
    double* imM_PsV  = PyArray_GETPTR2( pyArr_im_M_beams, 2, 3 );

    double* imM_VT   = PyArray_GETPTR2( pyArr_im_M_beams, 3, 0 );
    double* imM_VP   = PyArray_GETPTR2( pyArr_im_M_beams, 3, 1 );
    double* imM_VPs  = PyArray_GETPTR2( pyArr_im_M_beams, 3, 2 );
    double* imM_VV   = PyArray_GETPTR2( pyArr_im_M_beams, 3, 3 );

    int*    num_pixels      = (int *)PyArray_DATA( pyArr_num_pixels );
    int*    evalgrid_pixels = (int *)PyArray_DATA( pyArr_evalgrid_pixels );
    // Number of samples is the length of the boresight stream
    int npix_max = PyArray_DIM( pyArr_evalgrid_pixels, 1 );

    // Parse input maps to C-arrays

    float *I = (float *)PyArray_DATA( pyArr_I );
    float *Q = (float *)PyArray_DATA( pyArr_Q );
    float *U = (float *)PyArray_DATA( pyArr_U );
    float *V = (float *)PyArray_DATA( pyArr_V );

    // Parse feedhorn pointing to C-arrays
    double *feed_ra  = (double *)PyArray_DATA(pyArr_feed_ra  );
    double *feed_dec = (double *)PyArray_DATA(pyArr_feed_dec );
    double *feed_pa  = (double *)PyArray_DATA(pyArr_feed_pa  );

    // Parse detector streams output buffers to C-arrays
    double *det_stream = (double *)PyArray_DATA( pyArr_det_stream );
    
    printf( "%d\n", maxPix );

    // Call for trouble!!
	libconvolve_cuda_deproject_detector
	(
        // Pointing stream of the feedhorn + its polarization angle
        nsamples, feed_ra, feed_dec, feed_pa, det_pol_angle,
        // Evaluation grid specifications
        npix_max, num_pixels, evalgrid_pixels,
        // Beam specs
        // first row
        input_beam_nside,
        reM_TT  , imM_TT,
        reM_TP  , imM_TP,
        reM_TPs , imM_TPs,
        reM_TV  , imM_TV,
        // second row
        reM_PT  , imM_PT,
        reM_PP  , imM_PP,
        reM_PPs , imM_PPs,
        reM_PV  , imM_PV,
        // third row
        reM_PsT , imM_PsT,
        reM_PsP , imM_PsP,
        reM_PsPs, imM_PsPs,
        reM_PsV , imM_PsV,
        // fourth row
        reM_VT  , imM_VT,
        reM_VP  , imM_VP,
        reM_VPs , imM_VPs,
        reM_VV  , imM_VV,
        // max pixel
        maxPix,
        // Input map specs
        input_map_nside, input_map_size, I, Q, U, V,
        
        // Select GPU device to run this shit on
        gpu_device,
            
        // Buffer to save the synth. detector data
        det_stream  
	);

    // clean up!

    Py_DECREF( pyArr_num_pixels );
    Py_DECREF( pyArr_evalgrid_pixels );

	Py_DECREF( pyArr_feed_ra );
    Py_DECREF( pyArr_feed_dec );
    Py_DECREF( pyArr_feed_pa );

	Py_DECREF( pyArr_re_M_beams );
	Py_DECREF( pyArr_im_M_beams );

    Py_DECREF( pyArr_I );
    Py_DECREF( pyArr_Q );
    Py_DECREF( pyArr_U );
    Py_DECREF( pyArr_V );
    
    Py_INCREF( Py_None );

    return Py_None;

}

