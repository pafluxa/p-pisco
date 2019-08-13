#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#include "mapping_routines.h"

static char module_docstring[] = "\
todsynth.mapping C-extension for making sky projections of detector data into HEALPIX maps.";

static PyObject *projection_project_data_to_matrices(PyObject *self, PyObject *args);
static char      projection_project_data_to_matrices_docstring[] ="";

static PyObject *projection_project_data_to_cubes(PyObject *self, PyObject *args);
static char      projection_project_data_to_cubes_docstring[] ="";

static PyObject *projection_get_IQU_from_matrices(PyObject *self, PyObject *args);
static char      projection_get_IQU_from_matrices_docstring[] ="";


static PyMethodDef module_methods[] = {

    {"project_data_to_cubes",
        projection_project_data_to_cubes,
        METH_VARARGS,
        projection_project_data_to_cubes_docstring },
    
    {"project_data_to_matrices",
        projection_project_data_to_matrices,
        METH_VARARGS,
        projection_project_data_to_matrices_docstring },
    
    {"get_IQU_from_matrices",
        projection_get_IQU_from_matrices,
        METH_VARARGS,
        projection_get_IQU_from_matrices_docstring },
    
    // Centinel
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_mapping(void)
{
    PyObject *m = Py_InitModule3("_mapping", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

static PyObject *projection_project_data_to_cubes(PyObject *self, PyObject *args)
{  
    // Detector pointing.
    PyObject *pyObj_phi;
    PyObject *pyObj_theta;
    PyObject *pyObj_psi;
    
    // Detector polarization angles
    PyObject *pyObj_pol_angles;
        
    // Data to be projected
    PyObject *pyObj_data;

    // Data mask for bad data
    PyObject *pyObj_data_mask;
    
    // Detector mask with zeros for good detectors
    PyObject *pyObj_det_mask;

    // Projection resolution parameter
    int     map_nside;

    // AtA and AtD matrix buffers
	PyObject *pyObj_AtD, *pyObj_AtA;
    // Pixels to be comap_nsidered in the projection
    PyObject *pyObj_pixels_in_map;

    // Parser input parameters. 
    if (!PyArg_ParseTuple(args, "OOOOOOOiOOO", 
    
            &pyObj_phi, &pyObj_theta, &pyObj_psi, &pyObj_pol_angles,
            
            &pyObj_data, &pyObj_data_mask, &pyObj_det_mask,
            
            &map_nside,
            &pyObj_pixels_in_map,
            // output
			&pyObj_AtA, &pyObj_AtD ) ) {
        
        fprintf( stderr, "Incorrect use of function. RTFM!\n" );
        return NULL; 
    }
   
    // From Python Objects to Python Arrays
    PyArrayObject *pyArr_phi = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_phi  , NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_theta = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_theta, NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_psi = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_psi  , NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);

    PyArrayObject *pyArr_pol_angles = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_pol_angles, NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    
    PyArrayObject *pyArr_data             = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_data      , NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);    

    PyArrayObject *pyArr_data_mask = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_data_mask, NPY_INT   , NPY_ARRAY_IN_ARRAY);
    
    PyArrayObject *pyArr_det_mask         = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_det_mask       , NPY_INT   , NPY_ARRAY_IN_ARRAY);

    PyArrayObject *pyArr_pixels_in_map    = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_pixels_in_map   , NPY_INT   , NPY_ARRAY_IN_ARRAY);
    
    PyArrayObject *pyArr_AtD = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_AtD  , NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyArrayObject *pyArr_AtA = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_AtA  , NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    

    int nsamples = (int)PyArray_DIM( pyArr_data, 1 );
    int ndets    = (int)PyArray_DIM( pyArr_data, 0 );
    
    // mapMask is a npixel x 1 table with pixels that are masked off the projection. 
    int map_size = (int)PyArray_DIM( pyArr_pixels_in_map, 0 );

    // From Python Arrays to C arrays
    double *phi     = (double *)PyArray_DATA(pyArr_phi     );
    double *theta   = (double *)PyArray_DATA(pyArr_theta   );
    double *psi     = (double *)PyArray_DATA(pyArr_psi     );
    
    double *pol_angles = (double *)PyArray_DATA(pyArr_pol_angles );
    
    double  *data     = (double  *)PyArray_DATA(pyArr_data     );

    int    *data_mask = (int    *)PyArray_DATA(pyArr_data_mask );
    int    *det_mask         = (int    *)PyArray_DATA(pyArr_det_mask         );
    
    int    *pixels_in_map    = (int    *)PyArray_DATA(pyArr_pixels_in_map  );
    
    double *AtA      = (double *)PyArray_DATA(pyArr_AtA      );
    double *AtD      = (double *)PyArray_DATA(pyArr_AtD      );


    // Call libmapping
    libmapping_project_data_to_matrix_cube
    (
        nsamples, ndets, phi, theta, psi, pol_angles,
        data, data_mask, det_mask,
        map_nside, map_size, pixels_in_map,
        AtA, AtD
    );

    // Clean up
    Py_DECREF( pyArr_phi );
    Py_DECREF( pyArr_theta );
    Py_DECREF( pyArr_psi );
    Py_DECREF( pyArr_data );
    Py_DECREF( pyArr_data_mask );
    Py_DECREF( pyArr_pixels_in_map );
    
    Py_DECREF( pyArr_AtA );
    Py_DECREF( pyArr_AtD );

    // Return None
    Py_INCREF( Py_None );
	return Py_None;

}

static PyObject *projection_project_data_to_matrices(PyObject *self, PyObject *args)
{  
    // Detector pointing.
    PyObject *pyObj_phi;
    PyObject *pyObj_theta;
    PyObject *pyObj_psi;
    
    // Detector polarization angles
    PyObject *pyObj_pol_angles;
        
    // Data to be projected
    PyObject *pyObj_data;

    // Data mask for bad data
    PyObject *pyObj_data_mask;
    
    // Detector mask with zeros for good detectors
    PyObject *pyObj_det_mask;

    // Projection resolution parameter
    int     map_nside;

    // AtA and AtD matrix buffers
	PyObject *pyObj_AtD, *pyObj_AtA;
    // Pixels to be comap_nsidered in the projection
    PyObject *pyObj_pixels_in_map;

    // Parser input parameters. 
    if (!PyArg_ParseTuple(args, "OOOOOOOiOOO", 
    
            &pyObj_phi, &pyObj_theta, &pyObj_psi, &pyObj_pol_angles,
            
            &pyObj_data, &pyObj_data_mask, &pyObj_det_mask,
            
            &map_nside,
            &pyObj_pixels_in_map,
            // output
			&pyObj_AtA, &pyObj_AtD ) ) {
        
        fprintf( stderr, "Incorrect use of function. RTFM!\n" );
        return NULL; 
    }
   
    // From Python Objects to Python Arrays
    PyArrayObject *pyArr_phi = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_phi  , NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_theta = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_theta, NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_psi = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_psi  , NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);

    PyArrayObject *pyArr_pol_angles = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_pol_angles, NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);
    
    PyArrayObject *pyArr_data             = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_data      , NPY_DOUBLE , NPY_ARRAY_IN_ARRAY);    

    PyArrayObject *pyArr_data_mask = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_data_mask, NPY_INT   , NPY_ARRAY_IN_ARRAY);
    
    PyArrayObject *pyArr_det_mask         = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_det_mask       , NPY_INT   , NPY_ARRAY_IN_ARRAY);

    PyArrayObject *pyArr_pixels_in_map    = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_pixels_in_map   , NPY_INT   , NPY_ARRAY_IN_ARRAY);
    
    PyArrayObject *pyArr_AtD = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_AtD  , NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyArrayObject *pyArr_AtA = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_AtA  , NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    

    int nsamples = (int)PyArray_DIM( pyArr_data, 1 );
    int ndets    = (int)PyArray_DIM( pyArr_data, 0 );
    
    // mapMask is a npixel x 1 table with pixels that are masked off the projection. 
    int map_size = (int)PyArray_DIM( pyArr_pixels_in_map, 0 );

    // From Python Arrays to C arrays
    double *phi     = (double *)PyArray_DATA(pyArr_phi     );
    double *theta   = (double *)PyArray_DATA(pyArr_theta   );
    double *psi     = (double *)PyArray_DATA(pyArr_psi     );
    
    double *pol_angles = (double *)PyArray_DATA(pyArr_pol_angles );
    
    double  *data     = (double  *)PyArray_DATA(pyArr_data     );

    int    *data_mask = (int    *)PyArray_DATA(pyArr_data_mask );
    int    *det_mask         = (int    *)PyArray_DATA(pyArr_det_mask         );
    
    int    *pixels_in_map    = (int    *)PyArray_DATA(pyArr_pixels_in_map  );
    
    double *AtA      = (double *)PyArray_DATA(pyArr_AtA      );
    double *AtD      = (double *)PyArray_DATA(pyArr_AtD      );


    // Call libmapping
    libmapping_project_data_to_matrices
    (
        nsamples, ndets, phi, theta, psi, pol_angles,
        data, data_mask, det_mask,
        map_nside, map_size, pixels_in_map,
        AtA, AtD
    );

    // Clean up
    Py_DECREF( pyArr_phi );
    Py_DECREF( pyArr_theta );
    Py_DECREF( pyArr_psi );
    Py_DECREF( pyArr_data );
    Py_DECREF( pyArr_data_mask );
    Py_DECREF( pyArr_pixels_in_map );
    
    Py_DECREF( pyArr_AtA );
    Py_DECREF( pyArr_AtD );

    // Return None
    Py_INCREF( Py_None );
	return Py_None;

}


static PyObject *projection_get_IQU_from_matrices(PyObject *self, PyObject *args)
{   
    // Projection resolution parameter
    int map_nside;

    // AtA and AtD matrix buffers
	PyObject *pyObj_AtD, *pyObj_AtA;
    // Pixels to be considered in the projection
    PyObject *pyObj_pixels_in_map;

    // Output maps
	PyObject *pyObj_I, *pyObj_Q, *pyObj_U, *pyObj_W;

    // Parser input parameters. 
    if (!PyArg_ParseTuple(args, "iOOOOOOO", 
    
            &map_nside, &pyObj_AtA, &pyObj_AtD, &pyObj_pixels_in_map,
            
            // output
			&pyObj_I, &pyObj_Q, &pyObj_U, &pyObj_W ) ) {
        
        fprintf( stderr, "Incorrect use of function. RTFM!\n" );
        return NULL; 
    }
    
    PyArrayObject *pyArr_pixels_in_map    = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_pixels_in_map   , NPY_INT   , NPY_ARRAY_IN_ARRAY);
    
    PyArrayObject *pyArr_AtD = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_AtD  , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_AtA = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_AtA  , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    PyArrayObject *pyArr_I = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_I  , NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyArrayObject *pyArr_Q = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_Q  , NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyArrayObject *pyArr_U = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_U  , NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyArrayObject *pyArr_W = 
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_W  , NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    
    // mapMask is a npixel x 1 table with pixels that are masked off the projection. 
    int map_size = (int)PyArray_DIM( pyArr_pixels_in_map, 0 );

    int    *pixels_in_map    = (int    *)PyArray_DATA(pyArr_pixels_in_map  );
    
    double *I      = (double *)PyArray_DATA(pyArr_I);
    double *Q      = (double *)PyArray_DATA(pyArr_Q);
    double *U      = (double *)PyArray_DATA(pyArr_U);
    double *W      = (double *)PyArray_DATA(pyArr_W);
    
    double *AtA      = (double *)PyArray_DATA(pyArr_AtA      );
    double *AtD      = (double *)PyArray_DATA(pyArr_AtD      );
 
    // Call libmapping
    libmapping_get_IQU_from_matrices
    (
        map_nside, map_size,
        AtA, AtD, pixels_in_map,
        I,Q,U,W
    );

    // Clean up
    Py_DECREF( pyArr_AtD );
    Py_DECREF( pyArr_AtA );
    Py_DECREF( pyArr_pixels_in_map );

    // Return None
    Py_INCREF( Py_None );
	return Py_None;

}
