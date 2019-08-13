#include <Python.h>

#define NPY_NO_DEPRECATED_API   NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "pointing_correction.h"

static char module_docstring[] = "";
/*
"\
This is the TODSynth pointing correction C-extension. It implements communication between the standard\
antenna pointing model corrections (Jacob Baars, The Paraboloidal Reflector Antenna) implemented in\
libpointing and the Python unicorns-and-rainbows world.";
*/
static PyObject* pointing_correction_correct_pointing( PyObject *self, PyObject *args );
static char  pointing_correction_correct_pointing_ds[] = "";
/*
"Let me get this thing working. I'll write docs later!";
*/
static PyMethodDef module_methods[] = {

    {"correct_pointing",
     pointing_correction_correct_pointing, 
     METH_VARARGS, 
     pointing_correction_correct_pointing_ds } ,

     {NULL, NULL, 0, NULL} 

};

PyMODINIT_FUNC init_pointing_correction(void) 
{
    PyObject *m = Py_InitModule3("_pointing_correction", module_methods, module_docstring);                              
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

static PyObject* pointing_correction_correct_pointing(  PyObject *self, PyObject *args )
{

    PyObject *pyObj_az, *pyObj_alt, *pyObj_bs, *pyObj_tilt_para, *pyObj_tilt_perp;
    PyObject *pm_dict;
    
	if (!PyArg_ParseTuple(args, "OOOOOO",
                          &pyObj_az ,
                          &pyObj_alt,
                          &pyObj_bs,
                          &pyObj_tilt_para,
                          &pyObj_tilt_perp,
                          &pm_dict)) 
	{
        	fprintf(stderr, "(%s, %d) Incorrect usage of correct_pointing. RTFM.\n", __FILE__, __LINE__);
			return NULL;
    }

    /* Extract pointing model from pm_dict and allocate the equivalent C structure*/
    //******************************************************************************************************/
    
    // Extract azimuth correction coefficient list from dictionary
    PyObject *pyObj_az_coef      = PyDict_GetItemString(pm_dict, "az_coef");
    // Transform to numpy array object
    PyArrayObject *pyArr_az_coef =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_az_coef, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    // Transform to C-array like structure
    double *az_coef = PyArray_DATA( pyArr_az_coef );
    // Get number of coefficients from first dimension
	int n_az_coef  = (int)PyArray_DIM( pyArr_az_coef, 0 );
    	
    // Extract altitude correction coefficient list from dictionary
    PyObject *pyObj_alt_coef      = PyDict_GetItemString(pm_dict, "alt_coef");
    // Transform to numpy array object
    PyArrayObject *pyArr_alt_coef =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_alt_coef, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    // Transform to C-array like structure
    double *alt_coef = PyArray_DATA( pyArr_alt_coef );
    // Get number of coefficients from first dimension
	int n_alt_coef  = (int)PyArray_DIM( pyArr_alt_coef, 0 );
    
    // Extract altitude correction coefficient list from dictionary
    PyObject *pyObj_bs_coef      = PyDict_GetItemString(pm_dict, "bs_coef");
    // Transform to numpy array object
    PyArrayObject *pyArr_bs_coef =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_bs_coef, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    // Transform to C-array like structure
    double *bs_coef = PyArray_DATA( pyArr_bs_coef );
    // Get number of coefficients from first dimension
	int n_bs_coef  = (int)PyArray_DIM( pyArr_bs_coef, 0 );

	double az_pc  = PyFloat_AsDouble(PyDict_GetItemString(pm_dict,  "az_pc"));
	double alt_pc = PyFloat_AsDouble(PyDict_GetItemString(pm_dict, "alt_pc"));

	double x_center = PyFloat_AsDouble(PyDict_GetItemString(pm_dict, "x_center"));
	double y_center = PyFloat_AsDouble(PyDict_GetItemString(pm_dict, "y_center"));

    pmodel *pm = libpointing_alloc_pmodel( 
                    az_pc, alt_pc, x_center, y_center, 
                    n_az_coef ,  az_coef,
                    n_alt_coef, alt_coef,
                    n_bs_coef ,  bs_coef );
    /*******************************************************************************************************/

    /* Transform Python Objects representing the pointing stream into C-arrays */
    /*******************************************************************************************************/
    
    // Transform to numpy array object, remark: using INOUT array as pointing correction is made in place.
    PyArrayObject *pyArr_az =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_az, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    PyArrayObject *pyArr_alt =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_alt, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    PyArrayObject *pyArr_bs =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_bs, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    // Same as above, declaring this as IN array because we won't change anything in these arrays
    PyArrayObject *pyArr_tilt_para =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_tilt_para, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pyArr_tilt_perp =
        (PyArrayObject*)PyArray_FROM_OTF( pyObj_tilt_perp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    // Transform to C-Array
    double *az        =  PyArray_DATA( pyArr_az  );
    double *alt       =  PyArray_DATA( pyArr_alt );
    double *bs        =  PyArray_DATA( pyArr_bs  );
    double *tilt_para =  PyArray_DATA( pyArr_tilt_para );
    double *tilt_perp =  PyArray_DATA( pyArr_tilt_perp );

    // Get number of samples from dimension 0 of the numpy arrays
    int nsamples      = (int)PyArray_DIM( pyArr_az, 0 );
    /*******************************************************************************************************/
    
    /* Finally, apply the pointing correction to the data */
    /*******************************************************************************************************/
    libpointing_correct_pointing_stream( nsamples, az, alt, bs, tilt_para, tilt_perp, pm );
    /*******************************************************************************************************/

    /* Clean up */
    /*******************************************************************************************************/
    Py_DECREF( pyArr_az  );
    Py_DECREF( pyArr_alt );
    Py_DECREF( pyArr_bs  );
    Py_DECREF( pyArr_tilt_para );
    Py_DECREF( pyArr_tilt_perp );
    
    Py_DECREF( pyArr_az_coef  );
    Py_DECREF( pyArr_alt_coef );
    Py_DECREF( pyArr_bs_coef  );

    libpointing_free_pmodel( pm );
    /*******************************************************************************************************/
    
    Py_INCREF(Py_None);
    return Py_None;
}

