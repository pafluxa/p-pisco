#include <stdio.h>
#include <stdlib.h>

#include <Python.h>

#define NPY_NO_DEPRECATED_API   NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "pointing_routines.h"

static char module_docstring[] = "This is the todsynth.pointing C extension module.";

static PyObject* pointing_get_receiver_ICRS_coords( PyObject *self, PyObject *args );
static char pointing_get_receiver_ICRS_coords_ds[] = "";

static PyObject* pointing_transform_ICRS_to_horizontal_coords( PyObject *self, PyObject *args );
static char pointing_transform_ICRS_to_horizontal_coords_ds[] = "";

static PyObject* pointing_get_receiver_source_centered_coords( PyObject *self, PyObject *args );
static char pointing_get_receiver_source_centered_coords_ds[] = "";

static PyObject* pointing_get_receiver_ICRS_coords( PyObject *self, PyObject *args );
static char get_array_dims_ds[] = "Routine to get number of dimensions from an array.\n";

static PyObject* pointing_get_receiver_horizontal_coords( PyObject *self, PyObject *args );
static char pointing_get_receiver_horizontal_coords_ds[] = "";

static PyMethodDef module_methods[] = {
    
    {"transform_ICRS_to_horizontal_coords",
      pointing_transform_ICRS_to_horizontal_coords, 
      METH_VARARGS, 
      pointing_transform_ICRS_to_horizontal_coords_ds },
    
    {"get_receiver_source_centered_coords",
      pointing_get_receiver_source_centered_coords, 
      METH_VARARGS, 
      pointing_get_receiver_source_centered_coords_ds },
    
    {"get_receiver_horizontal_coords",
      pointing_get_receiver_horizontal_coords, 
      METH_VARARGS, 
      pointing_get_receiver_horizontal_coords_ds },

    {"get_receiver_ICRS_coords",
      pointing_get_receiver_ICRS_coords, 
      METH_VARARGS, 
      get_array_dims_ds },
	
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_pointing(void)
{
	PyObject *m = Py_InitModule3("_pointing", module_methods, module_docstring);
    if (m == NULL)
	{
        fprintf( stderr, "%s at %d : Could not initialize todsynth.pointing module\n", __FILE__, __LINE__ );
        return;
    }

    /* Load `numpy` functionality. */
    import_array();
}

static PyObject*
pointing_get_receiver_ICRS_coords(PyObject *dummy, PyObject *args)
{

    PyObject *pyObj_ctime;
    PyObject *pyObj_azimuth;
    PyObject *pyObj_altitude;
    PyObject *pyObj_rotation;
    PyObject *pyObj_bad_samples;

    PyObject *pyObj_det_dx;
    PyObject *pyObj_det_dy;
    PyObject *pyObj_det_pol_angle;

    PyObject *pyObj_det_ra;
    PyObject *pyObj_det_dec;
    PyObject *pyObj_det_pa;

    double site_lat;
    double site_lon;

    double xp;
    double yp;
    double dut1;
     
    // Parse input
    int err;
    err = PyArg_ParseTuple( args,
            "OOOOOOOOdddddOOO",
            &pyObj_ctime , &pyObj_azimuth, &pyObj_altitude, &pyObj_rotation, 
            &pyObj_bad_samples,
            &pyObj_det_dx, &pyObj_det_dy , &pyObj_det_pol_angle,
            &site_lat, &site_lon, &xp, &yp, &dut1,
            &pyObj_det_ra, &pyObj_det_dec, &pyObj_det_pa);
    
    if( !err )
	{
		fprintf( stderr, "%s at %d: there was an error parsing the input. RTFM!\n", __FILE__, __LINE__ );
        return NULL;
	}
    PyArrayObject *pyArr_ctime  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_ctime  ,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_azimuth =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_azimuth,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_altitude  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_altitude  ,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_rotation  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_rotation  ,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_bad_samples  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_bad_samples ,
        NPY_INT,
        NPY_ARRAY_IN_ARRAY );

    PyArrayObject *pyArr_det_dx =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_det_dx,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_det_dy =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_det_dy,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_det_pol_angle =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_det_pol_angle,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );

    PyArrayObject *pyArr_det_ra =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_det_ra,
        NPY_DOUBLE,
        NPY_ARRAY_OUT_ARRAY );
    PyArrayObject *pyArr_det_dec =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_det_dec,
        NPY_DOUBLE,
        NPY_ARRAY_OUT_ARRAY );
    PyArrayObject *pyArr_det_pa =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_det_pa,
        NPY_DOUBLE,
        NPY_ARRAY_OUT_ARRAY );
    
    // And finally, C-level woohoo!
    int nsamples        = (int)PyArray_DIM( pyArr_ctime , 0 );
    double *ctime       = (double *)PyArray_DATA(pyArr_ctime);
    double *azimuth     = (double *)PyArray_DATA(pyArr_azimuth);
    double *altitude    = (double *)PyArray_DATA(pyArr_altitude);
    double *rotation    = (double *)PyArray_DATA(pyArr_rotation);
    int    *bad_samples = (   int *)PyArray_DATA(pyArr_bad_samples);

    int ndets    = (int)PyArray_DIM( pyArr_det_dx, 0 );
    
	double *det_dx        = (double *)PyArray_DATA(pyArr_det_dx);
    double *det_dy        = (double *)PyArray_DATA(pyArr_det_dy);
    double *det_pol_angle = (double *)PyArray_DATA(pyArr_det_pol_angle);

    libpointing_get_receiver_ICRS_coords(

	    ndets, det_dx, det_dy, det_pol_angle,

        nsamples, ctime, azimuth, altitude, rotation, bad_samples,

        site_lat, site_lon, 0.0,

        xp, yp, dut1,
        PyArray_DATA(pyArr_det_ra),
        PyArray_DATA(pyArr_det_dec),
        PyArray_DATA(pyArr_det_pa ) );
    
	// clean up!
    Py_DECREF( pyArr_ctime    );
    Py_DECREF( pyArr_azimuth  );
    Py_DECREF( pyArr_altitude );
    Py_DECREF( pyArr_rotation );
    Py_DECREF( pyArr_bad_samples );
    Py_DECREF( pyArr_det_dx );
    Py_DECREF( pyArr_det_dy );
    Py_DECREF( pyArr_det_ra );
    Py_DECREF( pyArr_det_dec );
    Py_DECREF( pyArr_det_pa );
    
	Py_INCREF( Py_None );
	return Py_None;
}

static PyObject*
pointing_get_receiver_horizontal_coords( PyObject *self, PyObject *args )
{
    PyObject *pyObj_ctime;
    PyObject *pyObj_bore_az;
    PyObject *pyObj_bore_alt;
    PyObject *pyObj_bore_rot;
    
    PyObject *pyObj_recv_dx;
    PyObject *pyObj_recv_dy;
    PyObject *pyObj_recv_pol_angles;    

    PyObject *pyObj_recv_az;
    PyObject *pyObj_recv_alt;
    PyObject *pyObj_recv_rot;
    
    // Parse input
    int err;
    err = PyArg_ParseTuple( args,
            "OOOOOOOOOO", 
            &pyObj_ctime , &pyObj_bore_az, &pyObj_bore_alt, &pyObj_bore_rot,
            &pyObj_recv_dx, &pyObj_recv_dy, &pyObj_recv_pol_angles,
            &pyObj_recv_az, &pyObj_recv_alt, &pyObj_recv_rot );
    if( !err )
	{
		fprintf( stderr, "%s at %d: there was an error parsing the input. RTFM!\n", __FILE__, __LINE__ );
        return NULL;
	}

    // More parsing; to numpy arrays
    PyArrayObject *pyArr_ctime  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_ctime  ,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_bore_az =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_bore_az,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_bore_alt =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_bore_alt,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_bore_rot =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_bore_rot,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );

    PyArrayObject *pyArr_recv_dx =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_recv_dx,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_recv_dy =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_recv_dy,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_recv_pol_angles =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_recv_pol_angles,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );

    PyArrayObject *pyArr_recv_az =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_recv_az,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_recv_alt =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_recv_alt,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_recv_rot =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_recv_rot,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );

    // And finally, C-level woohoo!
    int nsamples      = (int)PyArray_DIM( pyArr_bore_az  , 0 );
    int ndets         = (int)PyArray_DIM( pyArr_recv_dx  , 0 );

    double *ctime      = (double *)PyArray_DATA(pyArr_ctime   );
    double *bore_az    = (double *)PyArray_DATA(pyArr_bore_az );
    double *bore_alt   = (double *)PyArray_DATA(pyArr_bore_alt);
    double *bore_rot   = (double *)PyArray_DATA(pyArr_bore_rot);
    
    double *recv_dx         = (double *)PyArray_DATA(pyArr_recv_dx);
    double *recv_dy         = (double *)PyArray_DATA(pyArr_recv_dy);
    double *recv_pol_angles = (double *)PyArray_DATA(pyArr_recv_pol_angles);
    
    double *recv_az  = (double *)PyArray_DATA(pyArr_recv_az );
    double *recv_alt = (double *)PyArray_DATA(pyArr_recv_alt);
    double *recv_rot = (double *)PyArray_DATA(pyArr_recv_rot);

    libpointing_get_receiver_horizontal_coords
    (
        ndets   , recv_dx, recv_dy, recv_pol_angles,
        nsamples, ctime, bore_az, bore_alt, bore_rot, 
        recv_az, recv_alt, recv_rot 
    );

	// clean up!
    Py_DECREF( pyArr_ctime    );
    Py_DECREF( pyArr_bore_az  );
    Py_DECREF( pyArr_bore_alt );
    Py_DECREF( pyArr_bore_rot );

    Py_DECREF( pyArr_recv_dx );
    Py_DECREF( pyArr_recv_dy );
    Py_DECREF( pyArr_recv_pol_angles );
 
    Py_DECREF( pyArr_recv_az  );
    Py_DECREF( pyArr_recv_alt );
    Py_DECREF( pyArr_recv_rot );
	Py_INCREF( Py_None );
	return Py_None;
}
static PyObject*
pointing_transform_ICRS_to_horizontal_coords( PyObject *self, PyObject *args )
{
    PyObject *pyObj_ctime;

    double ra;
    double dec;

    double site_lat;
    double site_lon;

    double xp;
    double yp;
    double dut1;
    
    PyObject *pyObj_azimuth;
    PyObject *pyObj_altitude;

    // Parse input
    int err;
    err = PyArg_ParseTuple( args,
            "OdddddddOO", 
            &pyObj_ctime , &ra, &dec,
            &site_lat, &site_lon, &xp, &yp, &dut1,
            &pyObj_azimuth, &pyObj_altitude );
    if( !err )
	{
		fprintf( stderr, "%s at %d: there was an error parsing the input. RTFM!\n", __FILE__, __LINE__ );
        return NULL;
	}
    // More parsing; to numpy arrays
    PyArrayObject *pyArr_ctime  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_ctime  ,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_azimuth =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_azimuth,
        NPY_DOUBLE,
        NPY_ARRAY_INOUT_ARRAY );
    PyArrayObject *pyArr_altitude  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_altitude  ,
        NPY_DOUBLE,
        NPY_ARRAY_INOUT_ARRAY );

    // And finally, C-level woohoo!
    int nsamples        = (int)PyArray_DIM( pyArr_ctime , 0 );
    double *ctime       = (double *)PyArray_DATA(pyArr_ctime);
    double *azimuth     = (double *)PyArray_DATA(pyArr_azimuth);
    double *altitude    = (double *)PyArray_DATA(pyArr_altitude);

    libpointing_transform_ICRS_to_horizontal_coords(
        nsamples, ctime, ra, dec,
        site_lat, site_lon, 0.0,
        xp, yp, dut1,
        azimuth,
        altitude );


	// clean up!
    Py_DECREF( pyArr_ctime    );
    Py_DECREF( pyArr_azimuth  );
    Py_DECREF( pyArr_altitude );

	Py_INCREF( Py_None );
	return Py_None;
}

static PyObject*
pointing_get_receiver_source_centered_coords( PyObject *self, PyObject *args )
{
    PyObject *pyObj_phi;
    PyObject *pyObj_theta;
    PyObject *pyObj_psi;
    
    PyObject *pyObj_phi_source;
    PyObject *pyObj_theta_source;

    PyObject *pyObj_phi_wrt_source;
    PyObject *pyObj_theta_wrt_source;
    
    // Parse input
    int err;
    err = PyArg_ParseTuple( args,
            "OOOOOOO", 
            &pyObj_phi,&pyObj_theta,&pyObj_psi,

            &pyObj_phi_source, &pyObj_theta_source,

            &pyObj_phi_wrt_source, &pyObj_theta_wrt_source );

    if( !err )
	{
		fprintf( stderr, "%s at %d: there was an error parsing the input. RTFM!\n", __FILE__, __LINE__ );
        return NULL;
	}
    // More parsing; to numpy arrays
    PyArrayObject *pyArr_phi  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_phi  ,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_theta  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_theta  ,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_psi  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_psi ,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
   
    PyArrayObject *pyArr_phi_source  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_phi_source ,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    PyArrayObject *pyArr_theta_source  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_theta_source ,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY );
    
    PyArrayObject *pyArr_phi_wrt_source  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_phi_wrt_source ,
        NPY_DOUBLE,
        NPY_ARRAY_OUT_ARRAY );
    PyArrayObject *pyArr_theta_wrt_source  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_theta_wrt_source ,
        NPY_DOUBLE,
        NPY_ARRAY_OUT_ARRAY );

    int ndets     = (int)PyArray_DIM( pyArr_phi , 0 );
    int nsamples  = (int)PyArray_DIM( pyArr_phi , 1 );
    
    double *phi   = (double *)PyArray_DATA(pyArr_phi);
    double *theta = (double *)PyArray_DATA(pyArr_theta);
    double *psi   = (double *)PyArray_DATA(pyArr_psi);

    double *phi_source   = (double *)PyArray_DATA(pyArr_phi_source  );
    double *theta_source = (double *)PyArray_DATA(pyArr_theta_source);

    double *phi_wrt_source   = (double *)PyArray_DATA(pyArr_phi_wrt_source  );
    double *theta_wrt_source = (double *)PyArray_DATA(pyArr_theta_wrt_source);
    
    libpointing_get_receiver_source_centered_coords
    (
        nsamples, ndets,
        theta, phi, psi,
        theta_source, phi_source,
        phi_wrt_source, theta_wrt_source
    );
        
	// clean up!
    Py_DECREF( pyArr_phi   );
    Py_DECREF( pyArr_theta );
    Py_DECREF( pyArr_psi   );
    
    Py_DECREF( pyArr_phi_source   );
    Py_DECREF( pyArr_theta_source );
    
    Py_DECREF( pyArr_phi_wrt_source   );
    Py_DECREF( pyArr_theta_wrt_source );

	Py_INCREF( Py_None );
	return Py_None;
}
