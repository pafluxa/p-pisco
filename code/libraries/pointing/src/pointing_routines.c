#include <sofa.h>

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#include <sph_trigo.h>
#include <pointing_routines.h>

#ifndef M_PI_2
#define M_PI_2 (1.5707963267948966)
#endif

#ifndef M_PI
#define M_PI (2*1.5707963267948966)
#endif

#define ERFA_DAYSEC 86400.0
#define UNIX_TO_JD  2440587.5

void
libpointing_get_receiver_source_centered_coords
(
    int nsamples, int ndets,
    // Detector coordinates, in some coordinate system
    double det_lat[], double det_lon[], double det_pa[],
    // Source coordinates
    double source_lat[], double source_lon[],
    // Output coordinates
    double bearing[], double dist[]
)
{

    #pragma omp parallel
    {
    int det, sample;
    double phi, theta, psi, lat1, lon1, lat2, lon2; 
    double delta, x, y, c;
    
    #pragma omp for
    for( det=0; det < ndets; det++ )
    {
        for( sample=0; sample < nsamples; sample++ )
        {
            // Convenience definitions 
            psi  = det_pa    [ det*nsamples + sample ];
            lat2 = det_lat   [ det*nsamples + sample ];
            lon2 = det_lon   [ det*nsamples + sample ];
            
            lat1 = source_lat[ sample ];
            lon1 = source_lon[ sample ];

            // Compute angle Zenith - Det - Source
            delta = lon1 - lon2;
            x     = sin(delta) * cos( lat1 );
            y     = cos(lat2) * sin(lat1) - sin(lat2)*cos(lat1)*cos(delta);
            phi   = atan2( x, y );
            if( phi < 0 )
                phi += 2*M_PI;

            // Compute distance between Det and Source
            c = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(delta);
            theta = acos(c);
            
            dist[ det*nsamples + sample ] = theta;
            bearing[ det*nsamples + sample ] = phi - psi;
        }
    }

    }
}

void
libpointing_get_receiver_horizontal_coords
(
// Focal plane
int ndets, double det_dx[], double det_dy[], double det_pol_angle[],

// Boresight coordinates
int nsamples, double utc_ctime[], double azimuth[], double altitude[], double rotation[],

// Output: table of ndets x nsamples for detector right ascention, declination and parallactic angles.
double *det_az, double *det_alt, double *det_rot
) {

    #pragma omp parallel for
    for( int sample = 0; sample < nsamples; sample++ )
    {
        // Define Array Center Horizontal coordinates
        double az_ac  =  azimuth[sample];
        double alt_ac = altitude[sample];
        double rot_ac = rotation[sample];

        double za_    = M_PI_2 - alt_ac;

        for( int det = 0; det < ndets; det++ )
        {
            double cdx, sdx, cdy, sdy, r, sr, cr, alpha;
            cdx = cos(det_dx[det]);
            sdx = sin(det_dx[det]);
            cdy = cos(det_dy[det]);
            sdy = sin(det_dy[det]);

            cr  = cdx * cdy;
            r   = acos(cr);
            sr  = sin(r);
            alpha = atan2(sdy, sdx * cdy);

            double gamma = (M_PI_2 - alpha - rot_ac);
            double sg    = sin(gamma);
            double cg    = cos(gamma);

            double zaprime = za_;
            double szap    = sin(zaprime);
            double czap    = cos(zaprime);
            double za      = acos(cr * czap + sr * szap * cg);
            double delta   = atan2(sr * sg, cr * szap - sr * czap * cg);

            det_az [ det * nsamples + sample ] = az_ac  + -delta;
            det_alt[ det * nsamples + sample ] = alt_ac + (za-zaprime);
            det_rot[ det * nsamples + sample ] = rot_ac;
        }
    }
}

void
libpointing_get_receiver_ICRS_coords
(
// Focal plane
int ndets, double det_dx[], double det_dy[], double det_pol_angle[],

// Boresight coordinates
int nsamples, double ut1_ctime[], double azimuth[], double altitude[], double rotation[], int bad_samples[],

// Site specifications (longitude, latitude, height )
double site_latitude, double site_longitude, double site_height,

// Polar motions and UTC - UT1 difference (from astropy!)
double xp, double yp, double dut1,

// Output: table of ndets x nsamples for detector right ascention, declination and parallactic angles.
double *det_ra, double *det_dec, double *det_pa
) {

    // Initialize parallel region
    #pragma omp parallel
    {
    // Initialize SOFA Astrom structure
    iauASTROM  a_sofa;
    double    EO_sofa;    
    double astrom_update_interval = 1; // Update ASTRNOM structure each second
    double last_update = ut1_ctime[ 0 ];
    
    #pragma omp for
    for( int sample = 0; sample < nsamples; sample++ )
    {
        // Define Array Center Horizontal coordinates
    	double az_ac  =  azimuth[sample];
	    double alt_ac = altitude[sample];
        double rot_ac = rotation[sample];

        // Update IAUAstrom structure
        double ut1 = ut1_ctime[ sample ];
        /* Update the ASTROM context*/
        iauApco13( UNIX_TO_JD, ut1 / ERFA_DAYSEC, 
                   0.0,
                   site_longitude, site_latitude, site_height,
                   xp, yp,
                   0.0, 0.0, 0.0, 1.,
                  &a_sofa, &EO_sofa );

        a_sofa.refa = 0.0;
        a_sofa.refb = 0.0;
                
        /* Compute ICRS coordinates of zenith. */
        double cirs_ra, cirs_dec,dec_z,ra_z;
        iauAtoiq( "A", 
                   0.0, 0.0,
                   &a_sofa,
                   &cirs_ra, &cirs_dec );
        iauAticq( cirs_ra, cirs_dec,
                  &a_sofa,
                  &ra_z, &dec_z );
        
        /* Compute ICRS coordinates of Array Center. */
        double ra_ac, dec_ac, pa_ac;
        iauAtoiq( "A", az_ac, M_PI_2 - alt_ac,
                   &a_sofa,
                   &cirs_ra, &cirs_dec );
        iauAticq( cirs_ra, cirs_dec,
                  &a_sofa,
                  &ra_ac, &dec_ac ); 
        /* 
           Compute paralliactic angle of Array Center using the
           spherical triangle formed by NCP - Zenith - Array Center. 
        */
        double sdec  = sin( dec_ac );
        double cdec  = cos( dec_ac );
        double ypa   = sin( ra_z - ra_ac );
        double xpa   = cdec * tan( dec_z ) - sdec * cos( ra_z - ra_ac );
        pa_ac = atan2(ypa, xpa);

        for( int det = 0; det < ndets; det++ )                                                                
        {  
            /* detector ICRS coordinates */
            double dec_det, ra_det;
            ra_dec_det( det_dx[det], det_dy[det], 
                        ra_ac, dec_ac, pa_ac, rot_ac, 
                        &ra_det, &dec_det);                                                                                                   
            /* feed position angle wrt NCP  */                                                                           
            double _pa_det = pa_det( det_dx[det], det_dy[det], 
                                     dec_ac, pa_ac, rot_ac );
                                                                                              
            det_pa [ det * nsamples + sample ] = _pa_det;                                             
                                                                                                              
            det_ra [ det * nsamples + sample ] =  ra_det;                                                      
                                                                                                              
            det_dec[ det * nsamples + sample ] = dec_det;                                                     
        } 
    
    }

    } // end parallel region
}

void
libpointing_transform_ICRS_to_horizontal_coords
(
// Source coordinates
int nsamples, double ut1_ctime[], double source_ra, double source_dec,

// Site specifications (longitude, latitude, height )
double site_latitude, double site_longitude, double site_height,

// Polar motions and UTC - UT1 difference (from astropy!)
double xp, double yp, double dut1,

double source_azimuth[], double source_altitude[]
) {

    #pragma omp parallel
    {
    // Initialize SOFA Astrom structure
    iauASTROM  a_sofa;
    double    EO_sofa;
    double update_interval = 0.1;
    double last_update = ut1_ctime[ 0 ];
    #pragma omp for
    for( int sample = 0; sample < nsamples; sample++ )
    {
        // Update IAUAstrom structure
        double ut1 = ut1_ctime[ sample ];
        if( last_update + update_interval > ut1 )
        {
            /* Update the ASTROM context*/
            iauApco13( UNIX_TO_JD, ut1 / ERFA_DAYSEC, 
                       0.0,
                       site_longitude, site_latitude, site_height,
                       xp, yp,
                       0.0, 0.0, 0.0, 1.,
                      &a_sofa, &EO_sofa );

            a_sofa.refa = 0.0;
            a_sofa.refb = 0.0;
            
            last_update = ut1;
        } 
        else
        {
            iauAper13( UNIX_TO_JD, ut1/ERFA_DAYSEC, &a_sofa );
        }

        /* Transform from ICRS to CIRS*/
        double source_ra_CIRS, source_dec_CIRS;
        iauAtciq(
            source_ra, source_dec,
            0.0, 0.0,
            0.0,
            0.0,
            &a_sofa,
            &source_ra_CIRS, &source_dec_CIRS );

        /*Transform from CIRS to observerd az/alt*/
        double source_az, source_za, source_ha, source_obs_dec, source_obs_ra;
        iauAtioq(
            source_ra_CIRS, source_dec_CIRS,
            &a_sofa,
            &source_az, &source_za, &source_ha, &source_obs_dec, &source_obs_ra );

        source_azimuth [ sample ] = source_az;
        source_altitude[ sample ] = M_PI_2 - source_za;
        
    }

    }//end parallel region
}
