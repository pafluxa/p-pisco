#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pointing_correction.h"
#include <assert.h>

pmodel* libpointing_alloc_pmodel
( 
	// Azimuth pointing correction
	double az_pc, 
	// Altitude pointing correction
	double alt_pc, 
	// Array center coordinates
	double x_center, double y_center,
	// Number of and azimuth correction coefficients, in degrees
	double n_az_corr_coeff, double az_corr_coeff[],
	// Number of and altitude correction coefficients, in degrees
	double n_alt_corr_coeff, double alt_corr_coeff[],
	// Number of and boresight correction coefficients, in degrees
	double n_bor_corr_coeff, double bor_corr_coeff[]
) 
{
    pmodel *pm = (pmodel *)malloc( sizeof(pmodel) );
    
    if( n_az_corr_coeff > 10 )
    {
        fprintf( stderr, "\
            todsynth::libpointing::alloc_pmodel::\
            Cannot allocate a pointing with more than 10 terms. Using only the first 10.\n" );
        n_az_corr_coeff = 10;
    }   
    
    if( n_alt_corr_coeff > 10 )
    {
        fprintf( stderr, "\
            todsynth::libpointing::alloc_pmodel::\
            Cannot allocate a pointing with more than 10 terms. Using only the first 10.\n" );
        n_alt_corr_coeff = 10;
    }   
    
    if( n_bor_corr_coeff > 10 )
    {
        fprintf( stderr, "\
            todsynth::libpointing::alloc_pmodel::\
            Cannot allocate a pointing with more than 10 terms. Using only the first 10.\n" );
        n_bor_corr_coeff = 10;
    }   
 
    pm->n_az_coef  =  n_az_corr_coeff;
    pm->n_alt_coef = n_alt_corr_coeff;
    pm->n_bs_coef  = n_bor_corr_coeff;
   
    pm->az_pc      =    az_pc/DEG_RAD;
    pm->alt_pc     =   alt_pc/DEG_RAD;
    pm->x_center   = x_center/DEG_RAD;
    pm->y_center   = y_center/DEG_RAD;

    pm->az_coef    = (double *)malloc( pm->  n_az_coef*sizeof(double) );
    pm->alt_coef   = (double *)malloc( pm-> n_alt_coef*sizeof(double) );
    pm->bs_coef    = (double *)malloc( pm->  n_bs_coef*sizeof(double) );

    for (int i=0; i<pm->n_az_coef; i++)
    {   
        pm->az_coef[i] = (az_corr_coeff[i])/DEG_RAD;
    }
    for (int i=0; i<pm->n_alt_coef; i++)
    {                                                                     
        pm->alt_coef[i] = (alt_corr_coeff[i])/DEG_RAD;
    }
    for (int i=0; i<pm->n_bs_coef; i++)
    {                                                                     
        pm->bs_coef[i] = (bor_corr_coeff[i])/DEG_RAD;
    }

    return pm;
}

void libpointing_free_pmodel( pmodel *pm )
{
    
	free(pm->az_coef);                                                                                        
    free(pm->alt_coef);                                                                                       
    free(pm->bs_coef);                                                                                        
    free(pm); 
}

void libpointing_correct_pointing_stream
(
	// Number of samples in the pointing stream
	int nsamples, 
	// Boresight coordinates: azimuth, altitude and boresight rotation
	double *azimuth, double *altitude, double *rotation,
	// Tilt, parallel to boresight direction
	double *para_tilt,
	// Tilt, perpendicular to boresight direction
	double *perp_tilt,
	// Antenna pointing model
	pmodel *pm
)
{
    int i;
	double azcor = 0.0, altcor = 0.0, bscor = 0.0, altcent = 0.0;
    	
    for( i=0; i < nsamples; i++ )
    {    
        double az0   =  azimuth[i];
        double alt0  = altitude[i];
        double bs0   = rotation[i];

        double tilt_para = para_tilt[i];
        double tilt_perp = perp_tilt[i];

		double altmnt = alt0 - altcor - pm->alt_pc;  
        double altcur = altmnt - altcent;
        double azmnt  = az0 - azcor - pm->az_pc/cos(altmnt);
        double bscur  = bs0 - bscor;                                                                          
        double azcur, azcent;
        
        //printf( "tilt_para = %lf tilt_perp = %lf\n", tilt_para, tilt_perp ); 
        
        // Iterate 3 times
		for (int j=0; j<3; j++) 
		{                                                                             
			// Applt mount pointing model:
            libpointing_coef_correct(pm, azmnt, altmnt, &azcor, &altcor);
            //printf( "  azcor = %lf altcor = %lf bscor = %lf\n", azcor, altcor, bscor ); 
            
            // Perform level correction 
            azcor  -= tilt_para * tan(altmnt);
			altcor += tilt_perp;
			altmnt  = alt0 - altcor - pm->alt_pc;
            
            // Receiver center offset:                                                                        
			libpointing_offset_correct(pm, &azcent, &altcent, altmnt, bscur);
            //printf( "%lf %lf %lf %lf %lf\n", azcent, altcent, altmnt, azmnt, bscur ); 
			
            bscor   = libpointing_bore_offset(pm, altcur, bscur);                                                           
			azmnt   = az0 - azcor - pm->az_pc/cos(altcur);                                                     
            altcur  = altmnt - altcent;                                                                        
			bscur   = bs0 - bscor;                                                                             	
            azcur   = azmnt - azcent;
		}
        
        assert( !isnan( azcur ) );
        assert( !isnan( altcur ) );
        assert( !isnan( bscur ) );

		azimuth[i]  = azcur;
		altitude[i] = altcur;
		rotation[i] = bscur;
    }
	
}


void libpointing_coef_correct(pmodel *pm, double az, double alt, double *azcor, double *altcor) {                  
    
    double saz = sin(az);                                                                                     
    double caz = cos(az);                                                                                     
    double salt = sin(alt);                                                                                   
    double calt = cos(alt);                                                                                   
    double s2az = sin(2.0*az);                                                                                
    double c2az = cos(2.0*az);                                                                                
    double s2alt = sin(2.0*alt);                                                                              
    double c2alt = cos(2.0*alt);                                                                              
    double az_terms[11], alt_terms[11];                                                                       
    
    // Az terms                                                                                               
    // Pointing Model Offsets                                                                                 
    az_terms[0]  = 1.0;                                                                                       
    az_terms[1]  = salt;                                                                                      
    az_terms[2]  = calt;                                                                                      
    az_terms[3]  = calt * saz;                                                                                
    az_terms[4]  = calt * caz;                                                                                
    az_terms[5]  = salt * saz;                                                                                
    az_terms[6]  = salt * caz;                                                                                
    az_terms[7]  = calt * s2az;                                                                               
    az_terms[8]  = calt * c2az;                                                                               
    az_terms[9]  = salt * s2az;                                                                               
    az_terms[10] = salt * c2az;                                                                               
    
    // Alt terms                                                                                              
    // Pointing Model Offsets                                                                                 
    alt_terms[0]  = 1.0;                                                                                      
    alt_terms[1]  = calt / salt;                                                                              
    alt_terms[2]  = calt;                                                                                     
    alt_terms[3]  = salt;                                                                                     
    alt_terms[4]  = c2alt;                                                                                    
    alt_terms[5]  = s2alt;                                                                                    
    alt_terms[6]  = caz;                                                                                      
    alt_terms[7]  = saz;                                                                                      
    alt_terms[8]  = c2az;                                                                                     
    alt_terms[9]  = s2az;                                                                                     
    alt_terms[10] = 0.0;                                                                                      
                                                                                                              
    *azcor = 0.0;                                                                                             
    *altcor = 0.0;                                                                                            
    for (int i=0; i<10 && i<pm->n_az_coef; i++) {                                                             
        *azcor += pm->az_coef[i] * az_terms[i];                                                               
    }                                                                                                         
    for (int i=0; i<10 && i<pm->n_alt_coef; i++) {                                                            
        *altcor += pm->alt_coef[i] * alt_terms[i];                                                            
    }                                                                                                         
    
    *azcor /= calt;                                                                                           
    
    return;                                                                                                   
}

void libpointing_offset_correct(pmodel *pm, double *azcor, double *elcor, double elcom, double bocom) {            
    
    double r, sr, za, alpha, gamma, azoff, zaprime, cdx, sdx, cdy, sdy, cr, szap, czap, sg, cg;
    double dx = pm->x_center;                                                                                 
    double dy = pm->y_center;                                                                                 
    
    if((fabs(dx) < 1.0e-10) && (fabs(dy) < 1.0e-10)) {                                                        
        *azcor = 0.0;                                                                                         
        *elcor = 0.0;                                                                                         
        return;                                                                                               
    }                                                                                                         
                                                                                                              
    za = PIO2 - elcom;                                                                                        
    if(za < 1.0e-10)                                                                                          
        za = 1.0e-10;                                                                                         
                                                                                                              
    cdx = cos(dx);                                                                                            
    sdx = sin(dx);                                                                                            
    cdy = cos(dy);                                         
    sdy = sin(dy);

    // radial angle of beam from boresight axis in receiver coordinate system                                 
    cr = cdx * cdy;                                                                                           
    r = acos(cr);                                                                                             
    sr = sin(r);                                                                                              
    
    // angle of beam from x axis of receiver coordinate system, CCW looking out = +                           
    alpha = atan2(sdy, sdx * cdy);                                                                            
    
    // angle of beam from y axis in boresight coordinate system, CW looking out = +                           
    gamma = PIO2 - alpha - bocom;                                                                             
    cg = cos(gamma);                                                                                          
    sg = sin(gamma);                                                                                          
                                                                                                              
    zaprime = za;                                                                                             
    szap = sin(zaprime);                                                                                      
    czap = cos(zaprime);                                                                                      
    za = acos(cr * czap + sr * szap * cg);                                                                    
    azoff = atan2(sr * sg, cr * szap - sr * czap * cg);                                                       
                                                                                                              
    *azcor = -azoff;                                                                                          
    *elcor = za - zaprime;                                                                                    
} 

double libpointing_bore_offset(pmodel *pm, double alt, double bs) {                                                
    
    double term = 1.0;                                                                                        
    double el = alt * DEG_RAD - 45.0;                                                                         
    double coef[3] = {0.0, 0.0, 0.0};                                                                         
    
    if (el > 15.0)                                                                                            
        el = 15.0;                                                                                            
    
    if (el < -15.0)                                                                                           
        el = -15.0;                                                                                           
    
    for (int k=0; k<3; k++) {                                                                                 
        coef[0] += pm->bs_coef[k  ]*term;                                                                       
        coef[1] += pm->bs_coef[k+3]*term;                                                                     
        coef[2] += pm->bs_coef[k+6]*term;                                                                     
        term *= el;                                                                                           
    }                                                                                                         
    
    double bo = bs * DEG_RAD;                                                                                 
    double boff = 0.0;                                                                                        
    
    term = 1.0;                                                                                               
    for (int k=0; k<3; k++) {                                                                                 
        boff += coef[k] * term;                                                                               
        term *= bo;                                                                                           
    }                                                                                                         
    
    return boff;                                                                                              
}                                                                                                             
                                                                                                              
// vim: set sw=4 ts=4 et: 
