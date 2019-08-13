#ifndef __POINTING_CORR__                                                                                     
#define __POINTING_CORR__   

#define PIO2     1.5707963267948966                                                                               
#define DEG_RAD 57.2957795130823200 

struct POINTING_MODEL {                                                                                       
	
	double az_pc;                                                                                             
    double alt_pc;                                                                                            
    double x_center;                                                                                          
    double y_center;                                                                                          
    
	int n_az_coef;                                                                                            
    double *az_coef;                                                                                          
    
	int n_alt_coef;                                                                                           
    double *alt_coef;                                                                                         
    
	int n_bs_coef;                                                                                            
    double *bs_coef;                                                                                          
    
}; 

typedef struct POINTING_MODEL pmodel; 

pmodel* libpointing_alloc_pmodel                                                                                     
(                                                                                                             
    // Azimuth pointing correction                                                                            
    double az_pc,                                                                                             
    // Altitude pointing correction                                                                           
    double alt_pc,                                                                                            
    // Array center coordinates                                                                               
    double x_center, double y_center,                                                                         
    // Number of and azimuth correction coefficients                                                          
    double n_az_corr_coeff, double az_corr_coeff[],                                                           
    // Number of and altitude correction coefficients,                                                        
    double n_alt_corr_coeff, double alt_corr_coeff[],                                                         
    // Number of and boresight correction coefficients                                                        
    double n_bor_corr_coeff, double bor_corr_coeff[]                                                          
);

void   libpointing_free_pmodel( pmodel *pm );

void   libpointing_coef_correct(pmodel *pm, double az, double alt, double *azcor, double *altcor);

void   libpointing_offset_correct(pmodel *pm, double *azcor, double *elcor, double elcom, double bocom);

double libpointing_bore_offset(pmodel *pm, double alt, double bs);

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
);

#endif                                                                                                        
// vim: set sw=4 ts=4 et:
