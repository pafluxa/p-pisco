#ifndef __SKYCOORDSH__
#define __SKYCOORDSH__

#include <math.h>

#define PIO2 (1.5707963267948966)

__device__
void theta_phi_psi_pix(double *theta, double *phi, double *psi,
                       double  ra_bc, double dec_bc, double  psi_bc,
                       double ra_pix, double dec_pix )
/*   Calculates radial, polar offsets and position angle at a location
 *   offset from beam center. All outputs use the HEALPix coordinate system
 *   and the CMB polarization angle convention.
 *
 *   Outputs:  theta is the radial offset
 *             phi is the polar offset clockwise positive on the sky from South
 *             psi is the position angle clockwise positive from North
 *
 *   Inputs:   ra_bc   Right Ascension of beam center
 *             dec_bc  Declination of beam center
 *             psi_bc  Position angle of beam center clockwise positive from North
 *             ra_pix  Right Ascension of offset position
 *             dec_pix Declination of offset position
 */
    
{
  double cdc = cos(dec_bc);
  double sdc = sin(dec_bc);
  double cdp = cos(dec_pix);
  double sdp = sin(dec_pix);
  double dra = ra_pix - ra_bc;
  double sd = sin(dra);
  double cd = cos(dra);
  double spc = sin(psi_bc);
  double cpc = cos(psi_bc);
  // FEBeCoP Mitra, et. al. Formulas (3-11) (3-13) (3-16)
  double sdelx = cdp * sdc * cd - cdc * sdp;
  double sdely = cdp * sd;
  double xbeam =  sdelx * cpc + sdely * spc;
  double ybeam = -sdelx * spc + sdely * cpc;
  double sdx = xbeam;
  double cdx = sqrt(1.0 - sdx * sdx);
  double sdy = ybeam / cdx;
  double cdy = sqrt(1.0 - sdy * sdy);
  double cr = cdx * cdy;
  double cl = cdc*cpc;
  double sl_seps = cdc*spc*cdy - sdc * sdy;
  double sl_ceps = sdc*cdy + cdc*spc*sdy;
  *psi  = atan2(sl_seps, cl*cdx + sl_ceps*sdx);
  if(cr >= 1.0) {
    *theta = 0.0;
    *phi = 0.0;
  }
  else {
    *theta = acos(cr);
    *phi = atan2(ybeam, xbeam);
    if(*phi < 0.0)*phi += 2.0 * M_PI;
  }
}

__device__
void theta_phi_pa_pix( 
                       double *theta, double *phi, double *pa, 
                       double  ra_ac, double dec_ac, double  pa_ac, double rot_ac, 
                       double ra_pix, double dec_pix ) 
{                                                                                                             
    double cdc = cos(dec_ac);                                                                                     
    double sdc = sin(dec_ac);                                                                                     
    double cdp = cos(dec_pix);                                                                                     
    double sdp = sin(dec_pix);
    //double dra = ra_ac - ra_pix; 
    double dra = ra_pix - ra_ac;

    double sd = sin(dra);
    double cd = cos(dra);
  
    double sdelx = cdp * sdc * cd - cdc * sdp;
    double sdely = cdp * sd;
    double eta = pa_ac + rot_ac;
    double seta = sin(eta);
    double ceta = cos(eta);
    // FEBeCoP Mitra, et. al. Formula (3-16)
    double xbeam =  sdelx * ceta + sdely * seta;
    double ybeam = -sdelx * seta + sdely * ceta;
    double sdy = -xbeam;
    double cdy = sqrt(1.0 - sdy * sdy);
    double sdx = ybeam / cdy;
    double cdx = sqrt(1.0 - sdx * sdx);
    double cr = cdx * cdy;
    double cl = cdc*ceta;
    double sl_seps = cdc*seta*cdx - sdc * sdx;
    double sl_ceps = sdc*cdx + cdc*seta*sdx;
    *pa  = atan2(sl_seps, cl*cdy - sl_ceps*sdy);
    if(cr > 1.0)*theta = 0.0;
    else *theta = acos(cr);
    *phi = atan2(-xbeam, ybeam);
    if(*phi < 0.0)*phi += 2.0 * M_PI;                         
}

__device__
void theta_phi_pa_pix_rdra( 
                       double *theta, double *phi, double *pa, 
                       double  ra_ac, double dec_ac, double  pa_ac, double rot_ac, 
                       double ra_pix, double dec_pix ) 
{                                                                                                             
    double cdc = cos(dec_ac);                                                                                     
    double sdc = sin(dec_ac);                                                                                     
    double cdp = cos(dec_pix);                                                                                     
    double sdp = sin(dec_pix);
    double dra = ra_ac - ra_pix;
    
    dra = -dra;

    double sd = sin(dra);
    double cd = cos(dra);
  
    double sdelx = cdp * sdc * cd - cdc * sdp;
    double sdely = cdp * sd;
    double eta = pa_ac + rot_ac;
    double seta = sin(eta);
    double ceta = cos(eta);
    // FEBeCoP Mitra, et. al. Formula (3-16)
    double xbeam =  sdelx * ceta + sdely * seta;
    double ybeam = -sdelx * seta + sdely * ceta;
    double sdy = -xbeam;
    double cdy = sqrt(1.0 - sdy * sdy);
    double sdx = ybeam / cdy;
    double cdx = sqrt(1.0 - sdx * sdx);
    double cr = cdx * cdy;
    double cl = cdc*ceta;
    double sl_seps = cdc*seta*cdx - sdc * sdx;
    double sl_ceps = sdc*cdx + cdc*seta*sdx;
    *pa  = atan2(sl_seps, cl*cdy - sl_ceps*sdy);
    if(cr > 1.0)*theta = 0.0;
    else *theta = acos(cr);
    *phi = atan2(-xbeam, ybeam);
    if(*phi < 0.0)*phi += 2.0 * M_PI;                         
}

__device__
void dx_dy_pa_pix( double *dx, double *dy, double *pa, double ra_ac, double dec_ac,                             
                   double pa_ac, double rot_ac, double ra_pix, double dec_pix )                                                                            
{                                                                                                             
    double cdc = cos(dec_ac);                                                                                     
    double sdc = sin(dec_ac);                                                                                     
    double cdp = cos(dec_pix);                                                                                     
    double sdp = sin(dec_pix);
    double dra = ra_ac - ra_pix;
    
    double sd = sin(dra);
    double cd = cos(dra);
  
    double sdelx = cdp * sdc * cd - cdc * sdp;
    double sdely = cdp * sd;
    double eta = pa_ac + rot_ac;
    double seta = sin(eta);
    double ceta = cos(eta);
    // FEBeCoP Mitra, et. al. Formula (3-16)
    double xbeam =  sdelx * ceta + sdely * seta;
    double ybeam = -sdelx * seta + sdely * ceta;
    double sdy = -xbeam;
    double cdy = sqrt(1.0 - sdy * sdy);
    double sdx = ybeam / cdy;
    double cdx = sqrt(1.0 - sdx * sdx);

    double cl = cdc*ceta;
    double sl_seps = cdc*seta*cdx - sdc * sdx;
    double sl_ceps = sdc*cdx + cdc*seta*sdx;
   
    *pa = atan2(sl_seps, cl*cdy - sl_ceps*sdy);                         
    *dy =  asin(sdy);
    *dx =  asin(sdx);
}


__host__ __device__
void ra_dec_pix( // Input
                 double dx, double dy,
                 double ra_bc, double dec_bc, double pa_bc,
                 // Output
                 double  *ra_pix,
                 double *dec_pix )
{
    double cdx = cos(dx);
    double sdx = sin(dx);
    double cdy = cos(dy);
    double sdy = sin(dy);

    double cr  = cdx * cdy;
    double r   = acos(cr);
    double sr  = sin(r);
    double alpha = atan2(sdy, sdx * cdy );

    double gamma = (PIO2 - alpha - pa_bc);
    double sg    = sin(gamma);
    double cg    = cos(gamma);

    double sdecbc  = sin( dec_bc );
    double cdecbc  = cos( dec_bc );

    *dec_pix = asin ( cr * sdecbc + sr * cdecbc * cg);
    *ra_pix  = ra_bc + atan2( sr * sg, cr * cdecbc - sr * sdecbc * cg);

}

__host__ __device__
void dx_dy_pix_rdelta( 
                // Input
                double ra_bc, double dec_bc, double pa_bc,
                double ra_pix, double dec_pix,
                // Output
                double *dx, double *dy )

{
    double cdc = cos(dec_bc);
    double sdc = sin(dec_bc);
    double cdp = cos(dec_pix);
    double sdp = sin(dec_pix);
    double delta = ra_bc - ra_pix;
    
    double sd = sin(delta);
    double cd = cos(delta);

    double cr  = sdc * sdp + cdc * cdp * cd;
    if( cr >  1.0 ) cr =  1.0;
    if( cr < -1.0 ) cr = -1.0;

    double r   = acos(cr);
    double sr  = sin(r);
    double gamma = atan2(sd * cdp, sdp * cdc - cdp * sdc * cd);

    double alpha = PIO2 - gamma - pa_bc;
    *dy = asin(sr * sin(alpha));
    *dx = asin(sr * cos(alpha) / cos(*dy));
}

/*
__host__ __device__
double pa_pix( double dx, double dy, double dec_bc, double pa_bc )
{
    double delta = atan2( cos(dec_bc)*sin(pa_bc), sin(dec_bc) );
    double     L =  acos( cos(dec_bc)*cos(pa_bc) );
    double     p = atan2( sin(dx+delta), cos(dy)*1./tan(L) - sin(dy)*cos(dx+delta) );
    return p;
}
*/

__device__
double pa_pix( double dx, double dy, double dec_ac, double pa_ac )
{  
    double eta = pa_ac;
    double delta = atan2( cos(dec_ac)*sin(eta), sin(dec_ac) ); 
    double eps = delta - dx;
    double cl = cos(dec_ac)*cos(eta);
    
    if( cl < -1.0 )
        cl = -1.0;
    if( cl > 1.0 )
        cl = 1.0;

    double sl = sqrt(1.0 - cl * cl);                                                           
    double p  = atan2( sl*sin(eps), cl*cos(dy) - sl*sin(dy)*cos(eps) );                         
                                                                                                              
    return p;                                                                                                 
} 

#endif
