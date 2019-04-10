#ifndef __SPHTRIGOCUH__
#define __SPHTRIGOCUH__

#define PI   (3.14159265358979323846264338327950288419716939937510582097494459230781)
#define PIO2 (PI/2.0)

__device__
void ra_dec_pix( double dx, double dy, double ra_bc, double dec_bc, double pa_bc,
                 double rot_bc, double *ra_pix, double *dec_pix )
{
    double cdx = cos(dx);
    double sdx = sin(dx);
    double cdy = cos(dy);
    double sdy = sin(dy);
    double cr  = cdx * cdy;
    double r   = acos(cr);
    double sr  = sin(r);
    double alpha = atan2(sdy, sdx * cdy );
    double eta = pa_bc + rot_bc;
    double gamma = (PIO2 - alpha - eta);
    double sg    = sin(gamma);
    double cg    = cos(gamma);
    double sdecbc  = sin( dec_bc );
    double cdecbc  = cos( dec_bc );
    *dec_pix = asin ( cr * sdecbc + sr * cdecbc * cg);
    *ra_pix  = ra_bc - atan2( sr * sg, cr * cdecbc - sr * sdecbc * cg);
}

__device__
void dx_dy_pix(  double ra_bc, double dec_bc, double pa_bc,
                 double rot_bc, double ra_pix, double dec_pix, double *dx, double *dy )
{
    double cdc = cos(dec_bc);
    double sdc = sin(dec_bc);
    double cdp = cos(dec_pix);
    double sdp = sin(dec_pix);
    double delta = ra_bc - ra_pix;
    double sd = sin(delta);
    double cd = cos(delta);
    double cr  = sdc * sdp + cdc * cdp * cd;
    if( cr > 1.0 )
        cr = 1.0;
    double sr  = sqrt(1.0 - cr * cr); 
    
    double gamma = atan2(sd * cdp, sdp * cdc - cdp * sdc * cd);
    
    double eta = pa_bc + rot_bc;
    double alpha = PIO2 - gamma - eta;
    
    *dy = asin(sr * sin(alpha));
    *dx = asin(sr * cos(alpha) / cos(*dy));
}

__device__
double pa_pix( double dx, double dy, double dec_bc, double pa_bc, double rot_bc )
{
    double eta = pa_bc + rot_bc;
    double delta = atan2( cos(dec_bc)*sin(eta), sin(dec_bc) );
    double eps = delta - dx;
    double cl = cos(dec_bc)*cos(eta);
    if( cl > 1.0 )
        cl = 1.0;

    double sl = sqrt(1.0 - cl * cl);
    double p     = atan2( sl*sin(eps), cl*cos(dy) - sl*sin(dy)*cos(eps) );
    return p;
}

#endif
