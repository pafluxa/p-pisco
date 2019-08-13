#ifndef __SPHTRIGOH__
#define __SPHTRIGOH__

#define PI   (3.14159265358979323846264338327950288419716939937510582097494459230781)
#define PIO2 (PI/2.0)

void ra_dec_det( double dx, double dy,
                 double ra_bc, double dec_bc, double pa_bc, double rot_bc,
                 // output
                 double *ra_det, double *dec_det );

void dx_dy_det( double  ra_bc, double dec_bc, double  pa_bc,
                double rot_bc, double ra_det, double dec_det,
                // output
                double *dx, double *dy );

double pa_det( double dx, double dy, double dec_bc, double pa_bc, double rot_bc );
#endif
