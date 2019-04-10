#ifndef _ROTATIONSH
#define _ROTATIONSH

__device__ void
alpha_r_gamma_from_dx_dy( double dx, double dy, 
                          double *alpha, double *r, double *gamma )
{

    double cdx = cos(dx);
    double sdx = sin(dx);
    double cdy = cos(dy);
    double sdy = sin(dy);

    double cr  = cdx * cdy;

    *r = acos(cr);
    *alpha = atan2(sdy, sdx * cdy );
    *gamma = atan2( cdx * sdy, sdx);
}

__device__ double
dot_product( double3 a, double3 b )
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ double3
cross_product( double3 a, double3 b )
{
    double3 c;

    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;

    return c;
}

__device__ double3
angle_axis_rot( double3 vec, double angle, double3 axis )
{
    double3 rot_vec;
    //cos(psi)*x + sin(psi)*( cross(p,x) ) + (1-cos(psi))*dot(p,x)*p
    
    double3 c1;
    c1 = cross_product( axis, vec );

    double  d1;
    d1 = dot_product( axis, vec );
    
    double cpsi;
    double spsi;
    cpsi = cos(psi);
    spsi = sin(psi);

    rot_vec.x = cpsi * vec.x + spsi * c1.x + (1 - cpsi)*d1*axis.x;
    rot_vec.y = cpsi * vec.y + spsi * c1.y + (1 - cpsi)*d1*axis.y;
    rot_vec.z = cpsi * vec.z + spsi * c1.z + (1 - cpsi)*d1*axis.z;
    
    return rot_vec;
}

__device__ void
build_antenna_basis( double theta, double phi, double psi, 
                     double3 *xa, double3 *ya, double3 *za )
{
    // \hat{r}
    double3 hat_r;
    hat_r.x = sin(theta)*cos(phi);
    hat_r.y = sin(theta)*sin(phi);
    hat_r.z = cos(theta);
    
    // \hat{\theta}
    double3 hat_theta;
    hat_theta.x =  cos(theta)*cos(phi);
    hat_theta.y =  cos(theta)*sin(phi);
    hat_theta.z = -sin(theta);

    // \hat{\phi}
    double3 hat_phi;
    hat_phi.x = -sin(phi);
    hat_phi.y =  cos(phi);
    hat_phi.z =  0.0;

    // Get antenna basis vectors
    double3 hat_rho, hat_sigma, hat_za;
    hat_rho   = axis_angle_rot( hat_theta, psi, hat_r );
    hat_sigma = axis_angle_rot( hat_phi  , psi, hat_r );
    hat_za    = cross_product( hat_rho, hat_sigma );
    
    // return
    xa->x = hat_sigma.x;
    xa->y = hat_sigma.y;
    xa->z = hat_sigma.z;
    
    (*ya).x = hat_rho.x;
    (*ya)->y = hat_rho.y;
    (*ya_->z = hat_rho.z;
    
    za->x = hat_za.x;
    za->y = hat_za.y;
    za->z = hat_za.z;
}

__device__ void
build_rho_sigma_vectors( double rho, double sigma, double3 xa, double3 ya, double3 za,
                         double3 *rho, double3 *sigma )
{
    
    double _x, _y, _z;
    _x =  cos(rho)*cos(sigma);
    _y =  cos(rho)*sin(sigma);
    _z = -sin(rho);
    
    rho->x = _x*xa.x + _y*ya.x + _z*za.x;
    rho->y = _x*xa.y + _y*ya.y + _z*za.y;
    rho->z = _x*xa.z + _y*ya.z + _z*za.z;
    
    double ssigma,csigma;
    ssigma = sin(sigma);
    csigma = cos(sigma);
    
    sigma->x = -ssigma*xa.x + csigma*ya.x + 0.0*za.x;
    sigma->y = -ssigma*xa.y + csigma*ya.y + 0.0*za.y;
    sigma->z = -ssigma*xa.z + csigma*ya.z + 0.0*za.z;
}

__device__ double
get_chi( double3 rho, double3 sigma, double alpha, double theta, double phi )
{
    // \hat{\theta}
    double3 hat_theta;
    hat_theta.x =  cos(theta)*cos(phi);
    hat_theta.y =  cos(theta)*sin(phi);
    hat_theta.z = -sin(theta);
    
    double3 e_co;
    e_co.x = rho.x * sin(alpha) + sigma.x * cos(alpha);
    e_co.y = rho.y * sin(alpha) + sigma.y * cos(alpha);
    e_co.z = rho.z * sin(alpha) + sigma.z * cos(alpha);
    
    double cos_chi = dot_product( e_co, hat_theta );

    return acos( chi );
}

#endif
