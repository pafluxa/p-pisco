#ifndef __COORDUTILSH__
#define __COORDUTILSH__

// Thanks Mike!!!!
//    COORDINATE ROTATION USING SPHERICAL TRIANGLES
__host__ __device__ void
coordutils_rotate_coords(
//  AO,BO = INPUT:  ORIGIN OF A2 IN A1,B1 COORDINATE SYSTEM
    double ao, double bo,
// AP,BP = INPUT:  POLE OF B2 IN A1,B1 COORDINATE SYSTEM
    double ap, double bp,
// A1,B1 = INPUT:  COORDINATES IN A1,B1 COORDINATE SYSTEM
    double a1, double b1,
// A2,B2 = OUTPUT: COORDINATES IN A2,B2 COORDINATE SYSTEM
    double *a2, double *b2)
{
    double sbo, cbo, sbp, cbp, sb1, cb1, sb2, cb2, saa, caa, sbb, cbb;
    double sa2, ca2, ta2o2;
    sbo = sin(bo);
    cbo = cos(bo);
    sbp = sin(bp);
    cbp = cos(bp);
    sb1 = sin(b1);
    cb1 = cos(b1);
    sb2 = sbp * sb1 + cbp * cb1 * cos(ap - a1);
    cb2 = sqrt((1.0 - sb2) * (1.0 + sb2));
    if(cb2 == 0.0)cb2 = 1.0e-30;
    *b2 = atan2(sb2, cb2);
    saa = sin(ap - a1) * cb1 / cb2;
    caa = (sb1 - sb2 * sbp) / (cb2 * cbp);
    sbb = sin(ap - ao) * cbo;
    cbb = sbo / cbp;
    sa2 = saa * cbb - caa * sbb;
    ca2 = caa * cbb + saa * sbb;
    if(ca2 > 0.0)ta2o2 = sa2 / (1.0 + ca2);
    else ta2o2 = (1.0 - ca2) / sa2;
    *a2 = 2.0 * atan(ta2o2);
    if(*a2 < 0.0)*a2 += TWOPI;
}

__host__ __device__ double
coordutils_haversin( double t )
{

    double a = sinf(t/2.0);

    return a*a;
}

__host__ __device__ double
coordutils_position_angle( double ra1, double  dec1, double ra2, double dec2 )
{

    double radif = ra2 - ra1;
    double angle = radif * sin(dec1);

    return angle;
}

__host__ __device__ void 
coordutils_outter_product( double v[], double u[], double p[3][3] )           
{                                                                       
    int _i, _j;                                                         
    for (_i = 0; _i < 3; _i++)                                       
        for (_j = 0; _j < 3; _j++)                                   
            p[_i][_j] = v[_i] * u[_j];                            
}

__host__ __device__ double
coordutils_norm ( double3 a )
{
    return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

__host__ __device__ double3
coordutils_normalize( double3 a )
{
    double n = rsqrt( a.x*a.x + a.y*a.y + a.z*a.z );
    return make_double3( a.x*n, a.y*n, a.z*n );
}

__host__ __device__ double
coordutils_dot  ( double3 a, double3 b )
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ double3
coordutils_cross( double3 a, double3 b )
{
    return make_double3( a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__host__ __device__ void
coordutils_multiply_3x3_matrix( double A[3][3], double B[3][3], double R[3][3] )
{
	double sum;
    int i,j,k;
    for (i = 0; i <= 2; i++) {
        for (j = 0; j <= 2; j++) {
            sum = 0;
            for (k = 0; k <= 2; k++) {
                sum = sum + A[i][k] * B[k][j];
            }

        R[i][j] = sum;

        }
    }

}

__host__ __device__ void
coordutils_square_3x3_matrix( double A[3][3], double R[3][3] )
{
	double sum = 0;
	int c,d,k;
	for (c = 0; c < 3; c++) {
		for (d = 0; d < 3; d++) {
			for (k = 0; k < 3; k++) {
				sum = sum + A[c][k]*A[k][d];
			}

			R[c][d] = sum;
			sum = 0;
		}
	}
}

__host__ __device__ void
coordutils_make_3x3_identity_matrix( double A[3][3] )
{
    // Avoid for loops like the plague
	A[0][0] = 1.0f;
    A[1][1] = 1.0f;
    A[2][2] = 1.0f;

    A[0][1] = 0.0f;
    A[0][2] = 0.0f;
    A[1][0] = 0.0f;
    A[1][2] = 0.0f;
    A[2][0] = 0.0f;
    A[2][1] = 0.0f;
}

__host__ __device__ void
coordutils_make_3x3_zero_matrix( double A[3][3] )
{
    // Avoid for loops like the plague
	A[0][0] = 0.0f;
    A[0][1] = 0.0f;
    A[0][2] = 0.0f;
    A[1][1] = 0.0f;
    A[1][0] = 0.0f;
    A[1][2] = 0.0f;
    A[2][0] = 0.0f;
    A[2][1] = 0.0f;
    A[2][2] = 0.0f;
}

__host__ __device__ void
coordutils_scale_3x3_matrix( double A[3][3], double c )
{
    // Avoid for loops like the plague
	A[0][0] *= c;
    A[0][1] *= c;
    A[0][2] *= c;
    A[1][0] *= c;
    A[1][1] *= c;
    A[1][2] *= c;
    A[2][0] *= c;
    A[2][1] *= c;
    A[2][2] *= c;
}

__host__ __device__ void
coordutils_add_3x3_matrices( double A[3][3], double B[3][3], double R[3][3] )
{
    // Avoid for loops like the plague
	R[0][0] = A[0][0] + B[0][0];
    R[0][1] = A[0][1] + B[0][1];
    R[0][2] = A[0][2] + B[0][2];
    R[1][0] = A[1][0] + B[1][0];
    R[1][1] = A[1][1] + B[1][1];
    R[1][2] = A[1][2] + B[1][2];
    R[2][0] = A[2][0] + B[2][0];
    R[2][1] = A[2][1] + B[2][1];
    R[2][2] = A[2][2] + B[2][2];
}

__host__ __device__ double3
coordutils_compute_vector_times_3x3_matrix ( double3 v, double R[3][3] )
{
    double x = R[0][0]*v.x + R[0][1]*v.y + R[0][2]*v.z;
    double y = R[1][0]*v.x + R[1][1]*v.y + R[1][2]*v.z;
    double z = R[2][0]*v.x + R[2][1]*v.y + R[2][2]*v.z;

    return make_double3( x, y, z );
}

__host__ __device__ void
coordutils_make_rot_matrix_a_to_b( double3 a, double3 b, double R[3][3] )
{
    // Compute cross and dot products (sin and cos of angles between a and b)
    // which are assumed to be unit vectors!!
    double3 v = coordutils_cross( a, b );
    double  s = coordutils_norm ( v );
    double  c = coordutils_dot  ( a, b );

    // Initialize matrices
    double  vx[3][3]; coordutils_make_3x3_zero_matrix( vx );
    double   I[3][3]; coordutils_make_3x3_identity_matrix( I );
    double vx2[3][3]; coordutils_make_3x3_zero_matrix( vx2 );
    double tmp[3][3]; coordutils_make_3x3_zero_matrix( tmp );

    // Zero out input
    coordutils_make_3x3_zero_matrix(   R );

    // Compute vx
    vx[0][1] = -v.z; vx[0][2] =  v.y;
    vx[1][0] =  v.z; vx[1][2] = -v.x;
    vx[2][0] = -v.y; vx[2][1] =  v.x;

    // Compute vx2
    coordutils_square_3x3_matrix( vx, vx2 );
    // Scale vx2
    coordutils_scale_3x3_matrix( vx2, 1./(1+c) );
    // Add vx to vx2 scaled to tmp
    coordutils_add_3x3_matrices( vx, vx2, tmp );
    // Add tmp to I
    coordutils_add_3x3_matrices( I, tmp, R );
}

__host__ __device__ void
coordutils_make_zxz_rotation_matrix( double psi, double theta, double sigma, double R[3][3] )
{
    double cz0 = cos( psi );
    double sz0 = sin( psi );

    double cx = cos( theta );
    double sx = sin( theta );

    double cz1 = cos( sigma );
    double sz1 = sin( sigma );

    coordutils_make_3x3_zero_matrix( R );
    R[0][0] =  cz0*cz1 - cx*sz0*sz1;
    R[0][1] = -cx*cz1*sz0 - cz0*sz1;
    R[0][2] =  sx*sz0;

    R[1][0] = cz1*sz0 + cx*cz0*sz1;
    R[1][1] = cz1*cz0*cz1 - sz0*sz1;
    R[1][2] =-sx*cz0;

    R[2][0] = sx*sz1;
    R[2][1] = sx*cz1;
    R[2][2] = cx;
}

__host__ __device__ void
coordutils_transpose_3x3_matrix( double R[3][3], double RT[3][3] )
{
    int c,d;
    for (c = 0; c < 3; c++)
      for( d = 0 ; d < 3 ; d++ )
         RT[d][c] = R[c][d];
}

__host__ __device__ void
coordutils_make_zyz_rotation_matrix( double alpha, double beta, double gamma, double R[3][3] )
{
    double cz0 = cos( alpha );
    double sz0 = sin( alpha );

    double cx = cos( beta );
    double sx = sin( beta );

    double cz1 = cos( gamma );
    double sz1 = sin( gamma );

    coordutils_make_3x3_zero_matrix( R );
    R[0][0] = cz0*cz1 - cx*sz0*sz1;
    R[0][1] = -cx*cz1*sz0 - cz0*sz1;
    R[0][2] = sx*sz0;

    R[1][0] = cz1*sz0 + cx*cz0*sz1;
    R[1][1] = cx*cz0*cz1 - sz0*sz1;
    R[1][2] = -sx*cz0;

    R[2][0] = sx*sz1;
    R[2][1] = sx*cz1;
    R[2][2] = cx;
}

__host__ __device__ void
coordutils_get_euler_angles_from_zyz_rot_matrix( double R[3][3], double *alpha, double *beta, double *gamma )
{
    double r02,r10,r11,r12,r20,r21,r22;

    r02 = R[0][2];

    r10 = R[1][0];
    r11 = R[1][1];
    r12 = R[1][2];

    r20 = R[2][0];
    r21 = R[2][1];
    r22 = R[2][2];

    if(r22 < +1 )
    {
        if( r22 > -1 )
        {
            *beta  = acos( r22 );
            *alpha = atan2( r12,  r02 );
            *gamma = atan2( r21, -r20 );
        }
        else
        {
            *beta = M_PI;
            *alpha = -atan2( -r10, r11 );
            *gamma = 0;
        }
    }
    else
    {
        *beta = 0;
        *alpha = atan2( -r10, r11 );
        *gamma = 0;
    }

    if( *alpha < 0 )
        *alpha += 2*M_PI;
}

__host__ __device__ void
coordutils_get_euler_angles_from_zxz_rot_matrix( double R[3][3], double *alpha, double *beta, double *gamma )
{
    double r00,r01,r02,r12,r20,r21,r22;

    r00 = R[0][0];
    r01 = R[0][1];
    r02 = R[0][2];

    r12 = R[1][2];

    r20 = R[2][0];
    r21 = R[2][1];
    r22 = R[2][2];

    if(r22 < 1 )
    {
        if( r22 > -1 )
        {
            *beta  = acos( r22 );
            *alpha = atan2( r02, -r12 );
            *gamma = atan2( r20, r21 );
        }
        else
        {
            *beta = M_PI;
            *alpha = -atan2( -r01, r00 );
            *gamma = 0;
        }
    }
    else
    {
        *beta = 0;
        *alpha = atan2( -r01, r00 );
        *gamma = 0;
    }

    if( *alpha < 0 )
        *alpha += 2*M_PI;
}


#endif
