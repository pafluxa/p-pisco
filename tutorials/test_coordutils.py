# coding: utf-8

import numpy
from numpy import sin,cos
from numpy import arccos, arctan2

    
def coordutils_rot_matrix_a_to_b( v1, v2 ):
    
    v = numpy.cross( v1, v2 )
    s = numpy.linalg.norm(v)
    c = numpy.dot( v1, v2 )

    vx = numpy.asarray( [
        [ 0.0, -v[2], v[1] ], 
        [ v[2], 0.0, -v[0] ],
        [-v[1], v[0], 0.0  ] ] )

    R  = numpy.eye(3) + vx + numpy.dot( vx, vx ) * 1./(1+c)

    return R

def coordutils_compute_vector_times_matrix( v, R ):

    x = R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2]
    y = R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2]
    z = R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2]

    return numpy.asarray( (x,y,z) )

def coordutils_compute_vector_from_angles( phi, theta ):

    v = numpy.zeros( 3 )

    v[0] = cos(phi)*cos(theta)
    v[1] = sin(phi)*cos(theta)
    v[2] = sin(theta)

    return v

# spherical coordinates
phi1 = 0.1
tht1 = 0.4
phi2 = 0.2
tht2 = 0.6

# create vectors
v1 = coordutils_compute_vector_from_angles( phi1, tht1 )
print v1

v2 = coordutils_compute_vector_from_angles( phi2, tht2 )
print v2

R = coordutils_rot_matrix_a_to_b( v2, v1 )

v1_from_v2 = coordutils_compute_vector_times_matrix( v2, R )
print v1_from_v2


