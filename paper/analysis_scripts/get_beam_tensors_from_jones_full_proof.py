#!/usr/bin/env python
import sympy

# "I" is the imaginary unit, not the identity matrix
from sympy import I, Matrix, symbols, conjugate, trace, simplify, cos, sin, collect
from sympy.physics.quantum import Dagger

def compute_beamsor_from_E( E_x_co, E_x_cx, E_y_co, E_y_cx ):
    '''
    '''

    # define Pauli matrices. I am using I,Q,U,V for clarity in the code.
    # s0
    sI = Matrix( ([1, 0],[0 ,1]) )
    sQ = Matrix( ([1, 0],[0,-1]) )
    sU = Matrix( ([0, 1],[1, 0]) )
    sV = Matrix( ([0,-I],[I, 0]) ) 

    # Define Jones matrix as described by MKB.
    J = Matrix( ( [E_x_co,E_x_cx], [-E_y_cx, E_y_co] ) )

    # Define the J^\dag 
    Jd = Dagger( J )

    # Compute B
    B = []
    for s1 in [sI,sQ,sU,sV]:
        s1J = s1 * J
        for s2 in [sI,sQ,sU,sV]:
            s2Jd = s2 * Jd
            B.append( 0.5*trace(s1J * s2Jd) )

    M = Matrix( ( [ B[0], B[1], B[2], B[3] ], 
                  [ B[4], B[5], B[6], B[7] ], 
                  [ B[8], B[9], B[10], B[11] ], 
                  [ B[12], B[13], B[14], B[15] ] ) )
    return M

# define symbols representing electric fields
Ex_co, Ex_cx, Ey_co, Ey_cx = symbols( 'Ex_co Ex_cx Ey_co Ey_cx' )
two_zeta, two_psi, eps = symbols('2zeta 2psi eps')
L = Matrix( ( [1,0,0,0], [0,cos(two_psi),-sin(two_psi),0], [0,sin(two_psi),cos(two_psi),0], [0,0,0,1] ) )
Z = Matrix( ( [1,0,0,0], [0,cos(two_zeta),-sin(two_zeta),0], [0,sin(two_zeta),cos(two_zeta),0], [0,0,0,1] ) )
T, Q, U, V = symbols('T Q U V')
S = Matrix( ( [T, Q, U, 0] ) )


M = compute_beamsor_from_E( Ex_co, Ex_cx, Ex_co, Ex_cx )
#ZMZ = Z * M * Z.transpose()
#M = simplify(ZMZ)

# set E_x_co = E_y_co, and E_x_cx = E_y_cx = 0
#E_x_cx = E_y_cx
#E_y_co = E_x_co

#B = compute_beamsor_from_E( E_x_co, E_x_cx, E_y_co, E_y_cx )

print( "" )
print( M[0,0] )
print( M[0,1] )
print( M[0,2] )
print( M[0,3] )

print( "" )
print( M[1,0] )
print( M[1,1] )
print( M[1,2] )
print( M[1,3] )

print( "" )
print( M[2,0] )
print( M[2,1] )
print( M[2,2] )
print( M[2,3] )

print( "" )
print( M[3,0] )
print( M[3,1] )
print( M[3,2] )
print( M[3,3] )


#Pedro's Ex
ML = M * L.transpose()
MLS = simplify(ML * S)
D = Matrix( ([1, cos(two_zeta), sin(two_zeta),0]) )
D1p = D.dot(MLS)

#Pedro's Ey
M = compute_beamsor_from_E( Ey_co, Ey_cx, Ey_co, Ey_cx )
ML = M * L.transpose()
MLS = simplify(ML * S)
D = Matrix( ([1, -cos(two_zeta), -sin(two_zeta),0]) )
D2p = D.dot(MLS)

print('Detector+ (Ey = Ex)')
D1 = simplify(D1p + eps * D2p)
D1 = collect(D1, cos(two_psi + two_zeta))
D1 = collect(D1, sin(two_psi + two_zeta))
print (D1)


print('Detector- (Ex = Ey)')
D2 = simplify(D2p + eps * D1p)
D2 = collect(D2, cos(two_psi + two_zeta))
D2 = collect(D2, sin(two_psi + two_zeta))
print (D2)

#O'Dea et al.
M = compute_beamsor_from_E( Ex_co, Ex_cx, Ey_co, Ey_cx )
ZMZ = Z * M * Z.transpose()
M = ZMZ
ML = M * L.transpose()
MLS = simplify(ML * S)

print('Detector+ (Full)')
D = Matrix( ([1 + eps, (1 - eps) * cos(two_zeta), (1 - eps) * sin(two_zeta),0]) )
D3 = simplify(D.dot(MLS))
D3 = collect(D3, cos(two_psi + two_zeta))
D3 = collect(D3, sin(two_psi + two_zeta))
print (D3)

print('Detector- (Full)')
D = Matrix( ([1 + eps, (eps - 1) * cos(two_zeta), (eps - 1) * sin(two_zeta),0]) )
D4 = simplify(D.dot(MLS))
D4 = collect(D4, cos(two_psi + two_zeta))
D4 = collect(D4, sin(two_psi + two_zeta))
print (D4)

print('Detector+ (Diff)')
print simplify(D3 - D1)
print('Detector- (Diff)')
print simplify(D4 - D2)








