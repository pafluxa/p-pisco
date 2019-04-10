# coding: utf-8
import numpy
from numpy import sin,cos,sqrt

import pisco
from pisco.beam_analysis import utils
from pisco.beam_analysis.mueller import ComplexMuellerMatrix as CM

numpy.set_printoptions(precision=2)

E_co = numpy.array( [1.0] )
E_cx = numpy.array( [0.0] )

P = CM.make_polarimeter_mueller_matrix( 1.0, 0.0 )

psi = numpy.radians(-45)
M_rot = numpy.eye(4, dtype='complex64')
M_rot[1][1] = numpy.exp( -1.j*2*psi )
M_rot[2][2] = numpy.exp(  1.j*2*psi )

s_in = numpy.asarray( (1.0, 0.0,1.0,0.0) )
p_in = numpy.asarray( (s_in[0], s_in[1]+1.j*s_in[2], s_in[1]-1.j*s_in[2], s_in[3] ) )

p_temp  = numpy.dot( M_rot , p_in   )
p_out   = numpy.dot( P.M[0], p_temp )

I =  numpy.real( p_out[0] )

print 'Power out from Mueller formalism:', I
print 'Power out from Jones formula    :', 0.5*(s_in[0] + s_in[1]*cos(2*psi) + s_in[2]*sin(2*psi) + s_in[3])

