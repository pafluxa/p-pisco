import sys

import healpy
import pylab
import numpy

from numpy import pi

r_value   = (float)( sys.argv[1] )
out_nside = (int)( sys.argv[2] )

# Read cl's from input file
ls_in,cl_TT_in,cl_EE_in,cl_BB_in,cl_TE_in = numpy.loadtxt( sys.argv[3] )

# Create maps using synfast
I,Q,U = healpy.synfast( (
    (2*pi)**2 * cl_TT_in/(ls_in*(ls_in+1)),
    (2*pi)**2 * cl_EE_in/(ls_in*(ls_in+1)),
    (2*pi)**2 * cl_BB_in/(ls_in*(ls_in+1)),
    (2*pi)**2 * cl_TE_in/(ls_in*(ls_in+1)) ), out_nside, pol=True , new=True )
# Poor V, always zero
V = numpy.zeros_like( I )

# Make some noise
fig_maps = pylab.figure( 0 )
healpy.mollview( I*1e6, sub=(1,1,1) , fig=fig_maps, unit='uK', min=-300, max=300, cmap='jet' )
#healpy.mollview( Q*1e6, sub=(1,3,2) , fig=fig_maps)
#healpy.mollview( U*1e6, sub=(1,3,3) , fig=fig_maps)

# Check output CL's are consistent with input
cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB = healpy.anafast( (I, Q, U), pol=True, alm=False )
ls = numpy.arange( cl_TT.size )

lmax = out_nside * 4 - 1

fig_cls = pylab.figure()
pylab.subplot( 131 )
pylab.plot( cl_TT_in )
pylab.plot( ls*(ls+1)*cl_TT )
pylab.xlim( 0,lmax )

pylab.subplot( 132 )
pylab.plot( cl_EE_in )
pylab.plot( ls*(ls+1)*cl_EE )
pylab.xlim( 0,lmax )

pylab.subplot( 133 )
pylab.plot( cl_BB_in )
pylab.plot( ls*(ls+1)*cl_BB )
pylab.xlim(0,lmax)

pylab.show()

numpy.savez( './lcdm_r=%1.3f_0000_nside=%d.npz' % (r_value,out_nside), I=I, Q=Q, U=U, V=V, nside=out_nside )
