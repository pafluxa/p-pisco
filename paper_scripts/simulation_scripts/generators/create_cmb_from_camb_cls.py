import sys

import healpy
import pylab
import numpy

r_value   = (float)( sys.argv[1] )
out_nside = (int)( sys.argv[2] )

# Read cl's from input file
ls_in,dl_TT_in,dl_EE_in,dl_BB_in,dl_TE_in = numpy.loadtxt( sys.argv[3] )

'''
cl_TT_in = dl_TT_in/(ls_in*(ls_in+1)/2*numpy.pi)
cl_EE_in = dl_EE_in/(ls_in*(ls_in+1)/2*numpy.pi)
cl_BB_in = dl_BB_in/(ls_in*(ls_in+1)/2*numpy.pi)
cl_TE_in = dl_TE_in/(ls_in*(ls_in+1)/2*numpy.pi)
'''
cl_TT_in = dl_TT_in/(ls_in*(ls_in+1))
cl_EE_in = dl_EE_in/(ls_in*(ls_in+1))
cl_BB_in = dl_BB_in/(ls_in*(ls_in+1))
cl_TE_in = dl_TE_in/(ls_in*(ls_in+1))

# Create maps using synfast
I,Q,U = healpy.synfast( (cl_TT_in,cl_EE_in,cl_BB_in,cl_TE_in), out_nside, pol=True , new=True )

# Poor V, always zero
V = numpy.zeros_like( I )

'''
# Make some noise
fig_maps = pylab.figure( 0 )
healpy.mollview( I , sub=(1,3,1) , fig=fig_maps)
healpy.mollview( Q , sub=(1,3,2) , fig=fig_maps)
healpy.mollview( U , sub=(1,3,3) , fig=fig_maps)
'''

# Check output CL's are consistent with input
cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB = healpy.anafast( (I, Q, U), pol=True, alm=False )
ls = numpy.arange( cl_TT.size )

lmax = out_nside * 4 - 1

fig_cls = pylab.figure()
pylab.subplot( 131 )
pylab.plot( dl_TT_in )
pylab.plot( ls*(ls+1)*cl_TT, color='red' )
pylab.xlim( 0,lmax )

pylab.subplot( 132 )
pylab.plot( dl_EE_in )
pylab.plot( ls*(ls+1)*cl_EE, color='red' )
pylab.xlim( 0,lmax )

pylab.subplot( 133 )
pylab.plot( dl_BB_in )
pylab.plot( ls*(ls+1)*cl_BB, color='red' )
pylab.xlim(0,lmax)

pylab.show()
#numpy.savez( './lcdm_r=%1.3f_0000_nside=%d.npz' % (r_value,out_nside), I=I, Q=Q, U=U, V=V, nside=out_nside )


