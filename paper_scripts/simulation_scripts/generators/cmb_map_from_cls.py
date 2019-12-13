import sys

import healpy
import pylab
import numpy

from numpy import pi

# read in positional parameters
out_nside = (int)  ( sys.argv[1] )
r_value   = (float)( sys.argv[2] )

# Read cl's from input file
cl_TT,cl_EE,cl_BB,cl_TE, = numpy.loadtxt( sys.argv[3], unpack=True )

I,Q,U = healpy.synfast( (cl_TT, cl_EE, cl_BB, cl_TE), out_nside, pol=True , new=True )
# Set V tos zero
V = numpy.zeros_like( I )

# Check output CL's are consistent with input
lmax = 250
cl_TT2, cl_EE2, cl_BB2, _, _, _ = healpy.anafast( (I,Q,U), pol=True, alm=False, iter=3, lmax=lmax )

cl_TT  *= 1e12
cl_EE  *= 1e12
cl_BB  *= 1e12
cl_TT = cl_TT[1:lmax+1]
cl_EE = cl_EE[1:lmax+1]
cl_BB = cl_BB[1:lmax+1]


cl_TT2 *= 1e12
cl_EE2 *= 1e12
cl_BB2 *= 1e12
cl_TT2 = cl_TT2[1::]
cl_EE2 = cl_EE2[1::]
cl_BB2 = cl_BB2[1::]

ls_in = numpy.arange( cl_TT.size + 1 )
ls_in = ls_in[0:lmax]
ls    = ls_in*(ls_in+1)/(2*(numpy.pi))

fig_cls = pylab.figure()
pylab.subplot( 131 )
pylab.plot( ls*cl_TT , '--', color='red', alpha=0.5  )
pylab.plot( ls*cl_TT2, color='black', alpha=0.5 )
pylab.xlim( 0,250 )
#pylab.ylim( 100,6000 )
pylab.ylabel( 'uK^2' )

pylab.subplot( 132 )
pylab.plot( ls*cl_EE , '--', color='red', alpha=0.5)
pylab.plot( ls*cl_EE2, color='black', alpha=0.5 )
pylab.xlim( 0,250 )
pylab.ylim( 0, 2 )
pylab.ylabel( 'uK^2' )

pylab.subplot( 133 )
pylab.plot( ls*cl_BB , '--', color='red', alpha=0.5)
pylab.plot( ls*cl_BB2, color='black', alpha=0.5 )
pylab.ylim( -1e-3, 5.0 )
pylab.xlim(0,250)
pylab.yscale( 'symlog', linthreshy=1e-4 )
pylab.ylabel( 'uK^2' )

pylab.show()

# save maps to disk
numpy.savez( './lcdm_r=%1.3f_0000_nside=%d.npz' % (r_value,out_nside), I=I, Q=Q, U=U, V=V, nside=out_nside )
