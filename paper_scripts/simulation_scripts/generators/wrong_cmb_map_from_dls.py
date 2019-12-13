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
ls_in = numpy.arange( cl_TT.size + 1 )
ls_in = ls_in[1:]

#dl2cl = (2*pi)**2 /( ls_in*(ls_in+1) ) 
#print( dl2cl.shape, dl_TT.shape )

# Create maps using synfast
'''
I,Q,U = healpy.synfast(
  ( dl2cl*dl_TT,
    dl2cl*dl_EE,
    dl2cl*dl_BB,
    dl2cl*dl_TE, ), out_nside, pol=True , new=True )
'''
I,Q,U = healpy.synfast( (cl_TT, cl_EE, cl_BB, cl_TE), out_nside, pol=True , new=True )
# Set V tos zero
V = numpy.zeros_like( I )

# Plot temperature CMB in uK
#fig_maps = pylab.figure( 0 )
#healpy.mollview( I*1e6, sub=(1,1,1) , fig=fig_maps, unit='uK', min=-350, max=350, cmap='jet' )

# Check output CL's are consistent with input
#cl_TT2, cl_EE2, cl_BB2, _, _, _cl_TE, cl_EB, cl_TB = healpy.anafast( (I, Q, U), pol=True, alm=False, iter=3 )
cl_TT2, cl_EE2, cl_BB2, _, _, _ = healpy.anafast( (I,Q,U), pol=True, alm=False, iter=3 )

cl_TT  *= 1e12
cl_EE  *= 1e12
cl_BB  *= 1e12

cl_TT2 *= 1e12
cl_EE2 *= 1e12
cl_BB2 *= 1e12

ls_in = numpy.arange( cl_TT.size + 1 )
ls_in = ls_in[1:]

fig_cls = pylab.figure()
pylab.subplot( 131 )
pylab.plot( cl_TT , '--', color='red', alpha=0.5  )
pylab.plot( cl_TT2, color='black', alpha=0.5 )
pylab.xlim( 0,250 )
pylab.ylim( 100,6000 )
pylab.ylabel( 'uK^2' )

pylab.subplot( 132 )
pylab.plot( cl_EE , '--', color='red', alpha=0.5)
pylab.plot( cl_EE2, color='black', alpha=0.5 )
pylab.xlim( 0,250 )
pylab.ylim( 0, 2 )
pylab.ylabel( 'uK^2' )

pylab.subplot( 133 )
pylab.plot( dl_BB , '--', color='red', alpha=0.5)
pylab.plot( cl_BB2, color='black', alpha=0.5 )
pylab.ylim( -1e-3, 5.0 )
pylab.xlim(0,250)
pylab.yscale( 'symlog', linthreshy=1e-4 )
pylab.ylabel( 'uK^2' )

pylab.show()

# save maps to disk
#numpy.savez( './lcdm_r=%1.3f_0000_nside=%d.npz' % (r_value,out_nside), I=I, Q=Q, U=U, V=V, nside=out_nside )
