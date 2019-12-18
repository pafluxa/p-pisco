import sys

import healpy
import pylab
import numpy

from numpy import pi

# read in positional parameters
out_nside = (int)  ( sys.argv[1] )
r_value   = (float)( sys.argv[2] )

# Read cl's from input file
dl_TT,dl_EE,dl_BB,dl_TE = numpy.loadtxt( sys.argv[3], unpack=True )
ls_in = numpy.arange( dl_TT.size + 1 )
ls_in = ls_in[1:]

dl2cl = (2*pi)**2 /( ls_in*(ls_in+1) ) 
print( dl2cl.shape, dl_TT.shape )

# Create maps using synfast
I,Q,U = healpy.synfast( 
  ( dl2cl*dl_TT,
    dl2cl*dl_EE,
    dl2cl*dl_BB*0,
    dl2cl*dl_TE, ), out_nside, pol=True , new=True )

# Set V tos zero
V = numpy.zeros_like( I )

# Plot temperature CMB in uK
#fig_maps = pylab.figure( 0 )
#healpy.mollview( I*1e6, sub=(1,1,1) , fig=fig_maps, unit='uK', min=-350, max=350, cmap='jet' )

# set lmax to maximum for nside=128
out_nside = 512
lmax = out_nside * 4 - 1

# Check output CL's are consistent with input
cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB = healpy.anafast( (I, Q, U), pol=True, alm=False )

ls_in = numpy.arange( cl_TT.size + 1 )
ls_in = ls_in[1:]
dl2cl = (2*pi)**2 /( ls_in*(ls_in+1) ) 

fig_cls = pylab.figure()
pylab.subplot( 131 )
pylab.plot( dl_TT, '--', color='red', alpha=0.5  )
pylab.plot( cl_TT/dl2cl, color='black' )
pylab.xlim( 0,lmax )
pylab.ylabel( 'K^2' )

pylab.subplot( 132 )
pylab.plot( dl_EE , '--', color='red', alpha=0.5)
pylab.plot( cl_EE/dl2cl, color='black' )
pylab.xlim( 0,lmax )
pylab.ylabel( 'K^2' )

pylab.subplot( 133 )
pylab.plot( dl_BB , '--', color='red', alpha=0.5)
pylab.plot( cl_BB/dl2cl, color='black' )
pylab.xlim(0,lmax)
pylab.ylabel( 'K^2' )

pylab.show()

# save maps to disk
numpy.savez( './lcdm_r=%1.3f_0000_nside=%d.npz' % (r_value,out_nside), I=I, Q=Q, U=U, V=V, nside=out_nside )
