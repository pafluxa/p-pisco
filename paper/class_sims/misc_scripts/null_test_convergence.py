# coding: utf-8
import sys

import pylab
import numpy
import healpy
import pisco
from pisco.mapping.core import matrices_to_maps

# Load input maps
imaps    = numpy.load( sys.argv[1] )
I_o      = imaps['I']
Q_o      = imaps['Q']*0
U_o      = imaps['U']*0
nside    = imaps['nside'][()]  

# Set lmax to be 400
lmax = (4*nside - 1)/2

# Smooth by given FWHM
fwhm     = (float)( sys.argv[2] )
I_o_s,Q_o_s,U_o_s = healpy.smoothing( (I_o,Q_o,U_o), fwhm=numpy.radians(fwhm), pol=True )

# Get power spectra of smoothed maps
TTo,EEo,BBo,_,_,_ = healpy.anafast( (I_o,Q_o,U_o), pol=True, lmax=lmax )

# Get map Pixel Window function
pixwin_temp, pixwin_pol = healpy.pixwin( nside, pol=True )
pixwin_temp = pixwin_temp[0:lmax+1]
pixwin_pol  = pixwin_pol [0:lmax+1]

# Get Gaussian window function
glTT, glEE, glBB, glTE = healpy.sphtfunc.gauss_beam( numpy.radians(fwhm), pol=True, lmax=lmax ).T
wl_TT = (glTT**2 * pixwin_temp )
wl_EE = (glEE**2 * pixwin_pol  )
wl_BB = (glBB**2 * pixwin_pol  )

# Create ell
ell = numpy.arange( TTo.size )
ell2 = ell*(ell+1)/(2*numpy.pi)

'''
data_128  = numpy.load( './runs/matrices_null_test_nscans_4_beam_nside_1x.npz' )
data_256  = numpy.load( './runs/matrices_null_test_nscans_4_beam_nside_2x.npz' )
data_512  = numpy.load( './runs/matrices_null_test_nscans_4_beam_nside_4x.npz' )
data_1024 = numpy.load( './runs/matrices_null_test_nscans_4_beam_nside_8x.npz' )
'''
data_128  = numpy.load( './runs/matrices_null_test_only_I_beam_nside_1x.npz' )
data_256  = numpy.load( './runs/matrices_null_test_only_I_beam_nside_2x.npz' )
data_512  = numpy.load( './runs/matrices_null_test_only_I_beam_nside_4x.npz' )
data_1024 = numpy.load( './runs/matrices_null_test_only_I_beam_nside_8x.npz' )

datas = [data_128,data_256,data_512,data_1024]

fig, axes = pylab.subplots( 1,3 )

ax_TT = axes[0]
ax_TT.set_title( r'$C_\ell^{\mathrm{TT}}$ and $\Delta C_\ell^{\mathrm{TT}}$' )
ax_TT.set_xlabel( r'$\ell$' )
ax_TT.set_ylabel( r'K$^2$' )
ax_TT.set_ylim( (1e-40,1e-9) )
ax_TT.set_xlim( (2,lmax) )
ax_TT.set_yscale( 'log' )
ax_TT.set_xscale( 'log' )

ax_EE = axes[1]
ax_EE.set_title( r'$C_\ell^{\mathrm{EE}}$ and $\Delta C_\ell^{\mathrm{EE}}$' )
ax_EE.set_xlabel( r'$\ell$' )
ax_EE.set_ylabel( r'K$^2$' )
ax_EE.set_ylim( (1e-40,1e-9) )
ax_EE.set_xlim( (2,lmax) )
ax_EE.set_yscale( 'log' )
ax_EE.set_xscale( 'log' )

ax_BB = axes[2]
ax_BB.set_title( r'$C_\ell^{\mathrm{BB}}$ and $\Delta C_\ell^{\mathrm{BB}}$' )
ax_BB.set_xlabel( r'$\ell$' )
ax_BB.set_ylabel( r'K$^2$' )
ax_BB.set_ylim( (1e-40,1e-9) )
ax_BB.set_xlim( (2,lmax) )
ax_BB.set_yscale( 'log' )
ax_BB.set_xscale( 'log' )

ax_TT.plot( ell, ell2*TTo ) 
ax_EE.plot( ell, ell2*EEo ) 
ax_BB.plot( ell, ell2*BBo ) 

for data in datas:
    
    I,Q,U,W = matrices_to_maps( data['nside'][()], data['AtA'], data['AtD'] )
    
    Ires = I - I_o_s
    Qres = Q - Q_o_s
    Ures = U - U_o_s

    TT_res, EE_res, BB_res, _, _, _ = healpy.anafast( (Ires,Qres,Ures), pol=True, lmax=lmax )

    ax_TT.plot( ell, ell2*TT_res/wl_TT, marker='.', linestyle='dashed', alpha=0.5 )
    ax_EE.plot( ell, ell2*EE_res/wl_EE, marker='.', linestyle='dashed', alpha=0.5 )
    ax_BB.plot( ell, ell2*BB_res/wl_BB, marker='.', linestyle='dashed', alpha=0.5 )

pylab.show()
