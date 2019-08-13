#from matplotlib import rc_file
#rc_file('./matplotlibrc')  # <-- the file containing your settings

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy

import healpy

import sys

def set_size( width, fraction=1, subplot=[1,1] ):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2.0

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    #fig_height_in = fig_width_in * golden_ratio
    fig_height_in = fig_width_in * golden_ratio * (subplot[0]*1.0/subplot[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

# Using seaborn's style
#plt.style.use('seaborn')
width = 750

nice_fonts = {
        # Use LaTex to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
}

mpl.rcParams.update(nice_fonts)

# Load maps
data = numpy.load( sys.argv[2] )
#data = numpy.load( './data/stokes_I_maps.npz' )

nside = data['nside'][()]
ra_s,dec_s = data['coords']
I = data['I']
Q = data['Q']
U = data['U']
W = data['W']

# Create maps 
I_s,Q_s,U_s = numpy.zeros( (3, healpy.nside2npix( nside )) )
pix = healpy.ang2pix( nside, numpy.radians(90-dec_s), numpy.radians(ra_s) )

I_s[ pix ] = 1.0
val = str( sys.argv[1] )

if val == 'Q':
    Q_s[pix] = 1.0
elif val == 'U':
    U_s[pix] = 1.0
else:
    raise RuntimeError( "" )

I_s,Q_s,U_s = healpy.smoothing( (I_s,Q_s,U_s), fwhm=numpy.radians(1.5), pol=True )

resI = (I - I_s)/I_s.max()
resQ = (Q - Q_s)/Q_s.max()
resU = (U - U_s)/U_s.max()

print( numpy.min( resI ), numpy.max( resI ) )
print( numpy.min( resQ ), numpy.max( resQ ) )
print( numpy.min( resU ), numpy.max( resU ) )

Ip  = healpy.gnomview( I  , sub=(2,3,1), rot=(ra_s, dec_s), return_projected_map=True , title='I', notext=True )
Qp  = healpy.gnomview( Q  , sub=(2,3,2), rot=(ra_s, dec_s), return_projected_map=True , title='Q', notext=True )
Up  = healpy.gnomview( U  , sub=(2,3,3), rot=(ra_s, dec_s), return_projected_map=True , title='U', notext=True )
Io  = healpy.gnomview( I_s, sub=(2,3,4), rot=(ra_s, dec_s), return_projected_map=True , title='I', notext=True )
Qo  = healpy.gnomview( Q_s, sub=(2,3,5), rot=(ra_s, dec_s), return_projected_map=True , title='Q', notext=True )
Uo  = healpy.gnomview( U_s, sub=(2,3,6), rot=(ra_s, dec_s), return_projected_map=True , title='U', notext=True )
'''
fig = plt.figure()
plt.savefig( '.theplot.pdf' )
del fig
'''
import pylab

fig, axes = plt.subplots( 1, 3, figsize=set_size( width,subplot=[1,3] ) )

im = \
axes[0].imshow( (Ip - Io)/Io.max(), extent=(-5,5,-5,5), vmin=-1e-4, vmax=1e-3 )
axes[0].set_title( 'Residual (I)' )
axes[0].set_xlabel( r"$^{\circ}$" )
axes[0].set_ylabel( r"$^{\circ}$" )
pylab.colorbar(im ) 

im = \
axes[1].imshow( (Qp - Qo)/Io.max(), extent=(-5,5,-5,5), vmin=-1e-4, vmax=1e-3   )
axes[1].set_title( 'Residual (Q)' )
axes[1].set_xlabel( r"$^{\circ}$" )
axes[1].set_ylabel( r"$^{\circ}$" )
pylab.colorbar(im) 

im = \
axes[2].imshow( (Up - Uo)/Io.max(), extent=(-5,5,-5,5), vmin=-1e-4, vmax=1e-3   )
axes[2].set_title( 'Residual (U)' )
axes[2].set_xlabel( r"$^{\circ}$" )
axes[2].set_ylabel( r"$^{\circ}$" )
pylab.colorbar(im) 

#cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat[0:3]])                                                
#pylab.colorbar(im, cax=cax, **kw) 
plt.savefig( sys.argv[3] )
pylab.show()

