#from matplotlib import rc_file
#rc_file('./matplotlibrc')  # <-- the file containing your settings
import sys
import pandas
import numpy

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here

def set_size( width, fraction=1.2, subplot=[1,1] ):
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
    fig_width_pt = width

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    _w = fig_width_pt * inches_per_pt
    fig_width_in = _w * fraction
    # Figure height in inches
    #fig_height_in = fig_width_in * golden_ratio
    fig_height_in = _w * golden_ratio * ( subplot[0]*1.0/subplot[1]) 

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

width = 700

nice_fonts = {
        # Use LaTex to write all text
        "text.usetex": False,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 11,
        "font.size": 11,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
}

mpl.rcParams.update(nice_fonts)
# configure fond to Seri# make x/y-ticks large

# make 3 axes in a row
lmax = 250

data = pandas.read_csv( sys.argv[1] )
clT,clE,clB,_, = numpy.loadtxt( sys.argv[2], unpack=True )
pi = numpy.pi
clT = clT[0:lmax+1] * 1e12
clE = clE[0:lmax+1] * 1e12
clB = clB[0:lmax+1] * 1e12

# x 1e12 to put in uK
TT  = data['TT_out'] 
EE  = data['EE_out'] 
BB  = data['BB_out'] 

TTo = data['TT_in'] 
EEo = data['EE_in'] 
BBo = data['BB_in'] 

wl_TT = data['wl_TT']
wl_EE = data['wl_EE']
wl_BB = data['wl_BB']

ell  = numpy.arange( 1, TT.size + 1 )
ell2 = (ell)*(ell+1)/(pi)

# I forgot to add the correct window function, but this is the equivalent gaussian beam anyways
import healpy
wT,wE,wB,_ = healpy.gauss_beam( numpy.deg2rad(1.4701), pol=True, lmax=lmax ).T


#width = 345 * 2
fig, axes = plt.subplots( 1, 3, figsize=set_size( width,subplot=[1,2] ), sharex=True )
plt.subplots_adjust( wspace=.28, bottom=0.15 )

axTT = axes[0]
axEE = axes[1]
axBB = axes[2]

axTT.set_xlabel( r'$\ell$' )
axEE.set_xlabel( r'$\ell$' )
axBB.set_xlabel( r'$\ell$' )

axTT.set_title( r'$ D_{\ell}^{TT}$' )
axTT.set_xlabel( r'$\ell$' )
axTT.set_ylabel( r'$\mu \rm{K}^2$' )
axTT.set_ylim( (100,6000) )
axTT.set_xlim( (2,lmax) )
axTT.ticklabel_format(axis='y', style='sci')
axTT.yaxis.major.formatter.set_powerlimits((0,1))

axEE.set_title( r'$ D_{\ell}^{EE}$' )
axEE.set_xlim( (2,lmax) )
axEE.set_ylim( (0, 2.0) )
#axEE.set_yscale('log', linthreshy=1e-1)
axEE.ticklabel_format(axis='y', style='sci')
axEE.yaxis.major.formatter.set_powerlimits((0,1))

axBB.set_title( r'$ D_{\ell}^{BB}$' )
axBB.set_xlim( (2,lmax) )
axBB.set_ylim( (-1e-3,1e0) )
axBB.set_yscale('symlog', linthreshy=1e-4)
#axBB.ticklabel_format(axis='y', style='sci')
#axBB.yaxis.major.formatter.set_powerlimits((0,1))

axTT.plot( ell2*clT,      alpha=0.8, label='ref', linestyle='dashed', color='black')
axTT.plot( ell2*TTo,     alpha=0.7, label= 'in', linestyle='dotted', color='green')
axTT.plot( ell2*TT/(wT)**2, alpha=0.5, label='out', linestyle= 'solid', color= 'red' )
axTT.legend()

axEE.plot( ell2*clE,      alpha=0.8, label='ref', linestyle='dashed', color='black')
axEE.plot( ell2*EEo ,     alpha=0.7, label= 'in', linestyle='dotted', color='green')
axEE.plot( ell2*EE/(wE)**2, alpha=0.5, label='out', linestyle= 'solid', color= 'red' )
axEE.legend()

axBB.plot( ell2*clB,      alpha=0.8, label='ref', linestyle='dashed', color='black')
axBB.plot( ell2*BBo ,     alpha=0.7, label= 'in', linestyle='dotted', color='green')
axBB.plot( ell2*BB/(wB)**2, alpha=0.5, label='out', linestyle= 'solid', color= 'red' )
axBB.legend()

# configure them like this: 
# first axis (TT) has ticks on the left
ax = axes[0]
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.major.formatter._useMathText = True
ax.yaxis.major.formatter._useMathText = True
ax.yaxis.set_minor_locator(  AutoMinorLocator(5) )
ax.xaxis.set_minor_locator(  AutoMinorLocator(5) )

# second axis (EE) has ticks on the right
ax = axes[1]
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.major.formatter._useMathText = True
ax.yaxis.major.formatter._useMathText = True
ax.xaxis.set_minor_locator(  AutoMinorLocator(2) )
ax.yaxis.set_minor_locator(  AutoMinorLocator(5) )

ax = axes[2]
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.major.formatter._useMathText = True
ax.yaxis.major.formatter._useMathText = True
ax.yaxis.set_minor_locator(  AutoMinorLocator(5) )
ax.xaxis.set_minor_locator(  AutoMinorLocator(5) )

#fig.tight_layout()

plt.savefig( "ps.pdf" )
plt.show()
    
	
