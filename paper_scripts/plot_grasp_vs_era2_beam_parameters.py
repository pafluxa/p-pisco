# coding: utf-8
import pylab
from matplotlib import patches

import numpy
import pandas

# pandas are great
moonBP  = pandas.read_csv( 'data/array_data/qband_era2_moon_subtracted_beam_parameters.csv' )
graspBP = pandas.read_csv( 'data/array_data/qband_grasp_beam_parameters.csv' )

# Setup figure and axes
fig = pylab.figure()

# Axis for H detectors
axH = fig.add_subplot( 121, aspect='equal' )
# Axis for V detectors
axV = fig.add_subplot( 122, aspect='equal' )

# clean axis with cla()
axH.cla()
axH.set_title( 'det_pol = -45 degrees' )
axH.set_ylim( -10, 10 )
axH.set_xlim( -12, 12 )
axH.set_xlabel( 'degrees' )
axH.set_ylabel( 'degrees' )

axV.cla()
axV.set_title( 'det_pol = +45 degrees' )
axV.set_ylim( -10, 10 )
axV.set_xlim( -12, 12 )
axV.set_xlabel( 'degrees' )
axV.set_ylabel( 'degrees' )

for i in range( len( graspBP ) ):
   
    # Choose whether to plot in the H of V subplot
    # based on detector polarization angle
    ax = None    
    if moonBP['det_pol'].iloc[i] > 0:
        ax = axV
    else:
        ax = axH
    
    # Create ellipse. the "g" is for "GRASP".
    # Note I am using AzOff and ElOff from Moon data
    # instead of the pointing offsets extracted from GRASP sims.
    # This is because we want to compare only the beams.
    eg = patches.Ellipse(  
     (   moonBP['AzOff'].iloc[i], 
         moonBP['ElOff'].iloc[i] ),  
         graspBP['fwhm_x'].iloc[i], 
         graspBP['fwhm_y'].iloc[i], 
         # Docs say that ellipse is anti-clockwise. 
         graspBP[ 'theta'].iloc[i], 
         fill=False,
         edgecolor='black',
         linewidth=1,
         alpha=0.5 )

    # 'm' is for "Moon"
    em = patches.Ellipse(  
         (  moonBP['AzOff'].iloc[i], 
            moonBP['ElOff'].iloc[i] ),  
            moonBP['FWHM_x'].iloc[i], 
            moonBP['FWHM_y'].iloc[i],    
            # Docs say that ellipse is anti-clockwise. 
            moonBP ['Theta'].iloc[i], 
            fill=True,  
            color='red', 
            alpha=0.2 )

    ax.add_patch( eg )
    ax.add_patch( em )

pylab.show()
