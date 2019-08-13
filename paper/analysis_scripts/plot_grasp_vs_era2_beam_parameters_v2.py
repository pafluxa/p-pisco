# coding: utf-8
import pylab
from matplotlib import patches

import numpy
import numpy as np
#import pandas

# pandas are great
#moonBP  = pandas.read_csv( 'data/array_data/qband_era2_moon_subtracted_beam_parameters.csv' )
#graspBP = pandas.read_csv( 'data/array_data/qband_grasp_beam_parameters.csv' )
moonBP = numpy.genfromtxt('./data/array_data/qband_era2_moon_subtracted_beam_parameters.csv', delimiter=',', names=True, dtype=None, encoding='ascii')
graspBP = numpy.genfromtxt('./data/array_data/grasp_beam_parameters_with_feeds.csv', delimiter=',', names=True, dtype=None, encoding='ascii')
#moonBP = numpy.genfromtxt('qband_era2_moon_subtracted_beam_parameters.csv', delimiter=',', names=True, dtype=None, encoding='ascii')
#graspBP = numpy.genfromtxt('grasp_beam_parameters_with_feeds.csv', delimiter=',', names=True, dtype=None, encoding='ascii')

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

# arrays to draw ellipses
theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))

for i, det in enumerate(moonBP['Detector']):
   
    # Choose whether to plot in the H of V subplot
    # based on detector polarization angle
    ax = None    
    if moonBP['det_pol'][i] > 0:
        ax = axV
    else:
        ax = axH
    
    # Create ellipse. the "g" is for "GRASP".
    # Note I am using AzOff and ElOff from Moon data
    # instead of the pointing offsets extracted from GRASP sims.
    # This is because we want to compare only the beams.
    idx = numpy.where(graspBP['det'] == det)

    if len(idx[0]):
        
        idx = idx[0][0]
        
        ag,bg = graspBP['fwhm_x'][idx], graspBP['fwhm_y'][idx]
        rotg  = graspBP[ 'theta'][idx]
    
        x0,y0 = moonBP['AzOff'][i], moonBP['ElOff'][i]
        am,bm = moonBP['FWHM_x'][i], moonBP['FWHM_y'][i]
        rotm  = moonBP ['Theta'][i]
    
        xg     = 0.5 * ag * np.cos(theta)
        yg     = 0.5 * bg * np.sin(theta)
        alphag = numpy.deg2rad( rotg )
    
        Rg = np.array([
            [np.cos(alphag), -np.sin(alphag)],
            [np.sin(alphag),  np.cos(alphag)],])
    
        xg, yg = np.dot(Rg, np.array([xg, yg]))
            [ np.cos(alphag), -np.sin(alphag)],
            [ np.sin(alphag),  np.cos(alphag)],])
    
        xg, yg = np.dot(Rg, np.array([xg, yg]))
        rg = np.sqrt( xg**2 + yg**2 )
        thetag = theta + alphag
        rg = np.interp(theta, thetag, rg, period=2.0*np.pi) 
    
        xm     = 0.5 * am * np.cos(theta)
        ym     = 0.5 * bm * np.sin(theta)
        alpham = numpy.deg2rad( rotm )

        Rm = np.array([
            [np.cos(alpham), -np.sin(alpham)],
            [np.sin(alpham),  np.cos(alpham)],])
    
        xm, ym = np.dot(Rm, np.array([xm, ym]))
    
        R = np.sqrt( xm**2 + ym**2 ) - np.sqrt( xg**2 + yg**2 )
        R = np.abs( R ) * 6
            [ np.cos(alpham), -np.sin(alpham)],
            [ np.sin(alpham),  np.cos(alpham)],])
    
        xm, ym = np.dot(Rm, np.array([xm, ym]))
        rm = np.sqrt( xm**2 + ym**2 )
        thetam = theta + alpham
        rm = np.interp(theta, thetam, rm, period=2.0*np.pi) 
    
        R = np.abs( rm - rg ) * 6

        ax.fill( x0 + R*np.cos(theta), y0 + R*np.sin(theta), facecolor='red', alpha=0.6, linewidth=1 )

pylab.show()
