import numpy
import numpy as np
import pylab as plt
import pandas

from mpl_toolkits.axes_grid1 import make_axes_locatable

from two_D_Gaussian_fit_util import  gaussian_elliptical_beam, to_covariance, from_covariance 

# Create x and y indices
x = np.arange(-4., 4., 0.05) + 0.025
#print x[0], x[-1]
x, y = np.meshgrid(x, x)

arrdata = pandas.read_csv( './data/array_data/qband_era2_moon_subtracted_beam_parameters.csv' )

# loop over the feeds
feeds = arrdata['Feed']
for feed in numpy.unique( feeds.values ):
    
    # find matching pairs
    pair = arrdata['Detector'][ arrdata['Feed'] == feed ].values
    v    = int( arrdata[ arrdata['Detector'] == pair[0] ].index.values )
    h    = int( arrdata[ arrdata['Detector'] == pair[1] ].index.values )
    
    print( v, h )

    x0 = ( arrdata[ 'AzOff'].iloc[v] + arrdata[ 'AzOff'].iloc[h] )/2.0
    y0 = ( arrdata[ 'ElOff'].iloc[h] + arrdata[ 'ElOff'].iloc[h] )/2.0
    
    fwx0 = ( arrdata['FWHM_x'].iloc[v] + arrdata['FWHM_x'].iloc[h] )/2.0
    fwy0 = ( arrdata['FWHM_y'].iloc[h] + arrdata['FWHM_y'].iloc[h] )/2.0
    tht0 = ( arrdata[ 'Theta'].iloc[h] + arrdata[ 'Theta'].iloc[h] )/2.0
    
    print( x0, y0, fwx0, fwy0, tht0  )

    pointing_mismatch = []
    beam_mismatch = []
    all_mismatch = []

    for p in [v,h]:
        
        fwx = arrdata['FWHM_x'].iloc[p]
        fwy = arrdata['FWHM_y'].iloc[p]
        tht = arrdata[ 'Theta'].iloc[p]

        xoff= arrdata[ 'AzOff'].iloc[p] - x0
        yoff= arrdata[ 'ElOff'].iloc[p] - y0
        
        #create data
        beam_mismatch.append( gaussian_elliptical_beam((x, y), 3, fwx, fwy, 0, 0, tht ) )
        pointing_mismatch.append( gaussian_elliptical_beam((x, y), 3, fwx0, fwy0, xoff, yoff, tht0 ) )
        all_mismatch.append( gaussian_elliptical_beam((x, y), 3, fwx, fwy, xoff, yoff, tht ) )
    
    pm = pointing_mismatch[1] - pointing_mismatch[0]
    bm = beam_mismatch[1] - beam_mismatch[0]
    cm = all_mismatch[1] - all_mismatch[0]

    # plot twoD_Gaussian data generated above
    fig, axes = plt.subplots( 1, 3 )
    i1 = axes[0].imshow( pm.reshape(160, 160),origin='lower', extent=(x.min(), x.max(), y.min(), y.max()),cmap='seismic')
    divider = make_axes_locatable( axes[0] )                                                                      
    cax1 = divider.append_axes('right', size='5%', pad=0.05)                                                      
    cbar1 = fig.colorbar( i1, cax=cax1, orientation='vertical' )  
    
    i2 = axes[1].imshow( bm.reshape(160, 160),origin='lower', extent=(x.min(), x.max(), y.min(), y.max()),cmap='seismic')
    divider = make_axes_locatable( axes[1] )                                                                      
    cax2 = divider.append_axes('right', size='5%', pad=0.05)                                                      
    cbar2 = fig.colorbar( i2, cax=cax2, orientation='vertical' )  
    
    i3 = axes[2].imshow( cm.reshape(160, 160),origin='lower', extent=(x.min(), x.max(), y.min(), y.max()),cmap='seismic')
    divider = make_axes_locatable( axes[2] )                                                                      
    cax3 = divider.append_axes('right', size='5%', pad=0.05)                                                      
    cbar3 = fig.colorbar( i3, cax=cax3, orientation='vertical' )  
    
    #plt.imshow(data.reshape(160, 160),origin='lower',cmap='seismic')
    # plt.colorbar( i1 )
    #plt.suptitle('Elliptical Gaussian Beam Pair Difference \n Delta x_off -0.05, Delta y_off 0.05')
    plt.show()
