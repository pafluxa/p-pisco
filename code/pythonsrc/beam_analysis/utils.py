import numpy
import numpy as np
from numpy import sin,cos

import healpy
import pylab

def azel_grid_to_healpix_beam( E, nx, ny, grid_size, output_map_nside ):
    '''
    '''

    # Convert to radians
    xs = -grid_size/2.0
    xe =  grid_size/2.0
    ys = -grid_size/2.0
    ye =  grid_size/2.0

    # Create arrays with angular az/el angular coordinates
    dx = numpy.linspace( xs, xe, nx )
    dy = numpy.linspace( ys, ye, ny )

    dx,dy = numpy.meshgrid( dx,dy )
    dx    = dx.ravel()
    dy    = dy.ravel()

    # Create arrays with actual spherical coordinates from az/el coordinates
    cdx = numpy.cos( dx )
    cdy = numpy.cos( dy )
    sdx = numpy.sin( dx )
    sdy = numpy.sin( dy )

    cr  = cdx * cdy
    r   = numpy.arccos( cr )
    alpha = numpy.arctan2( sdy, sdx*cdy )

    theta = r
    phi   = alpha

    # Convert spherical angular coordinates to pixels
    pixels = healpy.ang2pix( output_map_nside, theta, phi )

    # Create healpix map
    beam_map = numpy.zeros( healpy.nside2npix( output_map_nside ) , dtype='complex128' )

    # Project the fields/power to the map
    beam_map[ pixels ] = E

    # Apodize at 95% to avoid sharp edges
    beam_map[ pixels ] = beam_map[ pixels ]

    return beam_map


def __make_gaussian_elliptical_beam( nside, fwhm_x, fwhm_y, xo=0.0, yo=0.0, theta=0, size=0.1 ):
    '''
    Returns a Gaussian Beam as a square grid.
    '''
    # Create x and y indices
    x = np.linspace( -size, size, 512)
    y = np.linspace( -size, size, 512)
    x,y = np.meshgrid(x, y )

    # Convert FWHM in degres to sigma in radians
    sigma_x = numpy.radians( fwhm_x )/2.355
    sigma_y = numpy.radians( fwhm_y )/2.355
    
    theta = np.radians(theta)
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    gaussian = np.sqrt( np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2))) ).ravel()

    beam = azel_grid_to_healpix_beam( gaussian, 512, 512, size, nside )
    
    return beam

def make_gaussian_elliptical_beam( nside, fwhm_x, fwhm_y, phi_0, max_extent=10.0 ):
    '''
    Returns a Gaussian Beam as a HEALPix grid.

    input
        
        nside   : NSIDE resolution paratemer of the HEALPix grid.
        fwhm_x  : Full Width at Half Maximum, in the x (South-North) direction if phi_0=0 (degrees)
        fwhm_y  : Full Width at Half Maximum, in the y (East-West) direction if phi=0, (degrees)
        phi_0   : Angle between Noth-South direction and x-axis of the ellipse (degrees)
                  phi_0 increases clockwise towards East.
    
    optional input:
    
        max_extent : maximum extention of the beam. Defaults to 10. (degrees)
    
    returns
        
        beam    : HEALPix grid of NSIDE `nside`.
    '''
     
    # Convert FWHM in degres to sigma, in radians
    sigma_x = numpy.deg2rad( fwhm_x )/2.35482
    sigma_y = numpy.deg2rad( fwhm_y )/2.35482
    
    # Convert phi_0 to radians
    phi_0 = numpy.deg2rad( phi_0 )
    
    # Convert max_entent to radians
    max_extent = numpy.deg2rad( max_extent )

    # Get all pixels around the pole, up to a radius of `max_extent`
    beam_pixels = healpy.query_disc( nside, (0,0,1), max_extent )
    
    # Get co-latitude (increasing towards South) and longitude.
    theta, phi  = healpy.pix2ang( nside, beam_pixels )
    
    # Compute coefficients to properly rotate the ellipse.
    a =  (cos(phi_0)**2)/(2*sigma_x**2) + (sin(phi_0)**2)/(2*sigma_y**2)
    b = -(sin(2*phi_0) )/(4*sigma_x**2) + (sin(2*phi_0 ))/(4*sigma_y**2)
    c =  (sin(phi_0)**2)/(2*sigma_x**2) + (cos(phi_0)**2)/(2*sigma_y**2)
    
    # Elliptical beam using Polar Coordinates, by M.K. Brewer (CLASS Collaboration, 2018)
    beam_values = np.exp( -(a*cos(phi)**2 + 2*b*cos(phi)*sin(phi) + c*sin(phi)**2) * theta**2)
    
    # Beam is |E|^2. Transform to |E|.
    beam_values = numpy.sqrt( beam_values )
    
    # Create empty buffer and replace corresping zeroeth pixels with the beam values
    npixels  = healpy.nside2npix( nside )
    hpx_grid = numpy.zeros( npixels, dtype='float' )
    hpx_grid[ 0:beam_values.size ] = beam_values

    return hpx_grid

def make_gaussian_beam( nside, fwhm ):
    '''
    Returns a Gaussian Beam as a square grid.
    '''
    
    # Convert FWHM in degres to sigma in radians
    sigma = numpy.radians( fwhm )/2.35482
    
    pixels  = numpy.arange( healpy.nside2npix(nside) )
    tht,phi = healpy.pix2ang( nside, pixels )
    
    max_theta = numpy.radians( fwhm ) * 10
    
    #tht = tht[ healpy.ang2pix( nside, max_theta, 0.0 ) ]

    beam = numpy.sqrt( numpy.exp( -0.5*tht**2/sigma**2 ) )
   
    return beam.ravel()

def make_tophat_beam  ( size, grid_size, nx ):

    size = numpy.radians( size )
    
    # Create planar coordinates                                                                               
    x = numpy.linspace( -grid_size/2.0, grid_size/2.0, nx )                                                   
    y = numpy.linspace( -grid_size/2.0, grid_size/2.0, nx )                                                   
                                                                                                              
    x,y = numpy.meshgrid( x,y, sparse=True )                                                                               
    r   = numpy.arccos( numpy.cos(x) * numpy.cos(y) )  

    beam = numpy.zeros_like( r )
    beam[ r < size ] = 1.0

    return beam.ravel()

