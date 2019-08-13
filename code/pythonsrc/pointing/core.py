import astropy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy.utils import iers

import numpy

from . import _pointing
from . import _pointing_correction

import pylab

def compute_receiver_to_ICRS( tod, receiver, location, iers_table_path=None, return_sph=False ):
    """
    compute_receiver_to_ICRS:

    tod: an instance of <todsynth.tod>. Needs to have its pointing streams initialized (ctime,az,alt,rot)
         and its `pointing_mask` attribute set (zero by default, which means all samples in the TOD are
         taken as valid).

    receiver: an instance of <todsynth.receiver>. Needs its detector pointing values initialized( dx, dy, pol_angles )
              It is also advisable to set the `valid_detectors` to something else than the default (all detectors
              being valid).

    location: an instance of <astropy.coordinates.EarthLocation> representing the geographical location of
              the experiment (latitude, lontitude, altitude)

    iers_table: Path to the `finals_2000.all` file that `astropy` needs to compute the
                polar motions and utc_minus_ut1 difference. If set to `None`, astropy will try to download
                the table from the Internet. This can take a while so hold on tight!

    return_sph: If set to True, returns a tuple with receiver's longitude, co-latitude and parallactic angle.
    """
    
    # Output buffers
    ndets    = receiver.ndets
    nsamples = tod.nsamples

    # Get geographical coordinates from EarthLocation instance
    lat_rad  = numpy.radians( location.lat.value )
    lon_rad  = numpy.radians( location.lon.value )

	# Open IERS table
    IERS_table = iers.IERS_A.open()

    # Get the average polar motions for the TOD
    #----------------------------------------------------------------------------------------------------------#
    astropy_time = Time( numpy.average(tod.ctime), format='unix', scale='ut1' )
    #----------------------------------------------------------------------------------------------------------#
    # DUT1 is zero, as tod.ctime MUST BE in UT1 scale
    dut1 = 0.0
    pm_x, pm_y    = IERS_table.pm_xy( astropy_time )
    pm_x = numpy.radians( pm_x.value/3600.0 )
    pm_y = numpy.radians( pm_y.value/3600.0 )
    xp = pm_x
    yp = pm_y

    # setup output buffers
    recv_ra  = numpy.empty( (ndets, nsamples), dtype='float64' )
    recv_dec = numpy.empty( (ndets, nsamples), dtype='float64' )
    recv_pa  = numpy.empty( (ndets, nsamples), dtype='float64' )
    
    _pointing.get_receiver_ICRS_coords(
        tod.ctime, tod.az, tod.alt, tod.rot, 
        tod.pointing_mask, 
        receiver.dx, receiver.dy, receiver.pol_angles,
        lat_rad, lon_rad, 
        xp, yp, dut1, 
        recv_ra, recv_dec, recv_pa ) 

    if return_sph: recv_dec = numpy.pi/2.0 - recv_dec

    return (recv_ra, recv_dec, recv_pa)

def compute_receiver_to_horizontal( tod, receiver ):
    """
    """

    recv_az, recv_alt, recv_rot = numpy.zeros( (3, receiver.ndets, tod.nsamples), dtype='float64' )

    _pointing.get_receiver_horizontal_coords (
        tod.ctime, tod.az, tod.alt, tod.rot, 
        receiver.dx, receiver.dy, receiver.pol_angles,
        recv_az, recv_alt, recv_rot
    )

    return recv_az, recv_alt, recv_rot

 

def compute_receiver_source_centered( recv_lon, recv_lat, recv_pa, lon_source, lat_source ):
    """

    """

    phi_source_centered   = numpy.zeros_like( recv_pa  )
    theta_source_centered = numpy.zeros_like( recv_pa )

    _pointing.get_receiver_source_centered_coords(
        recv_lon, recv_lat, recv_pa,
        lon_source, lat_source,
        phi_source_centered, theta_source_centered )

    return phi_source_centered, theta_source_centered


def transform_horizontal_to_ICRS( ctime,  azimuth, altitude, rotation, location ):
    """
    """

def transform_ICRS_to_horizontal( ctime, ra, dec, location, iers_table_path=None ):
    """
    """
    # Get geographical coordinates from EarthLocation instance
    lat_rad  = numpy.radians( location.lat.value )
    lon_rad  = numpy.radians( location.lon.value )

    # Open IERS table
    IERS_table = iers.IERS_A.open( iers_table_path )

    # Get the average polar motions and utc_minus_ut1 values for the TOD
    astropy_time = Time( numpy.median(ctime), format='unix', scale='utc' )

    # get UTC - UT1 difference
    #----------------------------------------------------------------------------------------------------------#
    dut1 = IERS_table.ut1_utc( astropy_time ).value

    # get polar motions
    #----------------------------------------------------------------------------------------------------------#
    pm_x, pm_y    = IERS_table.pm_xy( astropy_time )
    pm_x = numpy.radians( pm_x.value/3600.0 )
    pm_y = numpy.radians( pm_y.value/3600.0 )
    xp = pm_x
    yp = pm_y

    # setup output buffers
    nsamples   = ctime.size
    source_az  = numpy.zeros( (nsamples), dtype='float64' )
    source_alt = numpy.zeros( (nsamples), dtype='float64' )

    _pointing.transform_ICRS_to_horizontal_coords( 
        ctime, 
        ra, dec, 
        lat_rad, lon_rad, xp, yp, dut1, 
        source_az, source_alt)


    return source_az, source_alt

    
def correct_pointing( tod, pmodel_dict ):
    """
    """

    # Check TOD has paraTilt and perpTilt attributes
    if not hasattr( tod, 'paraTilt' ):
        raise ValueError, 'TOD must have a Parallel Tilt stream called paraTilt.'
    
    if not hasattr( tod, 'perpTilt' ):
        raise ValueError, 'TOD must have a Perpendicular Tilt stream called perpTilt.'
    
    '''
    # Check dictionary keys
    if not all \
        (k in pmodel_dict for k in \
        ("az_coef","alt_coeff", "bs_coef", "az_pc", "alt_pc", "x_center", "y_center") ):

        raise ValueError, 'Incomplete Pointing Model dictionary.'
    '''
    # In-place pointing correction
    _pointing_correction.correct_pointing( 
        tod.az, tod.alt, tod.rot, 
        tod.paraTilt, tod.perpTilt, 
        pmodel_dict )
    return tod
