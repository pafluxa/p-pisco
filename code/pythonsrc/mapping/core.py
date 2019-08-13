import healpy
import numpy
from pisco.mapping._mapping import *

def update_cubes( 
    phi, theta, psi, pol_angles, 
    data, 
    map_nside,
    data_mask=None,
    det_mask=None,
    pixels_in_map=None,
    AtA=None, AtD = None ):
    '''
    '''
    
    ndets = data.shape[0]

    map_size = 0
    if pixels_in_map is None:
        map_size = healpy.nside2npix( map_nside )
        pixels_in_map = numpy.arange( map_size , dtype='int32' )
    
    else:
        map_size = pixels_in_map.size

    if data_mask is None:
        data_mask = numpy.zeros_like( data, dtype='int32' )

    if AtA is None:
        AtA = numpy.zeros( (ndets , map_size , 9 ) , dtype='float64')
    if AtD is None:
        AtD = numpy.zeros( (ndets , map_size , 3 ) , dtype='float64')

    if det_mask is None:
        det_mask = numpy.zeros( ndets, dtype='int32' )

    pixels_in_map = sorted( pixels_in_map )

    project_data_to_cubes(
		phi, theta, psi, pol_angles,
		data.astype('float64'), data_mask, det_mask,
		map_nside,
		pixels_in_map,
		AtA.ravel(), AtD.ravel() )
    
    # Reshape to original dimensions
    AtA = AtA.reshape( ( ndets, map_size, 9 ) )
    AtD = AtD.reshape( ( ndets, map_size, 3 ) )

    return AtA, AtD

def update_matrices( 
    phi, theta, psi, pol_angles, 
    data, 
    map_nside,
    data_mask=None,
    det_mask=None,
    pixels_in_map=None,
    AtA=None, AtD = None ):
    '''
    '''
    
    map_size = 0
    if pixels_in_map is None:
        map_size = healpy.nside2npix( map_nside )
        pixels_in_map = numpy.arange( map_size , dtype='int32' )
    
    else:
        map_size = pixels_in_map.size

    if data_mask is None:
        data_mask = numpy.zeros_like( data, dtype='int32' )

    if AtA is None:
        AtA = numpy.zeros( 9*map_size , dtype='float64')
    
    if AtD is None:
        AtD = numpy.zeros( 3*map_size , dtype='float64')

    if det_mask is None:
        det_mask = numpy.zeros( data_mask.shape[0], dtype='int32' )

    pixels_in_map = sorted( pixels_in_map )

    project_data_to_matrices(
		phi, theta, psi, pol_angles,
		data.astype('float64'), data_mask, det_mask,
		map_nside,
		pixels_in_map,
		AtA, AtD )
        
    return AtA, AtD

def matrices_to_maps( map_nside, AtA, AtD, pixels_in_map=None ):
    '''
    '''

    map_size = 0
    if pixels_in_map is None: 
    
        map_size = AtA.size/ 9
        pixels_in_map = numpy.arange( map_size, dtype='int32' )
    
    else:
    
        map_size = pixels_in_map.size
        
    pixels_in_map = sorted( pixels_in_map )

    I,Q,U,W = numpy.zeros( (4,map_size) )
    get_IQU_from_matrices( map_nside, AtA, AtD, pixels_in_map, I,Q,U,W )

    return I,Q,U,W




