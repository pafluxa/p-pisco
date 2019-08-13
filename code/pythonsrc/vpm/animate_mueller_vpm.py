import pylab
import healpy
import numpy
from mueller_matrix import M_vpm_to_healpix

angles    = numpy.linspace( 0.0, 2*numpy.pi, 10 )
distances = 0.003 * numpy.cos( angles )

k = 2*numpy.pi/0.008

step = 0
nside = 256
names = ['MII','MQI','MUI','MVI',
         'MIQ','MQQ','MUQ','MVQ',
         'MIU','MQU','MUU','MVU',
         'MIV','MQV','MUV','MVV', ]
for d in distances:

    M = M_vpm_to_healpix( d, k, 0.2, numpy.pi/3.0, beam_nside=nside )

    M_map = numpy.zeros( (4,4, healpy.nside2npix( nside ) ) )
   
    fig = pylab.figure( figsize=(10,8) )

    for i in range(16):
        
        m,n = numpy.unravel_index( i, (4,4) )

        M_map[m][n][0:M.shape[2]] = M[m][n]
        healpy.gnomview( M_map[m][n], rot=(0,90), sub=(4,4,i+1), title=names[i], reso=10, notext=True, min=0, max=5, fig=0  )
    
    pylab.savefig( 'step_%d_vpm_mueller_matrices.png' % (step) )

    step = step + 1

    del fig
