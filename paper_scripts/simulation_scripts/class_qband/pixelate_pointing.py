import sys

import numpy 
import healpy

nside = 128
pointing = numpy.load( sys.argv[1], mmap_mode='r')

det_ra  = pointing['ra']
det_dec = pointing['dec']
det_pa  = pointing['pa']

ipix = healpy.ang2pix(nside, numpy.pi / 2 - det_dec, det_ra)

det_tht_pix, det_ra_pix = healpy.pix2ang(nside, ipix)
det_dec_pix = numpy.pi / 2.0 - det_tht_pix

# det_pa doesn't variate too much, so we keep it the same
det_pa_pix = det_pa

numpy.savez( sys.argv[2], ra=det_ra_pix, dec=det_dec_pix, pa=det_pa_pix )

