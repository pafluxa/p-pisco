import numpy
import healpy

import pymp
import sys

import numpy                                                                                                  
import healpy  
import time
                                                                                             
def map_stats(nside, ra, dec):

    npix = healpy.nside2npix( nside )
    pixels = healpy.ang2pix(nside, numpy.pi / 2 - dec, ra)
    dec_pix, ra_pix = healpy.pix2ang( nside, pixels )
    dec_pix = numpy.pi / 2 - dec_pix
    dely = dec - dec_pix
    delx = ra - ra_pix
    mask = numpy.where(delx > numpy.pi)[0]
    delx[mask] -= 2.0 * numpy.pi
    mask = numpy.where(delx < -numpy.pi)[0]
    delx[mask] += 2.0 * numpy.pi
    delx *= numpy.cos(dec_pix)
    x_mean = numpy.zeros(npix)
    y_mean = numpy.zeros(npix)
    start = 0
    end = 0
    hits = numpy.bincount(pixels)
    pixels = numpy.argsort(pixels)
    for idx, num in enumerate(hits):
        if num:
            end += num
            jdx = pixels[start:end]
            y_mean[idx] = numpy.mean(dely[jdx])
            x_mean[idx] = numpy.mean(delx[jdx])
            start += num

    mask = numpy.where( hits > 0)[0]    
    
    print 'Dec RMS  ', numpy.degrees(numpy.std(y_mean[mask]))
    print 'RA  RMS  ', numpy.degrees(numpy.std(x_mean[mask]))
    print 'Mean Hits', numpy.mean(hits[mask])

    return x_mean, y_mean, hits, mask


nside = 256
pointing = numpy.load( sys.argv[1] )

ra,dec = pointing['ra'], pointing['dec']

print "computing on", ra.size, "pointing directions."

c0 = time.clock()
t0 = time.time()

raM,decM,hits,mask = map_stats(nside, ra.ravel(), dec.ravel() )

t1 = time.time()
c1 = time.clock()

print ("map_stats(): elapse: wall %6.3lf cpu %6.3lf" % ((t1 - t0), (c1 - c0)))

numpy.savez( './map_stats_for_CLASS.npz', raMean=raM, decMean=decM,hits=hits,mask=mask )
