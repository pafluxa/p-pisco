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

def map_stats_mkb(nside, ra, dec):

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
    delx /= numpy.cos(dec_pix)
    hits   = numpy.zeros(npix)
    x_mean = numpy.zeros(npix)
    y_mean = numpy.zeros(npix)
    for idx in range(npix):
        jdx = numpy.where(pixels == idx)[0]
        if len(jdx):
            hits[idx] = len(jdx)
            y_mean[idx] = numpy.mean(dely[jdx])
            x_mean[idx] = numpy.mean(delx[jdx])

    mask = numpy.where( hits > 0)[0]    

    print 'Dec RMS  ', numpy.degrees(numpy.std(y_mean[mask]))
    print 'RA  RMS  ', numpy.degrees(numpy.std(x_mean[mask]))
    print 'Mean Hits', numpy.mean(hits[mask])

    return x_mean, y_mean, hits

def map_stats_parallel(nside, ra, dec):
    '''
    returns 3 healpix maps with mean observed declination (x_mean),
    mean observed RA (y_mean) and hits per pixel (hits)
    '''
    
    # n threads
    nt = 2

    nsamp = ra.size
    npix  = healpy.nside2npix( nside )
    
    s_hits  = pymp.shared.array((nt, npix), dtype=float)
    s_xmean = pymp.shared.array((nt, npix), dtype=float)
    s_ymean = pymp.shared.array((nt, npix), dtype=float)
    
    with pymp.Parallel( nt ) as p:
    
        t = p.thread_num
        
        pixels = numpy.arange(npix)
        
        dec_pix, ra_pix = healpy.pix2ang( nside, pixels )
        dec_pix = numpy.pi / 2 - dec_pix
        
        for idx in p.range(0, nsamp):
        
            ipix = healpy.ang2pix(nside, numpy.pi / 2 - dec[idx], ra[idx])
            
            s_hits[t][ipix] += 1.0
            
            s_ymean[t][ipix] += (dec[idx] - dec_pix[ipix])
            
            delx = ra[idx] -  ra_pix[ipix]
            
            if delx > numpy.pi:    delx -= 2.0 * numpy.pi
            elif delx < -numpy.pi: delx += 2.0 * numpy.pi
            
            s_xmean[t][ipix] += delx * numpy.cos(dec_pix[ipix]) 
    
    hits   = numpy.zeros(npix)
    x_mean = numpy.zeros(npix)
    y_mean = numpy.zeros(npix)
    
    for t in range( nt ): 
        hits   +=  s_hits[t]
        x_mean += s_xmean[t]
        y_mean += s_ymean[t]

    # mask all pixels by default
    coverageMask = numpy.ones_like( hits, dtype=bool )

    # and unmask the ones with hits.
    coverageMask[ hits > 0 ] = False
    
    # transform hits, x_mean and y_mean to masked arrays
    hits   = numpy.ma.masked_array(   hits, mask=coverageMask )
    x_mean = numpy.ma.masked_array( x_mean, mask=coverageMask )
    y_mean = numpy.ma.masked_array( y_mean, mask=coverageMask )

    # I am pretty sure this avoids the unseen pixels
    y_mean /= hits
    x_mean /= hits
    
    print 'Dec RMS  ', numpy.degrees(y_mean.std())
    print 'RA  RMS  ', numpy.degrees(x_mean.std())
    print 'Mean Hits', hits.mean()
    
    return x_mean, y_mean, hits


nside = 128
pointing = numpy.load( sys.argv[1] )

ra,dec = pointing['ra'], pointing['dec']

print "computing on", ra.size, "pointing directions."

c0 = time.clock()
t0 = time.time()

map_stats(nside, ra.ravel(), dec.ravel() )
t1 = time.time()
c1 = time.clock()
print ("map_stats(): elapse: wall %6.3lf cpu %6.3lf" % ((t1 - t0), (c1 - c0)))

#numpy.savez( './map_stats_for_CLASS.npz', ram=ram, decm=decm, hitsm=hitsm )

