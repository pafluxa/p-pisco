import numpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import time

parser = argparse.ArgumentParser(description='Plot CLASS Q-band experiment elliptical Gaussian beam position angle vs declination hit map.')


parser.add_argument( '-pointing', action='store', type=str, dest='pointing',
                      help='NPZ file with the pointing of the season. See generate_scanning_icrs.py for more details.' )

args = parser.parse_args()
#----------------------------------------------------------------------------------------------------------#
# Read in pointing stream
#----------------------------------------------------------------------------------------------------------#
print 'loading pointing'
print args.pointing

pointing = np.load( args.pointing , mmap_mode='r')
feed_dec = pointing['dec']
feed_pa  = pointing['pa']
#----------------------------------------------------------------------------------------------------------#

#Set up hit map
resolution = 1 #degrees
n_dec      = 115.0 // resolution
n_pa       = 90.0  // resolution

dec  = np.linspace(-80.0, 35.0, n_dec)
pa   = np.linspace(  0.0, 90.0, n_pa)
hits = np.zeros( (len(dec) + 1, len(pa)+1)  )

c0 = time.clock()
t0 = time.time()

for p in pa:
    
    print p
    pa_rad = numpy.deg2rad( p )
    
    # all points falling in the range pa, pa + 90
    mask = numpy.where( 
                numpy.logical_and( feed_pa >= pa_rad , feed_pa <= pa_rad + numpy.pi/2.0 ) )
    
    # find all declinations
    _feed_pa  = feed_pa [ mask ]
    _feed_dec = feed_dec[ mask ]
    
    #Fold pa into one quadrant and convert to indices
    _feed_dec = (np.degrees( _feed_dec) + 80.0) // resolution
    _feed_pa  =  np.degrees( np.arctan( np.abs(np.tan(_feed_pa)) ) ) // resolution
    
    for k in range( _feed_dec.shape[0] ):
        print k, _feed_dec.shape[0]

        idx = int(_feed_pa[k])
        jdx = int(_feed_dec[k])
        hits[jdx][idx] += 1.0

t1 = time.time()
c1 = time.clock()
print ("for loop: wall %6.3lf cpu %6.3lf" % ((t1 - t0), (c1 - c0)))

plt.figure()
plt.title('Hit Map')
plt.xlabel('Position Angle')
plt.ylabel('Declination')
plt.imshow(hits, origin='lower', extent=(pa.min(), pa.max(), dec.min(), dec.max()), cmap='plasma')
plt.colorbar()
plt.savefig('beam_sym_map.png')




