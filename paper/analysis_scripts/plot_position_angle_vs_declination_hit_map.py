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
resolution = 0.25 #degrees
n_dec = 115.0 / resolution
n_pa = 90.0 / resolution
dec = np.linspace(-80.0, 35.0, n_dec)
pa =  np.linspace(  0.0, 90.0, n_pa)
hits = np.zeros((len(dec), len(pa)))

ndet = np.shape(feed_dec)[0]
nsample = np.shape(feed_dec)[1]
#Fold pa into one quadrant and convert to indices
feed_dec = (np.degrees(feed_dec) + 80.0) // resolution
feed_pa = np.degrees( np.arctan( np.abs(np.tan(feed_pa)) ) ) // resolution

c0 = time.clock()
t0 = time.time()

for k in range(ndet):
    for l in range(nsample):
        idx = int(feed_pa[k][l])
        jdx = int(feed_dec[k][l])
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
plt.savefig('pa_v_dec_hit_map.png')




