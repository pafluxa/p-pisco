import scipy.optimize as opt
import numpy as np
import pylab as plt
from astropy import wcs
from two_D_Gaussian_fit_util import  gaussian_elliptical_beam, to_covariance, from_covariance 

# Create x and y indices
x = np.arange(-4., 4., 0.05) + 0.025
#print x[0], x[-1]
x, y = np.meshgrid(x, x)

fwy = 1.50
fwx = 1.0
tht = 20.0
xoff= 0.0
yoff= 0.05
#create data
data1 =  gaussian_elliptical_beam((x, y), 3, fwx, fwy, xoff, yoff, tht)
fwy = 1.5
fwx = 1.0
tht = 20.0
xoff= 0.05
yoff= 0.0
data2 = gaussian_elliptical_beam((x, y), 3, fwx, fwy, xoff, yoff, tht)
data = data2 - data1
# plot twoD_Gaussian data generated above
plt.figure()
plt.imshow(data.reshape(160, 160),origin='lower', extent=(x.min(), x.max(), y.min(), y.max()),cmap='seismic')
#plt.imshow(data.reshape(160, 160),origin='lower',cmap='seismic')
plt.colorbar()
plt.suptitle('Elliptical Gaussian Beam Pair Difference \n Delta x_off -0.05, Delta y_off 0.05')
plt.show()
