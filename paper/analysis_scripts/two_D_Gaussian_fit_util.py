import numpy as np

'''
This routine is set to rotate the beam counterclockwise on the sky. 
To rotate clockwise, set rot = cw.
'''
ccw = 1.0
cw = -1.0
rot = ccw

def gaussian_elliptical_beam((az, el), amp, fwhm_x, fwhm_y, xo, yo, theta):
    '''
    Returns an elliptical Gaussian beam.
    All arguments in degrees.
    '''
    y = el
    x = az * np.cos(np.radians(el))
    theta = np.radians(theta)
    # Convert FWHM to sigma 
    sigma_x = fwhm_x / 2.35482
    sigma_y = fwhm_y / 2.35482
   
    sx2 = sigma_x * sigma_x
    sy2 = sigma_y * sigma_y
    st = np.sin(theta)
    ct = np.cos(theta)
    st2 = st * st
    ct2 = ct * ct

    a = ct2 / sx2 + st2 / sy2
    b = 2.0 * st * ct * rot * (1.0 / sx2 - 1.0 / sy2)
    c = st2 / sx2 + ct2 / sy2
    gaussian = amp * np.exp( -0.5 * (a*(x-xo)**2 + b*(x-xo)*(y-yo) + c*(y-yo)**2)) 
    return gaussian.ravel()

def to_covariance(fwhm_x, fwhm_y, theta):
    '''
    An elliptical Gaussian of the form a * x^2 + b * x * y + c * y^2
    can be expressed as v^T * M * v where v is the vector (x,y) and 
    M is a 2 x 2 Hermetian matrix with covariance matrix M^-1.
    The average of such Gaussians has a covariance which is the 
    average of the covariances of the individual Gaussians that make
    up the average. Hence these functions.
    All arguments in degrees.
    ''' 
    theta = np.radians(theta)
   
    sx2 = fwhm_x * fwhm_x
    sy2 = fwhm_y * fwhm_y
    st = np.sin(theta)
    ct = np.cos(theta)
    st2 = st * st
    ct2 = ct * ct

    a = ct2 / sx2 + st2 / sy2
    b = 2.0 * st * ct * rot * (1.0 / sx2 - 1.0 / sy2)
    c = st2 / sx2 + ct2 / sy2
  
    det = sx2 * sy2
    c00 =  c * det
    c01 = -b * det / 2.0
    c11 =  a * det
    return c00, c01, c11

def from_covariance(c00, c01, c11):
    '''
    An elliptical Gaussian of the form a * x^2 + b * x * y + c * y^2
    can be expressed as v^T * M * v where v is the vector (x,y) and 
    M is a 2 x 2 Hermetian matrix with covariance matrix M^-1.
    The average of such Gaussians has a covariance which is the 
    average of the covariances of the individual Gaussians that make
    up the average. Hence these functions.
    Return values in degrees.
    ''' 
    det = c00 * c11 - c01 * c01
    a = c11 / det
    b = -2.0 * c01 * rot / det
    c = c00 / det
    # this is actually 2 * theta
    theta = np.arctan2(b, a - c)
    ct = np.cos(theta)
    st = np.sin(theta)
    if abs(ct) > abs(st):
        fwhm_x = np.sqrt(2.0 / (a + (a - c) / ct + c))
        fwhm_y = np.sqrt(2.0 / (a - (a - c) / ct + c))
    else:
        fwhm_x = np.sqrt(2.0 / (a + b / st + c))
        fwhm_y = np.sqrt(2.0 / (a - b / st + c))
    theta = np.degrees(theta / 2.0)
    return fwhm_x, fwhm_y, theta

