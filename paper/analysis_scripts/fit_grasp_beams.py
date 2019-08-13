import numpy as np
from glob import glob
from scipy.optimize import curve_fit
from two_D_Gaussian_fit_util import  gaussian_elliptical_beam, to_covariance, from_covariance

print 'det,azoff,az_std,eloff,el_std,fwhm_x,fwx_std,fwhm_y,fwy_std,theta,theta_std'
data_files = sorted(glob('./results/detector_*.npz'))
for i,  data_file in enumerate(data_files):
    data = np.load(data_file)
    det = data_file.split('_')[1]
    det = int(det.split('.')[0])
    
    az = np.linspace(data['beam_limits_deg'][0], data['beam_limits_deg'][1], data['beam_size'][0])
    el = np.linspace(data['beam_limits_deg'][2], data['beam_limits_deg'][3], data['beam_size'][1])
    beam = data['freq_avg_beam_linear']

    xg, yg = np.meshgrid(az, el, indexing='xy')   
    initial_guess = (13333.0, 1.5, 1.5, 0.0, 0.0, 0.0)
    popt, pcov = curve_fit(gaussian_elliptical_beam,(xg.ravel() , yg.ravel() ), beam.ravel(), p0=initial_guess)
    for i in range(6):    
        pcov[i][i] = np.sqrt(pcov[i][i])

    # This will make sure that fwy is the semi major axis and theta is in the correct quadrant
    # theta is counterclockwise positive from the y axis
    fwx0 = popt[1]
    fwx_std = pcov[1][1]
    fwy0 = popt[2]
    fwy_std = pcov[2][2]
    tht = popt[5]
    c00, c01, c11 = to_covariance(fwx0, fwy0, tht)
    fwx, fwy, tht = from_covariance(c00, c01, c11)
    if (fwx0 > fwy0) and (fwy > fwx):
        # axes reversed
        #print 'semi-axes reversed'
        fwy_std = pcov[1][1]
        fwx_std = pcov[2][2] 
    
    print "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf" % (det,popt[3],pcov[3][3],popt[4],pcov[4][4],fwx,fwx_std,fwy,fwy_std,tht,pcov[5][5])
 

