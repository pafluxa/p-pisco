# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt

import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

nside = 512
lmax = 4*nside - 1

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()

#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965)

pars.set_for_lmax( lmax, lens_potential_accuracy=2)
pars.DoLensing   = False
pars.WantTensors = True
results = camb.get_transfer_functions(pars)

inflation_params = initialpower.InitialPowerLaw()

plt.ylabel( r"$C_\ell^{BB} \, (\mu \mathrm{K})^2$" )
plt.xlabel( r"$\ell$" )

for r in [0,0.01,0.1]:

    inflation_params.set_params(ns=0.96, r=r )
    results.power_spectra_from_transfer(inflation_params) #warning OK here, not changing scalars
    cl = results.get_total_cls(lmax, CMB_unit='K')

    plt.loglog( cl[2:,2], label="r = {:2.2f}".format(r) ) 
    plt.legend()

    np.savetxt( '../data/cls/cls_lcdm_512_r{:3.2f}.txt'.format(r) , cl )

plt.show()

