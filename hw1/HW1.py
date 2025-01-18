# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 03:58:20 2025

@author: Leo
"""

import astropy.units as u
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#%%
# 1. blackbody spectrum of the Sun as observed at 10 pc

#   Define constant
h = const.h.to('erg s')
c = const.c.to('cm/s')
kB = const.k_B.to('erg/K')
R_sun = const.R_sun.to('pc')
R_earth = const.R_earth.to('pc')
pc = const.pc.to('km')
au = const.au.to('pc')

#   Define the wavelength range (0.1 - 100um)
nv = np.logspace(np.log10(3e12), np.log10(3e15), 1000)


#   The Planck function
def planck(T):
    B = (2*h*nv**3/c**2)*((np.exp(np.array(h*nv/kB/T))-1)**(-1))
    return B

#   Combining with distance effect
def flux_den(T, r, d):
    B = planck(T)
    return np.pi*B*(r/d)**2

spec_sun = flux_den(5500, R_sun, 10)     #   Assuming the sun is 5500K

fig, ax = plt.subplots(1, 1)
ax.plot(nv, spec_sun)
ax.set_xscale('log')
ax.set_xlim(3e12, 3e15)
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'Flux Density at 10 pc (erg/s/cm$^{2}$/Hz)')

#%%
#   M5 dwarf + Sunlike star
spec_M5 = flux_den(3050, 0.207*R_sun, 10)   #   Spectrum of M5 dwarf

fig, ax = plt.subplots(1, 1)
ax.plot(nv, spec_M5, color = 'blue', label = 'M5')
ax.plot(nv, spec_sun, color = 'orange', label = 'Sun-like star')
ax.plot(nv, spec_M5 + spec_sun, color = 'black', label = 'total')
ax.set_xscale('log')
ax.set_xlim(3e12, 3e15)
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'Flux Density at 10 pc (erg/s/cm$^{2}$/Hz)')
ax.legend()

#   M5 dwarf + white dwarf
spec_white = flux_den(10000, R_earth, 10)   #   Spectrum of WD

fig, ax = plt.subplots(1, 1)
ax.plot(nv, spec_M5, color = 'blue', label = 'M5')
ax.plot(nv, spec_white, color = 'orange', label = 'white dwarf')
ax.plot(nv, spec_M5 + spec_white, color = 'black', label = 'total')
ax.set_xscale('log')
ax.set_xlim(3e12, 3e15)
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'Flux Density at 10 pc (erg/s/cm$^{2}$/Hz)')
ax.legend()

#   Arcturus + neutron star
spec_arct = flux_den(4286, 25.4*R_sun, 10)  #   Spectrum of Arcturus
spec_neu = flux_den(1000000, 11/pc.value, 10)  #   Spectrum of neutron star

fig, ax = plt.subplots(1, 1)
ax.plot(nv, spec_arct, color = 'blue', label = 'Arcturus')
ax.plot(nv, spec_neu, color = 'orange', label = 'neutron star')
ax.plot(nv, spec_arct.value + spec_neu.value, color = 'black', label = 'total')
ax.set_xscale('log')
ax.set_xlim(3e12, 3e15)
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'Flux Density at 10 pc (erg/s/cm$^{2}$/Hz)')
ax.legend()


#%%

#   Calculating the temperture of the cloud given the flux
def temp(flux):
    return (flux/2/(5.67e-5))**(1/4)

#   Integrating equation 3
T = 3000 * u.K
nv = np.logspace(0, 20, 10000)
def integrand(nv):
    nv = nv * u.Hz
    return ((np.pi*2*h*nv**3/c**2)*((np.exp(h*nv/kB/T)-1)**-1)*((5*R_sun/(100*au))**2)*(1-np.exp(-nv/(3e14 * u.Hz)))).value

x = quad(integrand, (const.c/(100*u.micron)).to('Hz').value, (const.c/(10*u.nm)).to('Hz').value)   #   10e18 since most of the flux is included
print(temp(x[0]))

#%%
#   Define the wavelength range
nv = np.logspace(np.log10(3e10), np.log10(3e16), 10000)
spec_star = flux_den(3000, 5*R_sun, 10) * np.exp(-nv/3e14)
spec_cloud =  flux_den(temp(x[0]), 5*R_sun + 100*au, 10) * (1-np.exp(-nv/3e14))

fig, ax = plt.subplots(1, 1)
ax.plot(nv, spec_star, color = 'blue', label = 'star')
ax.plot(nv, spec_cloud, color = 'orange', label = 'cloud')
ax.plot(nv, spec_star + spec_cloud, color = 'black', label = 'total')
ax.set_xscale('log')
ax.set_xlim(3e10, 3e16)
ax.set_ylim(10e-40, 10e-20)
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'Intensity (erg/s/cm$^{2}$/Hz)')
ax.legend()


        









