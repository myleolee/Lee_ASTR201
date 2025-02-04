# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 01:02:33 2025

@author: Leo
"""

import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

#%%
#   Set up constants
e = const.e.esu.value
a0 = const.a0.to('cm').value
me = const.m_e.to('g').value
#%%
#   Set up the problem and initial condition
Z = 5
charge_center = Z * e
charge = -e
m_charge = me
b0 = 1500*a0
t_step = 10**-16   #   Set the time scale of each step

ini_px = -500*a0
ini_py = b0
ini_vx = 2*10**7

#   Set up the initial considtion
px = [ini_px]
py = [ini_py]


vx = [ini_vx]
vy = [0]

ax = [0]
ay = [0]

#%%
#   Set number of steps
num = 6000

#%%
def cal_f(x, y):
    dis = (x**2 + y**2)**0.5
    F = charge*charge_center/dis**2  #   Coulomb's Law
    direction = [x/dis, y/dis]
    ax = direction[0]*F/m_charge 
    ay = direction[1]*F/m_charge
    return ax, ay

#%%
def update():
    temp_ax, temp_ay = cal_f(px[-1], py[-1])
    px.append(px[-1] + vx[-1]*t_step)
    py.append(py[-1] + vy[-1]*t_step)
    vx.append(vx[-1] + ax[-1]*t_step)
    vy.append(vy[-1] + ay[-1]*t_step)
    ax.append(temp_ax)
    ay.append(temp_ay)

#%%    
for i in range(num):
    update()


#%%
#   Plot the xy path
fig, axes = plt.subplots(1, 1, figsize = (8, 8))
axes.plot(px, py)
axes.plot(0, 0, 'r.')
axes.set_title('xy path of the election')
axes.set_xlabel('x position (cm)')
axes.set_ylabel('y position (cm)')
#%%
time_arr = np.arange(num+1)
#   Plot xy velocity with time
fig, axes = plt.subplots(1, 1, figsize = (8, 8))
axes.plot(time_arr, vx, label = 'x velocity')
axes.plot(time_arr, vy, label = 'y velocity')
axes.legend()
axes.set_xlabel('Time ('+str(t_step)+'s)')
axes.set_ylabel(r'Velocity (cm $s^{-1}$)')
axes.set_title('xy velocity of the election')
#%%
#   Plot xy acceleration with time
fig, axes = plt.subplots(1, 1, figsize = (8, 8))
axes.plot(time_arr, ax, label = 'x acceleration')
axes.plot(time_arr, ay, label = 'y acceleration')
axes.legend()
axes.set_xlabel('Time ('+str(t_step)+'s)')
axes.set_ylabel(r'Acceleration (cm $s^{-2}$)')
axes.set_title('xy acceleration of the election')
#%%
arr = []
for i in range(num+1):
    arr.append((ax[i]**2 + ay[i]**2)**0.5)
acc = np.array(arr)

#   Doing the ffT   
fft_res = np.abs(fft.fft(acc))[:num//2]

#   Getting the fft frequency
fft_freq = fft.fftfreq(num+1, t_step)[:num//2]
#plt.plot(fft_freq, fft_res)
pw = (t_step*fft_res)**2*8*np.pi/3*e**2/const.c.cgs.value**3/t_step/num

#   Plot the poewr spectrum
fig, axes = plt.subplots(1, 1, figsize = (8, 8))
axes.plot(fft_freq, pw)
axes.set_xlim([-1e13, 1e14])
axes.set_xlabel('Frequency (Hz)')
axes.set_ylabel(r'Power per frequency (erg s$^{-1}$ Hz$^{-1}$)')
axes.set_title('Power spectum')
