# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:22:09 2025

@author: Leo
"""

import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
#%%
#   This part is to generate a uniform sphere, that can generate random directions for scattering

N = 1000    #   Number of points

#   Generate N random points representing N random direction on a sphere
def set_N_points(N):
    theta_list = []
    phi_list = []
    for i in range(N):
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.arccos(1 - 2*np.random.uniform(0,1))
        theta_list.append(theta)
        phi_list.append(phi)
    return np.array(theta_list), np.array(phi_list)

theta, phi = set_N_points(N)

#   Convert theta and phi into x, y and z
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

#   Plot the uniform sphere that consist of N points
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(x,y,z, '.')

#%%
#   Set up the constants in the problem

AU = const.au.to('cm').value
M_cloud = const.M_sun.to('g').value   # Mass of the cloud
M_proton = const.m_p.to('g').value   # Mass of proton
N_proton = M_cloud/M_proton   # Number of proton
R_cloud = 0.001*const.pc.to('cm').value   # Radius of the cloud
V_cloud = 4/3*R_cloud**3*np.pi   # Volume of the cloud
n_proton = N_proton/V_cloud   # Number density of proton
sig_T = const.sigma_T.to('cm^2').value  #  Thompson cross section
l_mfp = 1/(sig_T*n_proton)   #  The mean free path of photon between scattering
R_shell = np.linspace(0, R_cloud, 101)   # An array consisting of the radius of each shell (The cloud is divided into 100 shell)

#%%

#   Function for generating a random direction
def gen_dir():
    phi = np.random.uniform(0,2*np.pi)
    theta = np.arccos(1 - 2*np.random.uniform(0,1))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

#   Function for updating the position of the photon
def update(x_pos, y_pos, z_pos):
    dir_x, dir_y, dir_z = gen_dir()
    return [x_pos+dir_x*l_mfp, y_pos+dir_y*l_mfp, z_pos+dir_z*l_mfp]

#   Function to propagate a photon from its initial position to ouside the cloud
def propagate(ini_pos):
    R = 0
    pos = ini_pos
    pos_x = [ini_pos[0]]
    pos_y = [ini_pos[1]]
    pos_z = [ini_pos[2]]
    rad_bin = np.zeros(len(R_shell)-1)
    while (R < R_cloud):
        pos = update(pos_x[-1], pos_y[-1], pos_z[-1])   #   Update position of photon
        pos_x.append(pos[0])    #   Record the position of photon
        pos_y.append(pos[1])    #   Record the position of photon
        pos_z.append(pos[2])    #   Record the position of photon
        R = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)  #   Calculate the distance between the center 
        #of the cloud and the photon to check whether the photon is still in the cloud
        #   Record which shell the photon is in for each step   
        if (len(np.where(R_shell > R)[0]) > 0):
            r_bin = np.where(R_shell > R)[0][0]-1
            rad_bin[r_bin] += 1
    last_sca = np.sqrt(pos_x[-2]**2 + pos_y[-2]**2 + pos_z[-2]**2)  #   Record the position of the photon where the last scatter occurs
    return pos_x, pos_y, pos_z, rad_bin, last_sca

#%%
#   Round scitific number
def precision_round(number, digits=3):
    power = "{:e}".format(number).split('e')[1]
    return round(number, -(int(power) - digits))

#   Plot a sphere with the radius of the cloud
def plot_sphere():
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    x = R_cloud * np.sin(theta) * np.cos(phi)
    y = R_cloud * np.sin(theta) * np.sin(phi)
    z = R_cloud * np.cos(theta)
    return x/AU, y/AU, z/AU
#%%
#   Function to start one single simulation
def sim(plot = False):
    ini_pos = [0, 0, 0]
    x, y, z, rad_bin, last_sca = propagate(ini_pos)
    total_dis = precision_round(((np.sqrt(x[-1]**2 + y[-1]**2 + z[-1]**2)*l_mfp)/AU))
    if (plot == True):
        ax = plt.axes(projection='3d')
        ax.plot(x/AU, y/AU, z/AU, '-')
        #ax.set_title('Total distance:' + str(total_dis) + ' AU')
        sph_x, sph_y, sph_z = plot_sphere()
        ax.plot_surface(sph_x, sph_y, sph_z, cmap='gray', alpha=0.1)
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
    return total_dis, rad_bin, last_sca

sim(True)

#%%
num_of_sim = 1000   #   Number of simulations

#%%
total_dis = np.zeros(num_of_sim)
total_rad_bin = np.zeros(len(R_shell) - 1)
last_sca = np.zeros(num_of_sim)

#   For each simulation, the total distance travelled, distribution of of photons in each radial bin 
#   and the postion that the last scatter occurs are marked down
for i in range(num_of_sim):
    t_dis, r_bin, last = sim()
    total_dis[i] = t_dis
    total_rad_bin = total_rad_bin + r_bin
    last_sca[i] = last

#   Calculate the volume of each shell to be divied when calculating the radial distribution of intensity "density"
V_shell = 4*np.pi*(R_shell[1:]**2)*(R_shell[1:] - R_shell[:-1])
#%%
#   Plot the distribution of total distance travelled and light travel times

fig, ax = plt.subplots(2, 1, figsize = (6,9))
ax[0].hist(total_dis/AU, bins = 20, edgecolor='black')
ax[0].set_xlabel('Distance travelled (AU)')
ax[0].set_ylabel('Number of simulations')
ax[1].hist(total_dis/const.c.to('cm/s').value/86400, bins = 20, edgecolor='black')
ax[1].set_xlabel('Time travelled (days)')
ax[1].set_ylabel('Number of simulations')
fig.tight_layout()
#%%
#   Plot the radial distribution of the intensity “density” through the cloud
fig, ax = plt.subplots(1, 1, figsize = (9,9))
ax.plot(R_shell[:-1]/AU, total_rad_bin/V_shell*(AU**3), 'k-')
ax.set_yscale('log')
ax.axvline(l_mfp/AU)
ax.set_xlabel('Radial distance from center (AU)')
ax.set_ylabel(r'Scattering density (AU$^{-3}$)')

#%%
#   Plot the radial distribution of the last scattering for the rays
fig, ax = plt.subplots(1, 1, figsize = (9,9))
ax.hist(last_sca/AU, bins = np.linspace((R_cloud - l_mfp)/AU, R_cloud/AU, 30), edgecolor='black')
ax.set_ylabel('Number of simulations')
ax.set_xlabel('Distance from the center (AU)')

