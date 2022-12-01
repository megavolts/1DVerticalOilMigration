#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
import os
import matplotlib.pyplot as plt  # Hunter J., 2007
import numpy as np  # VanDerWalt S. et al., 2011
import scipy.optimize as spo  # Virtanen et al., 2020

# Figures
w_fig = 6
h_fig = 3

# Directories
data_dir = '/mnt/data/UAF-data/paper/4/'
fig_dir = '/home/megavolts/UAF/paper/04-oil_movement_model/figures/'

# Filenames
mu_oil_fn = 'oil_properties/oil_viscosity-ScottLoranger.csv'
rho_oil_fn = 'oil_properties/oil_density.csv'

# Import data
mu_oil_fp = os.path.join(data_dir, mu_oil_fn)
mu_oil_data = np.genfromtxt(mu_oil_fp, delimiter=',', skip_header=2)

rho_oil_fp = os.path.join(data_dir, rho_oil_fn)
rho_oil_data = np.genfromtxt(rho_oil_fp, delimiter=',', skip_header=2)

# Fit arrhenius Law to viscosity data
popt, pcov = spo.curve_fit(lambda T, A, B: A*np.exp(-B/(np.pi*(T+273.15))), mu_oil_data[:, 0], mu_oil_data[:, 1])

def arrheniuslaw(t, popt):
    return popt[0] * np.exp(-popt[1]/(np.pi*(t+273.15)))

T = np.arange(-20, 31, 1)
fig = plt.figure(figsize=[w_fig, h_fig])
ax = plt.subplot(1, 1, 1)
ax.plot(mu_oil_data[:, 0], mu_oil_data[:, 1]*1e-3, 'ko', label='measurement')
ax.plot(T, arrheniuslaw(T, popt)*1e-3, 'r', label='fitted Arrhenius Law')
ax.set_xlabel('Temperature ($^{\circ}C$)')
ax.set_ylabel('Viscosity (Pa s)')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()


# Fit linear curve to density data
popt, pcov = spo.curve_fit(lambda T, A, B: A*T+B, rho_oil_data[:, 0], rho_oil_data[:, 1])

def linear(t, popt):
    return popt[0]*t + popt[1]

# Figure
T = np.arange(-20, 31, 1)
fig = plt.figure(figsize=[w_fig, h_fig])
ax = plt.subplot(1, 1, 1)
ax.plot(rho_oil_data[:, 0], rho_oil_data[:, 1]*1e3, 'ko', label='measurement')
ax.plot(T, linear(T, popt)*1e3, 'r', label='fitted Linear curve')
ax.set_xlabel('Temperature ($^{\circ}C$)')
ax.set_ylabel('Density (kg m$^{-3}$)')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()

