#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt  # Hunter J., 2007
import matplotlib.colors as colors  # Log scale in contour

try:
    from project_helper import *
except ImportError:
    import sys
    if '/home/megavolts/git/paper3' in sys.path:
        raise
    sys.path.append('/home/megavolts/git/paper3')
    from project_helper import *

# Variable definition
data_dir = '/mnt/data/UAF-data/paper/4/'
case_fn = 'oil_spill_case/Test-1_10_10.ini'
fig_dir = '/home/megavolts/UAF/paper/Chapter4/figures/'

# Initial condition
T_i = -5  # °C, initial sea-ice temperature
T_sw = -5  # °C, initial sea-ice temperature
S_si = 5  # ‰, initial sea-ice bulk salinity
S_b = 60  # ‰, initial brine salinity
S_sw = 32  # ‰, initial sea-water salinity

# Load boundary condition
case_fp = case_fn
bc_dict = load_case(case_fp)

# Create material dicitionnary
mat_dict = {'sw': {'S': S_sw},
            'o': {'Tsw': t_sw_f(S_sw), 'Ti': T_i, 'gamma': GAMMA_o, 'theta': THETA_o},
            'b': {'Ti': T_i, 'S': S_b, 'gamma': GAMMA_b, 'theta': THETA_b}}

HI = bc_dict['HI']
HD = bc_dict['HD']

# Fix
HR = 0.1
VR = 1000
Rmin = 1e-5
Rmax = 1e-2

# Create oil spill case dictionnary
os_dict = bc_dict
os_dict['HR'] = HR
os_dict['VR'] = VR
os_dict.update(mat_dict)

hr = np.linspace(0, 1, 1001)
ho = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 1]

r = np.logspace(np.log10(Rmin), np.log10(Rmax), 1001)  # m
hr = np.linspace(0, 1, 101)
R, HR = np.meshgrid(r, hr)
HO = ho_lim_C1(R, HR, os_dict)

fig, ax = plt.subplots(figsize=[6.5, 3])
ax.plot(r_lim_initial(hr, os_dict)*1e3, hr*1e2, ':k')
ax.plot([r_lim_hb(r, os_dict)*1e3]*2, [0, 1e2], 'k--')
cf = ax.contourf(R*1e3, HR*1e2, HO, levels=np.logspace(-3, 0, 21), norm=colors.LogNorm())
cb = ax.contour(cf, r*1e3, hr*1e2, HO, colors='w', levels=np.logspace(-3, 0, 4), norm=colors.LogNorm())
ax.clabel(cb, cb.levels, inline=True, inline_spacing=100, fmt='%.0e')
cbar = fig.colorbar(cf)
cbar.ax.set_ylabel('Maximum penetration depth (m)')
ax.text(1.1e-2, 50, '$H_b = H_i$ \n & r < $r_{c}$\n(A)')
ax.text(4e-2, 75, '$H_b = H_i$ \n & r $\geq r_{c}$\n (C3)')
ax.text(1.1e-1, 2.5, '$H_b < H_i$ \n & r < $r_{c} (B)$')
ax.text(1.1, 95, '$h_o + H_b > H_i$ (C2)')
ax.text(1.1, 50, '(C1)')
ax.set_ylim([0, 1*1e2])
ax.set_xlim([Rmin*1e3, Rmax*1e3])
ax.set_xscale('log')
ax.set_xlabel('Pore radius (mm)')
ax.set_ylabel('Oil lens thickness (cm)')
plt.tight_layout()
fig_fn = os.path.join(fig_dir, 'fig_XX-analytical.jpg')
plt.savefig(fig_fn, dpi=500)
pdf_fn = os.path.join(fig_dir, 'pdf', 'fig_XX-analytical.pdf')
plt.savefig(pdf_fn)
plt.show()
