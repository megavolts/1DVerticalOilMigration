# -*- coding: utf-8 -*-
#

import matplotlib.pyplot as plt  # Hunter J., 2007
from matplotlib.cm import viridis as cmap

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
fig_dir = '/home/megavolts/Desktop'
# Discretization
N_layers = 1000

# Porespace
Rs = [1e-3]  # m

# Oil properties
GAMMA_o = 20e-3  # kg s^-2 or 1e3 dyne/cm, interfacial tension between crude oil and water and ice, Malcom et al., 1979
THETA_o = 180  # deg, contact angel between crude oil and ice, in water Malcom et al., 1979

# Brine properties
GAMMA_b = 75.65e-3  # kg s^-2 interfacial tension between brine and sea ice in air
THETA_b = 0  # deg, contact angle between brine and sea ice in air

# Initial condition
T_i = -5  # °C, initial sea-ice temperature
T_sw = -5  # °C, initial sea-ice temperature
S_si = 5  # ‰, initial sea-ice bulk salinity
S_b = 60  # ‰, initial brine salinity
S_sw = 32  # ‰, initial sea-water salinity

# Load boundary condition
case_fp = case_fn
# case_fp =  os.path.join(data_dir, case_fn)
bc_dict = load_case(case_fp)

# Compute material properties according to physical properties and temperature
mat_dict = {'sw': {'S': S_sw},
            'o': {'Tsw': t_sw_f(S_sw), 'Ti': T_i, 'gamma': GAMMA_o, 'theta': THETA_o},
            'b': {'Ti': T_i, 'S': S_b, 'gamma': GAMMA_b, 'theta': THETA_b}}

HI = bc_dict['HI']
HD = bc_dict['HD']

# Fix
R = 2e-3
HR = 0.1
VR = 1000
Rmin = 1e-5
Rmax = 1e-2

# Create oil spill case dictionnary
os_dict = bc_dict
os_dict['HR'] = HR
os_dict['VR'] = VR
os_dict.update(mat_dict)

# ta, t1, t2, t3, C = oil_penetration_depth(ho, R, os_dict, mat_dict, DEBUG=True)

r = np.logspace(-5, -2, 100)
ho = np.linspace(0, 1, 501)
R, HO = np.meshgrid(r, ho)
R = R.transpose()  # Required to have R as row, and HO as column
HO = HO.transpose()   # Required to have R as row, and HO as column
RHO_sw = rho_seawater(os_dict['sw']['S'], t_sw_f(os_dict['sw']['S']))
RHO_b = rho_brine(os_dict['b']['Ti'])
RHO_o_R = rho_oil(os_dict['o']['Tsw'])  # oil density in sea water
RHO_o_c = rho_oil(os_dict['o']['Ti'])  # oil density in sea ice
MU_b = mu_brine(os_dict['b']['S'], os_dict['b']['Ti'])
MU_o = mu_oil(os_dict['o']['Ti'])

hb = brine_hydraulic_head(r, HI, HD, RHO_sw, RHO_b)
t, t1, t2, t3, C = oil_penetration_depth(HO, R, os_dict, DEBUG=True)  # row of t is t(ho) for given r

ho_max = [ho[ii] for ii in np.argmax(t, axis=1)]

fig, ax = plt.subplots()
ax.plot(r, [HI]*len(r), 'k', alpha=0.5, label='Ice surface')
ax.plot(r, [HD]*len(r), 'k:', alpha=0.5, label='Ice draft')
ax.plot(r, hb, label='Brine hydraulic head')
ax.set_xscale('log')

# plot on y-axis:
ax1 = ax.twinx()
ax1.plot(r, ho_max, 'r', label='Max penetration depth')
ax.set_ylim(0, 1.05)
ax.set_ylim(0, 1.05)

fig.legend()
plt.show()

# plot on y-axis:
r = np.logspace(-5, -2, 100)
HD = os_dict['HD']

def b1(r):
    hb = brine_hydraulic_head(r, HI, HD, RHO_sw, RHO_b)
    b1 = RHO_sw * g * r * HD + (RHO_sw - RHO_o_R) * g * r * HR - pc_o(1) + pc_b(1) - RHO_b * g * r * hb
    return b1
def c1(r):
    c1 = np.pi * (RHO_sw - RHO_o_R) * g * r ** 3 * HR / VR + RHO_o_c * g * r
    return c1
def a1(r):
    hb = brine_hydraulic_head(r, HI, HD, RHO_sw, RHO_b)
    a1 = MU_b / MU_o * hb
    return a1

fig, ax = plt.subplots()
ax.plot(r, a1(r), c=cmap(0), label='a1')
ax.plot(r, b1(r), c=cmap(0.8), label='b1')
ax.set_xscale('log')
ax1 = ax.twinx()
ax1.plot(r, c1(r), c=cmap(0.4), label='c1')
ax.plot(r, b1(r) - c1(r) * r, c='k', label='b1 - c1 r')
ax.plot(r, b1(r) / (b1(r) - c1(r) * r), 'k:', label='b1 / (b1 - c1 r)')
ax.plot(r, [0] * r, c='k', alpha=0.5, label='y=0')
ax.set_ylabel('a1, b1, b1 -p.pi * (RHO_sw - RHO_o_R) * g * r ** 3 * HR / VR + RHO_o_c * g * r c1 r, b1 / (b1 - c1 r)')
ax1.set_ylabel('c1')
ax1.set_xlabel('pore radius')
ax.set_ylim([-2, 2])
l_, h_ = ax.get_legend_handles_labels()
l1_, h1_ = ax1.get_legend_handles_labels()
l_.extend(l1_)
h_.extend(h1_)
plt.legend(l_, h_, loc='best', ncol=2)
plt.title('Case 1 - coefficient')
import os
plt.savefig(os.path.join(fig_dir, 'case1.jpg'))
plt.show()


def b2(r):
    b2 = RHO_sw * g * r * HD - RHO_b * g * r * hi + (RHO_sw - RHO_o_R) * g * r * HR - pc_o(1)
    return b2


def c2(r):
    c2 = np.pi * (RHO_sw - RHO_o_R) * g * r ** 3 * HR / VR - (RHO_b - RHO_o_c) * g * r
    return c2


def a2(r):
    a2 = MU_b / (MU_o - MU_b) * hi
    return a2

hi = HI*np.ones_like(r)

fig, ax = plt.subplots()
ax.plot(r, a2(r), c=cmap(0.9), label='a2')
ax.plot(r, b2(r), 'x-', c=cmap(0.7), label='b2')
ax.set_xscale('log')
ax1 = ax.twinx()
ax1.plot(r, c2(r), c=cmap(0.3), label='c2')
ax.plot(r, b2(r) - c2(r) * r, c='k', label='b2 - c2 r')
ax.plot(r, b2(r) / (b2(r) - c2(r) * r), 'k:', label='b2 / (b2 - c2 r)')
ax.plot(r, [0] * r, c='k', alpha=0.5, label='y=0')
ax.set_ylabel('a2, b2, b2 - c2 r, b2 / (b2 - c2 r)')
ax1.set_ylabel('c2')
ax1.set_xlabel('pore radius')
ax.set_ylim([-2, 2])
l_, h_ = ax.get_legend_handles_labels()
l1_, h1_ = ax1.get_legend_handles_labels()
l_.extend(l1_)
h_.extend(h1_)
plt.legend(l_, h_, loc='best', ncol=2)
plt.title('Case 2 - coefficient')
import os
plt.savefig(os.path.join(fig_dir, 'case2.jpg'))
plt.show()

ho = np.arange(0, 1, 0.01)
r = 2e-3
fig, ax = plt.subplots()
ax.plot(ho + (hi-hb), b2(r) + c2(r)*ho, 'x-', c=cmap(0.7), label='b2 - c2 h0')
ax1 = ax.twinx()
ax.plot(ho + (hi-hb), b2(r) / (b2(r) + c2(r) * ho), c='k', label='b2 / (b2 - c2 ho)')
ax1.set_xlabel('ho')
l_, h_ = ax.get_legend_handles_labels()
l1_, h1_ = ax1.get_legend_handles_labels()
l_.extend(l1_)
h_.extend(h1_)
plt.legend(l_, h_, loc='best', ncol=2)
plt.title('Case 2 - coefficient')
import os
plt.savefig(os.path.join(fig_dir, 'case3.jpg'))
plt.show()
