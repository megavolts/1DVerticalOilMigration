#!/usr/bin/python3
# -*- coding: utf-8 -*-
#

import matplotlib.pyplot as plt  # Hunter J., 2007
import pandas as pd  # McKinney, 2010
import matplotlib.gridspec as gridspec

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

g = 9.80665  # m s^-2, Earth Gravitational Constant

# Discretization
N_layers = 1000

# Porespace
R = 1  # m, initial radius diameter

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
S_b = 100  # ‰, initial brine salinity
S_sw = 32  # ‰, initial sea-water salinity

# Load boundary condition
# case_fp = case_fn
# # case_fp =  os.path.join(data_dir, case_fn)
# bc_dict = load_case(case_fp)
# HI = bc_dict['HI']  # m, initial ice thickness
# HD = bc_dict['HD']  # m, initial ice draft
# HF = bc_dict['HF']  # m, initial freeboard
# HR = bc_dict['HR']  # m, initial oil lens thickness / reservoir height
# VR = bc_dict['VR']  # m3, initial oil volume release / reservoir volume
HI = 1
HF = 0.1
HD = 0.9
HR = 1.5
VR = 100

def brine_hydraulic_head(r, h_d, gamma, theta, h_i):
    theta_rad = np.deg2rad(theta)
    h_b = RHO_sw / RHO_b * h_d + 2 * gamma * np.cos(theta_rad) / (RHO_b * np.abs(g) * r)
    if h_b > h_i:
        return h_i
    else:
        return h_b

T_sw = np.array([-1.75])  #pysic.property.sw.freezingtemp(S_sw)[0]
T_o = T_sw  # °C, initial oil temperature
MU_b = 1.7e-3  # kg m^-1 s^-1, Vancoppenolle et al. (2005)
MU_o = mu_oil(T_o)[0]
MU_b = 0.0023  # pysic.property.sw.dynamic_viscosity(S_b, T_i, override_s=True, override_t=True)[0]
RHO_o = rho_oil(T_o)[0]
RHO_b = 1069  # pysic.property.brine.density(T_i)[0]
RHO_sw = 1025  # pysic.property.sw.density_p0(S_sw, T_sw)[0]

HRs = [HR, HR/2, HR/10]
Rs = [1e-3]

# Create figure boundary according to HRs and Rs
fig = plt.figure(figsize=[11, 8.5])
gs1 = gridspec.GridSpec(len(Rs), len(HRs), height_ratios=[1]*len(Rs), width_ratios=[1]*len(HRs))
ax = [[fig.add_subplot(gs1[0, 0])]]
ax = np.array(ax)
for ii in range(0, len(HRs) - 1):
    ax = np.hstack((ax, [[fig.add_subplot(gs1[0, ii+1])]]))
for ii in range(0, len(Rs) - 1):
    ax_row = np.atleast_2d([])
    for jj in range(0, len(HRs)):
        ax_row = np.hstack((ax_row, [[fig.add_subplot(gs1[ii+1, jj])]]))
    ax = np.vstack((ax, ax_row))

for ii_r, HR in enumerate(HRs):
    for ii_HR, R in enumerate(Rs):
        H_b = brine_hydraulic_head(R, HD, GAMMA_b, THETA_b, HI)
        def oil_penetration_deph(h_o):
            H_b = brine_hydraulic_head(R, HD, GAMMA_b, THETA_b, HI)
            b = RHO_sw * g * R * HD + (RHO_sw - RHO_o) * g * R * HR - 2 * GAMMA_o * np.cos(np.deg2rad(THETA_o))
            a1 = MU_b / MU_o * H_b
            b1 = b - RHO_b * g * R * H_b
            c1 = np.pi * (RHO_sw - RHO_o) * g * R ** 3 * HR / VR + RHO_o * g * R
            a2 = MU_b / (MU_o - MU_b) * HI
            b2 = b - RHO_b * g * R * HI
            c2 = np.pi * (RHO_sw - RHO_o) * g * R ** 3 * HR / VR + (RHO_o - RHO_b) * g * R

            def f1(h_o):
                t = 8 / R * ((a1 * c1 + b1) * np.log(b1 / (b1 - c1 * h_o)) - c1 * h_o) / c1 ** 2
                return t

            def f2(h_o):
                t = ((a2 * c2 + b2) * np.log(((b2 - c2 * (HI - H_b)) / (b2 - c2 * h_o))) - c2 * h_o + c2 * (HI - H_b))/ c2 ** 2
                t += ((a1 * c1 + b1) * np.log(b1/(b1 - c1 * (HI - H_b))) - c1 * (HI - H_b)) / c1 ** 2
                t = 8 / R * t
                return t

            def f(h_o):
                if h_o == 0:
                    t = 0
                elif h_o <= HI - H_b:
                    t = f1(h_o)
                else:
                    t = f2(h_o)
                return t

            return f(h_o), f1(h_o), f2(h_o)

        Delta_h = HI / N_layers

        t = 0  # ime
        v_o = 0  # amount of oil on the surface
        v_s = 0  # amount of brine on the surface
        h_o = 0  # height of oil
        h_R = -HR  # thickness of the oil lens
        h_co = 0  # z at the bottom of the oil lens, check if there is enough oil in the reservoir
        H_co = 0  # Maximal heigh to the oil pocke, after the reservoir is empty
        v_sb = 0  # amount of brine on the surface
        Eb = 0                        # brine weight above H_b
        ER = (RHO_sw - RHO_o) * HR    # buoyancy of oil in lens
        Ec = 0                        # bupyancy of oil in channel

        data_headers = ['t', 'h_o', 'h_R', 'h_co', 'v_o', 'v_s', 'v_sb', 't_f1', 't_f2', 'Eb', 'ER', 'Ec']
        data = [[t, h_o, h_R, h_co, v_o, v_s, v_sb, 0, 0, Eb, ER, Ec]]
        print('CI', t, h_o, h_R, v_o, v_s)

        for ii in range(0, N_layers):
            h_o += Delta_h
            t, f1_o, f2_o = oil_penetration_deph(h_o)

            # Thin the oil lens as needed:
            if data[-1][2] < 0:
                dv = np.pi * R ** 2 * Delta_h
                v_o += dv
                h_R = - HR/VR * (VR - v_o)
                if h_R > 0:
                    h_R = 0
                    v_o = VR
                H_co = h_o
            else:
                h_co = h_o - H_co
            # Compute oil volume and brine leaving the channel
            if h_o > HI:
                v_s += dv
            if h_o + H_b > HI:
                v_sb += dv

            Ec = (RHO_sw - RHO_o) * h_o
            ER = -(RHO_sw - RHO_o) * h_R
            if h_o + H_b < HI:
                Eb += RHO_b * (h_o - data[-1][1])
            elif h_o > HI:
                Eb = 0
            else:
                Eb = RHO_b * Delta_h
            data.append([t, h_o, h_R, h_co, v_o, v_s, v_sb, f1_o, f2_o, Eb, ER, Ec])

        data_df = pd.DataFrame(data, columns=data_headers)

        # Plot 1
        ax[ii_HR, ii_r].plot(data_df.t/3600, data_df.h_o, 'k-', label='h$_o$')
        ax[ii_HR, ii_r].plot(data_df.t_f1/3600, data_df.h_o, ':', label=str('f1, h$_o$ < H$_i$ - H$_b$'))
        ax[ii_HR, ii_r].plot(data_df.t_f2/3600, data_df.h_o, ':', label=str('f2, h$_o$ + H$_b$ > H$_i$'))
        ax[ii_HR, ii_r].plot(data_df.t/3600, [HI-H_b] * len(data_df), label='H$_i$ - H$_b$')
        ax[ii_HR, ii_r].plot(data_df.t/3600, [HI] * len(data_df), 'k', label='Ice surface')
        ax[ii_HR, ii_r].plot(data_df.t/3600, [HD] * len(data_df), 'b', label='Water level')
        ax[ii_HR, ii_r].plot(data_df.t/3600, [H_b] * len(data_df), 'b:', alpha=0.5, label='brine level')

        ax[ii_HR, ii_r].fill_between(data_df.t/3600, [-0.2] * len(data_df), data_df[['h_R', 'h_co']].max(axis=1), color='b', alpha=0.5)
        _h = data_df[data_df.h_R == data_df.h_R.max()].index.min()
        ax[ii_HR, ii_r].fill_between(data_df[data_df.index <= _h].t/3600,
                                     data_df[data_df.index <= _h].h_R,
                                     [0] * len(data_df[data_df.index <= _h]), color='k')
        ax[ii_HR, ii_r].plot(data_df[data_df.index >= _h].t/3600, [0] * len(data_df[data_df.index >= _h]), 'k', alpha=0.5, label='ice bottom')

        _x_text = data_df.t.max()
        ax[ii_HR, ii_r].set_ylabel('penetration depth (m)')
        ax[ii_HR, ii_r].set_xlabel('time (h)')
        ax[ii_HR, ii_r].set_title(str('H$_R$= %.2f (m), R=%.2e (m)' % (HR, R)))
        ax[ii_HR, ii_r].set_xlim([data_df.t.min()/3600, 0.25])
        ax[ii_HR, ii_r].set_ylim([-0.2, HI+0.1])


        ax[ii_HR, ii_r].text(data_df.t.max() / 3600 * 0.02, 0, 'Oil', color='w',
                             horizontalalignment='left', verticalalignment='top')
        ax[ii_HR, ii_r].text(data_df.t.max() / 3600 / 2, -0.2, 'Sea water', color='w',
                             horizontalalignment='center', verticalalignment='bottom')
        ax[ii_HR, ii_r].text(data_df.t.max() / 3600 * .98, HI/2, 'Ice cover', color='k',
                             horizontalalignment='right', verticalalignment='center')

l_, h_ = ax[ii_HR, ii_r].get_legend_handles_labels()
gs1.tight_layout(fig, rect=[0, 0.1, 1, 1])
fig.legend(l_, h_, loc='lower center', ncol=6, labelspacing=0.2, columnspacing=1,
           bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure,
           fancybox=False, shadow=False, frameon=False)

fig.tight_layout()
plt.show()
