#!/usr/bin/python3
# -*- coding: utf-8 -*-
#

import matplotlib.pyplot as plt  # Hunter J., 2007
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import pandas as pd  # McKinney, 2010

import yaml
import os

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
fig_dir = '/home/megavolts/UAF/paper/Chapter4/figures/model/'
case_fn = 'oil_spill_case/Test-1_10_10.ini'
cmap = cm.viridis

# Discretization
N_layers = 5000

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
HI = bc_dict['HI']
HD = bc_dict['HD']
R = 2e-3
HR = 0.1
VR = 1000
# Compute material properties according to physical properties and temperature
mat_dict = {'sw': {'S': S_sw},
            'o': {'Tsw': t_sw_f(S_sw), 'Ti': T_i, 'gamma': GAMMA_o, 'theta': THETA_o},
            'b': {'Ti': T_i, 'S': S_b, 'gamma': GAMMA_b, 'theta': THETA_b}}


# Create oil spill case dictionnary
os_dict = bc_dict
os_dict['HR'] = HR
os_dict['VR'] = VR
os_dict['R'] = R
os_dict['N'] = N_layers
os_dict.update(mat_dict)


def oil_spill_case(os_dict, save=True):
    """
    :param os_dict: dictionary
        Dictionary containing the oil spill parameters
    :param save: boolean
        if True save the data
    :return:
    """
    r = os_dict['R']
    hi = os_dict['HI']
    hd = os_dict['HD']
    n_layers = os_dict['N']

    hr = os_dict['HR']
    vr = os_dict['VR']

    rho_sw = rho_seawater(os_dict['sw']['S'], t_sw_f(os_dict['sw']['S']))
    rho_b = rho_brine(os_dict['b']['Ti'])
    rho_o_R = rho_oil(os_dict['o']['Tsw'])  # oil density in sea water
    rho_o_c = rho_oil(os_dict['o']['Ti'])  # oil density in sea ice

    hb = brine_hydraulic_head(r, hi, hd, rho_sw, rho_b)
    ho = np.linspace(0, hi, n_layers+1)
    t, tc1, tc2, tc3, t_coef = oil_penetration_depth(ho, r, os_dict, DEBUG=True)

    # Initialize condition
    v_o = 0  # amount of oil on the surface
    v_s = 0  # amount of brine on the surface
    h_R = -hr  # thickness of the oil lens
    h_co = 0  # z at the bottom of the oil lens, check if there is enough oil in the reservoir
    H_co = 0  # Maximal heigh to the oil pocke, after the reservoir is empty
    v_sb = 0  # amount of brine on the surface
    pbs = 0                          # brine weight above H_b
    poR = (rho_sw - rho_o_R) * hr * g   # buoyancy of oil in lens
    poc = 0                          # buoyancy of oil in channel
    pco = pc_o(r)
    pcb = pc_b(r)
    pbc = 0
    data_headers = ['t', 'h_o', 'h_R', 'h_co', 'v_o', 'v_s', 'v_sb', 't_f1', 't_f2', 't_f3', 'poc', 'poR', 'pbs', 'pbc', 'pco', 'pcb', ]
    data = [[t[0], ho[0], h_R, h_co, v_o, v_s, v_sb, 0, 0, 0, poc, poR, pbs, pbc, pco, pcb]]
    dh = np.diff(ho)

    for ii_h, h in enumerate(ho[1:]):
        # if np.isnan(t[ii_h]):
        #     break
        # Thin the oil lens as needed:
        if data[ii_h][2] < 0:
            dv = np.pi * r ** 2 * dh[ii_h]
            v_o += dv
            h_R = - hr/vr * (vr - v_o)
            if h_R > 0:
                h_R = 0
                v_o = vr
            H_co = h
        else:
            h_co = h - H_co

        # Compute oil volume and brine leaving the channel
        if h > HI:
            v_s += dv
        if h + hb > HI:
            v_sb += dv

        poc = (rho_sw - rho_o_c) * h * g
        poR = (rho_sw - rho_o_R) * (-h_R) * g
        pbc = (rho_sw - rho_b) * h * g
        pbs = rho_b * h * g
        pco = pc_o(r)
        pcb = pc_b(r)

        if hd - h < 0:
            pbc = 0

        if HI - hb < h < HI:  # Case C1
            pbs = rho_b * dh[ii_h] * g
            pco = pc_o(r)
            pcb = 0
        elif HI <= h:  # Case C2
            pbs = 0
            pco = 0
            pcb = 0
            poc = (rho_sw - rho_o_c) * hi * g

        # Weight of rise of brine > Buoyancy in the columns

        #if Eb > (Eo+ER):
        #     break
        # if t[ii_h] < 0:
        #     break
        data.append([t[ii_h+1], h, h_R, h_co, v_o, v_s, v_sb, tc1[ii_h], tc2[ii_h], tc3[ii_h], poc, poR, pbs, pbc, pco, pcb])
        # if pressure_equilibrium(ho[ii_h], r, os_dict) < 0:
        #      break
    data_df = pd.DataFrame(data, columns=data_headers)

    if save:
        fp = os.path.join(data_dir, str('OS-R_%.0f-HR_%.0f-V_%.1e.csv' % (r*1e6, hr*1e3, vr)))
        data_df.to_csv(fp)
        yaml_fp = os.path.join(data_dir, str('OS-R_%.0f-HR_%.0f-V_%.1e.yaml' % (r*1e6, hr*1e3, vr)))
        with open(yaml_fp, 'w') as f:
            os_dict['Hb'] = hb
            yaml.dump(os_dict.update(mat_dict), f, default_flow_style=False)
    return data_df, os_dict


def double_fig(data_df, os_dict, ax=None, opt=2, legend=False, display=True):
    Hb = os_dict['Hb']
    HR = os_dict['HR']
    R  = os_dict['R']
    VR = os_dict['VR']

    if ax is None:
        fig, ax = plt.subplots(1, opt)

    if opt == 2 and len(ax) == 1:
        opt = 1

    # Plot 1
    ax[0].plot(data_df.t/3600, [HI] * len(data_df), 'k', alpha=0.5, label='$H_i$')
    ax[0].plot(data_df.t/3600, [Hb] * len(data_df), 'k-.', alpha=0.5, label='$H_B$')
    ax[0].plot(data_df.t/3600, [HD] * len(data_df), 'k:', alpha=0.5, label='H_D')
    ax[0].plot(data_df.t/3600, [HI-Hb] * len(data_df), 'k--', alpha=0.5, label='H$_i$ - H$_b$')

    ax[0].plot(data_df.t/3600, data_df.h_o, 'k-', label='h$_o$')
    # ax[0].plot(data_df.t_f1/3600, data_df.h_o, ':', label=str('f1, h$_o$ < H$_i$ - H$_b$'))
    # ax[0].plot(data_df.t_f2/3600, data_df.h_o, ':', label=str('f2, h$_o$ + H$_b$ > H$_i$'))

    _h = data_df[data_df.h_R == data_df.h_R.max()].index.min()
    ax[0].fill_between(data_df[data_df.index <= _h].t/3600,
                                 data_df[data_df.index <= _h].h_R,
                                 [0] * len(data_df[data_df.index <= _h]), color='k',
                           alpha=0.3)
    ax[0].fill_between(data_df.t/3600, [-HR-0.1] * len(data_df), data_df['h_R'], color='b', alpha=0.3)

    #ax[0].plot(data_df[data_df.index >= _h].t/3600, [0] * len(data_df[data_df.index >= _h]), 'k', alpha=0.5, label='ice bottom')

    t_max = data_df.loc[data_df.h_o <= data_df.h_o.max(skipna=True), 't'].max(skipna=True)/3600
    t_min = data_df.loc[data_df.t >= 0, 't'].min(skipna=True)/3600


    ax[0].set_xlim([t_min, t_max])


    _x_text = data_df.t.max()
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_xlabel('Time (h)')
    ax[0].set_ylim([-HR - 0.1, HI + 0.1])
    ax[0].text(data_df.t.max() / 3600 /2, 0.01, 'Oil', color='k',
                         horizontalalignment='center', verticalalignment='top')
    ax[0].text(data_df.t.max() / 3600 / 2, -HR-0.1, 'Sea water', color='k',
                         horizontalalignment='center', verticalalignment='bottom')
    ax[0].text(data_df.t.max() / 3600 * .98, HI/2, 'Ice cover', color='k',
                         horizontalalignment='right', verticalalignment='center')

    l_, h_ = ax[0].get_legend_handles_labels()


    if opt == 2:
#        ax[0].set_title(str('R=%.1e m' % R))
#        ax[1].set_title(str('H$_R$= %.2f m, V=%.2e l' % (HR, VR)))
        # ax[1].plot(data_df.h_o, data_df.pbs, c=cmap(0.8), label='Brine overhead')
        # ax[1].plot(data_df.h_o, data_df.pbc, c=cmap(0.8), ls=':', label='Brine channel')
        ax[1].plot(data_df.h_o, data_df.poR + data_df.poc, 'k--', label='Oil driving pressure')
        ax[1].plot(data_df.h_o, data_df.poR + data_df.poc + data_df.pcb, 'r', label='Driving pressure')
        #ax[1].plot(data_df.h_o, -data_df.pbc, '--', c=cmap(0.8), label='Brine driving pressure')
        #
        # ax[1].plot(data_df.h_o, data_df.poR + data_df.poc + data_df.pcb, 'k', label='Driving pressure')
        ax[1].plot(data_df.h_o, data_df.pbs + data_df.pco, c=cmap(0.5), label='Retarding pressure')

        # ax[1].plot(data_df.h_o, data_df.poc, 'k:', label='oil channel')

        ax[1].set_xlabel('Depth (m)')
        ax[1].set_ylabel('Presssure differential (Pa)')
        ax[1].set_xlim([0, data_df.loc[data_df.t.notna(), 'h_o'].max()+0.01])
        l2_, h2_ = ax[1].get_legend_handles_labels()
    else:
        ax[0].set_title(str('R=%.0e m, H$_R$= %.2f m, V=%.2e l' % (R, HR, VR)))

    if legend:
        ax[0].legend(l_, h_, loc='lower center', ncol=8, labelspacing=0.2, columnspacing=1,
                   bbox_to_anchor=(0, 0, 1, 1),
                   fancybox=False, shadow=False, frameon=False)
        if opt == 2:
            ax[1].legend(l2_, h2_, loc='lower center', ncol=8, labelspacing=0.2, columnspacing=1,
                        bbox_to_anchor=(0, 0, 1, 1),
                        fancybox=False, shadow=False, frameon=False)
        plt.subplots_adjust(bottom=0.2)
    if display:
        plt.show()

    return ax

# Case number
PLOT_N = 2
# os_CI = [[1e-3, 0.10, 100],
#          [9.0e-4, 0.10, 100],
#          [8.99e-4, 0.10, 100],
#          [0.5e-4, 0.10, 100],
#          [3.0695e-5, 0.10, 100],
#          [3.069e-5, 0.10, 100]]
# os_CI = [[1e-2, 0.10, 1e6],
#          [1e-3, 0.10, 1e6],
#          [2e-4, 0.10, 1e6],
#          [5e-5, 0.10, 1e6],
#          [1e-5, 0.10, 1e6],
#          [1e-6, 0.10, 1e6]]
# os_CI = [[1e-2, 0.10, 3e-4],
#          [1e-2, 0.01, 3e-4],
#          [2e-4, 0.10, 1e-6],
#          [5e-5, 0.10, 1e-6],
#          [1e-5, 0.10, 1e-6],
#          [1e-6, 0.10, 1e-6]]
os_CI = [[2e-4, 0.90, 1e3],
         [1e-3, 0.90, 1e3],
         [5e-3, 0.90, 1e3],
         [2e-4, 0.80, 1e3],
         [1e-3, 0.80, 1e3],
         [5e-3, 0.80, 1e3],
         [2e-4, 0.40, 1e3],
         [1e-3, 0.40, 1e3],
         [5e-3, 0.40, 1e3],
         [2e-4, 0.02, 1e3],
         [1e-3, 0.02, 1e3],
         [5e-3, 0.02, 1e3]]
#os_CI = [[2e-4, 0.10, 100]]
os_CI = [[2e-3, 0.10, 1e3],
         [2e-3, 0.70, 1e3],
         [2e-3, 0.90, 1e3],
         [2e-4, 0.90, 1e3],
         [2e-3, 0.90, 1e-4],
         [2e-3, 0.90, 1e-6]]
# os_CI = [[2e-3, 0.70, 1e3],
#          [2e-3, 0.90, 1e3],
#          [2e-3, 0.90, 1e-4],
#          [2e-3, 0.90, 1e-6]]

# os_CI = [[2e-3, 0.10, 1e3]]
n_col = int(np.floor(np.sqrt(len(os_CI)))) * PLOT_N
if n_col > 6:
    n_col = 6
n_row = int(np.ceil(len(os_CI)*PLOT_N/n_col))

# TODO: split figure over several plot if n_row > 3
fig = plt.figure(figsize=[11, 8.5])
gs1 = gridspec.GridSpec(n_row, n_col, height_ratios=[1]*n_row, width_ratios=[1]*n_col)
ax = [[fig.add_subplot(gs1[0, 0])]]
ax = np.array(ax)
for ii in range(0, n_col - 1):
    ax = np.hstack((ax, [[fig.add_subplot(gs1[0, ii+1])]]))
for ii in range(0, n_row - 1):
    ax_row = np.atleast_2d([])
    for jj in range(0, n_col):
        ax_row = np.hstack((ax_row, [[fig.add_subplot(gs1[ii+1, jj])]]))
    ax = np.vstack((ax, ax_row))

print('R\tHR\tVR\tHb\tt_max\th_o')
alpha_numeric = 'abcdefghijklmno'

for ii_ci, ci in enumerate(os_CI):
    ax_r = int(np.floor(ii_ci*PLOT_N / n_col))
    ax_c = (ii_ci*PLOT_N - ax_r * n_col)
    os_dict = bc_dict.copy()
    os_dict['N'] = N_layers
    os_dict['HR'] = ci[1]
    os_dict['VR'] = ci[2]
    os_dict['R'] = ci[0]
    data_df, os_dict = oil_spill_case(os_dict, mat_dict)

    r_lim = r_lim_C1(os_dict['HR'], 0, os_dict)
    text = ''
    if os_dict['R'] < r_lim:
        text += 'r<r$_c$'
    if os_dict['Hb'] == os_dict['HI']:
        if text != '':
            text += '\n'
        text += 'H$_b$=H$_i$'
    if text != '':
        ax[ax_r, ax_c].text(0, 0.75*os_dict['HI'], text)

    ax[ax_r, ax_c].text(0, 0.75*os_dict['HI'], str('h$_o$=%.01f cm' % (data_df.loc[data_df.t.notna(), 'h_o'].max()*100)))
    xlim = ax[ax_r, ax_c].get_xlim()
    print(ii_ci, alpha_numeric[2*ii_ci])
    # ax[ax_r, ax_c].text(-0.05, 1.05, '(' + alpha_numeric[2*ii_ci] + ')')

    if PLOT_N == 2:
        ax[ax_r, ax_c:ax_c+2] = double_fig(data_df, os_dict, ax=ax[ax_r, ax_c:ax_c+2], display=False)
        # ax[ax_r, ax_c].text(-np.diff(xlim) * 0.01, 1.05, '(' + alpha_numeric[2*ii_ci+1] + ')')

        #ax[ax_r, ax_c+1].set_xlim(0, 0.25)
        #ax[ax_r, ax_c + 1].set_ylim(0, 25)
    else:
        ax[ax_r, ax_c:ax_c + 1] = double_fig(data_df, os_dict, ax=ax[ax_r], opt=1, display=False)
    print(os_dict['R'], os_dict['HR'], os_dict['VR'], os_dict['Hb'], data_df.t.max(), data_df.loc[data_df.t.notna(), 'h_o'].max())

# Legend
l_, h_ = ax[0, 0].get_legend_handles_labels()
if PLOT_N == 2:
    l2_, h2_ = ax[0, 1].get_legend_handles_labels()
    l_.extend(l2_)
    h_.extend(h2_)

gs1.tight_layout(fig, rect=[0, 0.05, 1, 1])
fig.legend(l_, h_, loc='lower center', ncol=8, labelspacing=0.2, columnspacing=1,
           bbox_to_anchor=(0, 0, 1, 1),
           fancybox=False, shadow=False, frameon=False)
fig_fn = os.path.join(fig_dir, str('OS-R_%.0f-HR_%.0f-V_%.1e.jpg' % (ci[0]*1e6, ci[1]*1e3, ci[2])))
plt.savefig(fig_fn, dpi=500)
pdf_fn = os.path.join(fig_dir, 'pdf', str('OS-R_%.0f-HR_%.0f-V_%.1e.pdf' % (ci[0]*1e6, ci[1]*1e3, ci[2])))
plt.savefig(pdf_fn)
plt.show()
