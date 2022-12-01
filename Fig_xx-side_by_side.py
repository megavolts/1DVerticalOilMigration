#!/usr/bin/python3
# -*- coding: utf-8 -*-
# WORKING WITH NO LOSS, RADIUS, TORTUOSITY
import matplotlib.pyplot as plt  # Hunter J., 2007
import pandas as pd  # McKinney, 2010
import yaml
import os
import pickle
import matplotlib.gridspec as gridspec
from matplotlib.cm import viridis as cmap

try:
    from project_helper import *
except ImportError:
    import sys
    if '/home/megavolts/git/SimpleOilModel' in sys.path:
        raise
    sys.path.append('/home/megavolts/git/SimpleOilModel')
    from project_helper import *
try:
    from numerical_helper import *
except ImportError:
    import sys
    if '/home/megavolts/git/SimpleOilModel' in sys.path:
        raise
    sys.path.append('/home/megavolts/git/SimpleOilModel')
    from numerical_helper import *

# Variable definition
data_dir = '/mnt/data/UAF-data/paper/4/'
fig_dir = '/home/megavolts/UAF/paper/Chapter4/figures/model/'

case_list = [25, 28, 46, 49]

n_row = 1
n_col = len(case_list)

fig = plt.figure(figsize=[7, 4])
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

ax0 = ax.copy()

for ii, run_N in enumerate(case_list):
    print(ii)
    # Import data
    N_layers = 1000
    t_step = 0.1
    DEV = False
    if DEV:
        run_fn = str('run_dev-%04i-N%i_ts%.1f' % (run_N, N_layers, t_step))
    else:
        run_fn = str('run_%04i-N%i_ts%.1f' % (run_N, N_layers, t_step))

    run_fd = os.path.join(data_dir, str('run-%i' %N_layers))
    run_fp = os.path.join(run_fd, run_fn)
    pickle_fp = os.path.join(run_fd, run_fn, run_fn + '.pkl')

    if os.path.exists(run_fp) and os.path.exists(pickle_fp):
        with open(pickle_fp, 'rb') as f:
            data_df, ps_df, data_p_df, data_q_df, os_dict = pickle.load(f)
        print(pickle_fp)
    HI = os_dict['BC']['HI']
    HB = data_df.hb.min()
    HD = os_dict['BC']['HD']
    HR = os_dict['BC']['HR']
    HG = os_dict['BC']['HG']
    VR = os_dict['BC']['VR']
    dL = os_dict['config']['dL']
    TORT = os_dict['config']['TORT']
    SI_PHYSIC = os_dict['config']['SI_PHYSIC']

    # Plot 1
    ax[0, ii].plot(data_df.t / 3600, data_df.ho, 'k', label='$h_o$')
    ax0[0, ii] = ax[0, ii].twinx()
    ax0[0, ii].plot(data_df.t / 3600, data_df.qo, c=cmap(0.6), label='$Q_{o}$')
    if os_dict['config']['dL']:
        ax0[0, ii].plot(data_df.t / 3600, data_df.qb_loss, '--', c=cmap(0.8), label='$Q_{b,loss}$')

    ax[0, ii].plot(data_df.t / 3600, data_df.hb, c=cmap(0.3), label='$h_b+h_o$')
    ax[0, ii].plot(data_df.t / 3600, [HI] * len(data_df), 'k', alpha=0.5, label='$H_i$')
    ax[0, ii].plot(data_df.t / 3600, [HD] * len(data_df), 'k:',  alpha=0.5, label='$H_D$')
    ax[0, ii].plot(data_df.t / 3600, [HI - HB] * len(data_df), 'k--', alpha=0.5, label='$H_i - H_b$')

    ax[0, ii].fill_between(data_df.t.astype(float) / 3600, [-HR - 0.1] * len(data_df), -data_df.hr.astype(float), color='b',
                       alpha=0.3)  # ocean underneath
    ax[0, ii].fill_between(data_df.t.astype(float) / 3600, -data_df.hr.astype(float), [0]*len(data_df), color='k',
                       alpha=0.3)  # oil lens

    t_max = data_df.loc[data_df.ho <= data_df.ho.max(skipna=True), 't'].max(skipna=True) / 3600
    t_min = data_df.loc[data_df.t >= 0, 't'].min(skipna=True) / 3600
    ax[0, ii].set_xlim([t_min, t_max])

    _x_text = data_df.t.max()

    ax[0, ii].set_xlabel('Time (h)')
    ax0[0, ii].spines['right'].set_color(c=cmap(0.6))
    ax0[0, ii].tick_params(axis='y', colors=cmap(0.6))

    if ii == 0:
        ax[0, ii].set_ylabel('Penetration depth (m)')
        ax[0, ii].tick_params(labelright=False)
        ax0[0, ii].tick_params(labelleft=False, left=False)
        ax0[0, ii].tick_params(labelright=False, right=True)
    elif ii == len(case_list)-1:
        ax0[0, ii].set_ylabel('Flow Q (m$^3$s$^{-1}$)', c=cmap(0.6))
        ax0[0, ii].tick_params(labelright=True, right=True)
        ax0[0, ii].tick_params(labelleft=False, left=True)
        ax[0, ii].tick_params(labelright=False, right=False)
        ax[0, ii].tick_params(labelleft=False, left=True)
    else:
        ax[0, ii].tick_params(labelright=False, right=False)
        ax[0, ii].tick_params(labelleft=False, left=True)
        ax0[0, ii].tick_params(labelright=False, right=True)
        ax0[0, ii].tick_params(labelleft=False, left=False)

    ax[0, ii].text(data_df.t.max() / 3600 / 2, -0.06, 'Oil', color='k',
                   horizontalalignment='center', verticalalignment='center')
    ax[0, ii].text(data_df.t.max() / 3600 / 2, -HR - 0.06, 'Sea water', color='k',
                   horizontalalignment='center', verticalalignment='center')
    ax[0, ii].text(data_df.t.max() / 3600 * .98, HI / 2, 'Ice', color='k',
                   horizontalalignment='right', verticalalignment='center')
    if data_df.ho.max() != HI:
        ax[0, ii].text(data_df.t.max() / 3600 / 2, 0.8 * HI, str('h$_o$=%.01f cm' % (data_df.ho.max() * 100)),
                       horizontalalignment='center', verticalalignment='center')

    ax[0, ii].set_ylim([-1, HI + 0.1])


    ax0[0, ii].set_yscale('log')
    ax0[0, ii].set_ylim([1e-9, 1e-6])
    alphanum = 'abcd'
    ax[0, ii].text(0, 1.15, '('+alphanum[ii]+')')


# # Legend
l_, h_ = ax[0, 0].get_legend_handles_labels()
l0_, h0_ = ax0[0, 0].get_legend_handles_labels()
l_.extend(l0_)
h_.extend(h0_)
plt.legend(l_, h_, loc='lower center', ncol=7, labelspacing=0.2, columnspacing=1,
           bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure,
           frameon=False)
fig.subplots_adjust(bottom=0.2, top=0.95)
ps_fp = os.path.join('/home/megavolts/UAF/paper/Chapter4/figures', 'fig_10')
plt.tight_layout()
plt.savefig(ps_fp, dpi=300)
plt.show()



