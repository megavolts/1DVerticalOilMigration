#!/usr/bin/python3
# -*- coding: utf-8 -*-
# WORKING WITH NO LOSS, RADIUS, TORTUOSITY
import matplotlib.pyplot as plt  # Hunter J., 2007
import pandas as pd  # McKinney, 2010
import yaml
import os
import pickle

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
case_fn = 'oil_spill_case/Test-1_10_10.ini'


ii_list = [31]

for ii in ii_list:
    run_N = ii

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

        HI = os_dict['BC']['HI']
        HB = data_df.hb.min()
        HD = os_dict['BC']['HD']
        HR = os_dict['BC']['HR']
        HG = os_dict['BC']['HG']
        QBLOSS = os_dict['config']['QBloss']
        TEMP_CST = os_dict['cst']['TEMP_CST']
        VR = os_dict['BC']['VR']
        dL = os_dict['config']['dL']
        TORT = os_dict['cst']['TORT']

        if 'TORT_OVERRIDE' not in os_dict['config']:
            if 'TORT' in os_dict['cst'] and TORT:
                TORT_OVERRIDE = TORT
            else:
                TORT_OVERRIDE = False
        else:
            TORT_OVERRIDE = os_dict['config']['TORT_OVERRIDE']

        ## Double figure:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.subplots(1, 2)

        # Plot 1
        ax[0].plot(data_df.t / 3600, data_df.ho, 'k-', label='$h_o$')
        ax[0].plot(data_df.t / 3600, data_df.hb, 'b', label='$h_b+h_o$')
        ax[0].plot(data_df.t / 3600, [HI - HB] * len(data_df), label='$H_i - H_b$')
        ax[0].plot(data_df.t / 3600, [HI] * len(data_df), 'k', alpha=0.5, label='$H_i$')
        ax[0].plot(data_df.t / 3600, [HD] * len(data_df), 'b:',  alpha=0.5, label='$H_D$')
        ax0 = ax[0].twinx()
        ax0.plot(data_df.t / 3600, data_df.qo, 'r', label='$Q_{o}$')
        if QBLOSS:
            ax0.plot(data_df.t / 3600, data_df.qb_loss, ':', c='orange', label='$Q_{b,loss}$')
        ax0.spines['right'].set_color('r')

        ax[0].fill_between(data_df.t.astype(float) / 3600, [-HR - 0.1] * len(data_df), -data_df.hr.astype(float), color='b',
                           alpha=0.5)  # ocean underneath
        ax[0].fill_between(data_df.t.astype(float) / 3600, -data_df.hr.astype(float) , [0]*len(data_df), color='k')  # oil lens
        ax[0].plot(data_df.t / 3600, -data_df.hr, color='k')


        t_max = data_df.loc[data_df.ho <= data_df.ho.max(skipna=True), 't'].max(skipna=True) / 3600
        t_min = data_df.loc[data_df.t >= 0, 't'].min(skipna=True) / 3600
        ax[0].set_xlim([t_min, t_max])

        _x_text = data_df.t.max()
        ax[0].set_ylabel('penetration depth (m)')
        ax[0].set_xlabel('Time (h)')
        ax[0].set_ylim([-HR - 0.1, HI + 0.1])
        ax[0].text(data_df.t.max() / 3600 * 0.02, 0, 'Oil', color='w',
                   horizontalalignment='left', verticalalignment='top')
        ax[0].text(data_df.t.max() / 3600 / 2, -HR - 0.1, 'Sea water', color='w',
                   horizontalalignment='center', verticalalignment='bottom')
        ax[0].text(data_df.t.max() / 3600 * .98, HI / 2, 'Ice cover', color='k',
                   horizontalalignment='right', verticalalignment='center')
        ax[0].text(data_df.t.max() / 3600 * 0.98, 0.75 * HI, str('h$_o$=%.02f cm' % (data_df.ho.max() * 100)),
                   horizontalalignment='right', verticalalignment='center')

        ax0.set_yscale('log')
        ax0.set_ylabel('Flow (m$^3$s$^{-1}$)', c='red')

        y_lim = ax0.get_ylim()
        y_lim_min = 10**np.floor(np.log10(min(y_lim)))
        y_lim_max = 10**np.ceil(np.log10(max(y_lim)))

        if np.ceil(np.log10(max(y_lim))) < np.floor(np.log10(min(y_lim))) + 3:
            y_lim_max = 10**np.floor(np.log10(min(y_lim)+3))

        if y_lim_max > 1e-6:
            y_lim_max = 1e-6

        ax0.set_ylim([y_lim_min, y_lim_max])

        ax[1].plot(data_p_df.ho, data_p_df.delta_p, 'k', label='$\Delta p_{tot}$')
        ax[1].plot(data_p_df.ho, data_p_df.poc + data_p_df.poR, label='$p_o$')  #large
        ax[1].plot(data_p_df.ho, data_p_df.pbc, label='$p_{o\\rightarrow b,c}$')
        ax[1].plot(data_p_df.ho, data_p_df.poc + data_p_df.poR - data_p_df.pc_o + data_p_df.pc_b + data_p_df.p_qloss + data_p_df.pbc, label='$[p+]$')

        ax1_label = '$\Delta p_{tot}$, $p_o$, $p+$'
        ax2_label = ' $p_{c,o}$, $p_{c,b}$'

        ax1 = ax[1].twinx()
        if max(data_p_df.pbs) > 200:
            ax[1].plot(data_p_df.ho, data_p_df.pbs, 'r', label='$p_b|_{h> H_b}$ [p-]')  # Small or Large
            ax1_label = ax1_label + ', $p_b|_{h> H_b}$ [p-]'
        else:
            ax1.plot(data_p_df.ho, data_p_df.pbs, 'r:', label='$p_b|_{h> H_b}$ [p-]')  # Small or Large
            ax2_label = ax2_label + ', $p_b|_{h> H_b}$ [p-]'

        ax1.plot(data_p_df.ho, data_p_df.pc_b, ':', label='$p(h_b)$')  # Small
        ax1.plot(data_p_df.ho, -data_p_df.pc_o, ':', label='-$p_{c,o}$')  # Small

        if QBLOSS:
            if max(data_p_df.p_qloss) > 200:
                ax[1].plot(data_p_df.ho, data_p_df.p_qloss, label='$p_{q_{b,loss}}$')
                ax1_label = ax1_label + ', $p_{q_{b,loss}}$'
            else:
                ax1.plot(data_p_df.ho, data_p_df.p_qloss, ':', label='$p_{q_{b,loss}}$')
                ax2_label = ax2_label + ', $p_{q_{b,loss}}$'

        y_lim = ax[1].get_ylim()
        ax[1].set_ylim()
        ax[1].set_xlim([0, data_p_df.ho.max()])

        if HI-HB < ax[1].get_xlim()[-1]:
            ax[1].plot([HI-HB]*2, y_lim, 'k--', alpha=0.5, label='$H_i - H_b$')
        if HB < ax[1].get_xlim()[-1]:
            ax[1].plot([HB]*2, y_lim, 'k-.',  alpha=0.5, label='$H_D$')
        if TORT or TORT_OVERRIDE:
            if HI - HG < ax[1].get_xlim()[-1]:
                ax[1].plot([HI-HG] * 2, y_lim, 'k:', alpha=0.5, label='$H_C$')

        ax[1].set_xlabel('Penetration depth (m)')
        ax[1].set_ylabel('Pressure (Pa): '+ ax1_label)
        ax1.set_ylabel('Pressure (Pa): '+ ax2_label)

        # Legend
        l_, h_ = ax[0].get_legend_handles_labels()
        l0_, h0_ = ax0.get_legend_handles_labels()
        l_.extend(l0_)
        h_.extend(h0_)
        l1_, h1_ = ax[1].get_legend_handles_labels()
        l2_, h2_ = ax1.get_legend_handles_labels()
        l1_.extend(l2_)
        h1_.extend(h2_)
        ax[0].legend(l_, h_, loc='upper center', ncol=2, labelspacing=0.2, columnspacing=1,
                     bbox_to_anchor=(0.5, -0.1), frameon=False)
        ax[1].legend(l1_, h1_, loc='upper center', ncol=2, labelspacing=0.2, columnspacing=1,
                     bbox_to_anchor=(0.5, -0.1), frameon=False)

        fig.subplots_adjust(bottom=0, top=1)
        ho_fn = run_fn + '-ho.jpg'
        ps_fp = os.path.join(run_fd, ho_fn)
        if TORT_OVERRIDE:
            plt.suptitle(str('$\overline{R}$=%.1e, H$_R$=%.2f, V$_R$=%.2e, $\Delta$L=%.2f, Q:%i, t: o, TS:%i\n N$_{layer}$=%i, dt=%.2f ' %
                             (ps_df.r[:-1].mean(), HR, VR, dL, QBLOSS, not TEMP_CST, N_layers, t_step)))
        else:
            plt.suptitle(str('$\overline{R}$=%.1e, H$_R$=%.2f, V$_R$=%.2e, $\Delta$L=%.2f, Q:%i, t: %i, TS:%i\n N$_{layer}$=%i, dt=%.2f ' %
                             (ps_df.r[:-1].mean(), HR, VR, dL, QBLOSS, TORT, TEMP_CST, N_layers, t_step)))
        plt.tight_layout()
        plt.savefig(ps_fp, dpi=300)
        plt.show()

        print(ps_fp)

