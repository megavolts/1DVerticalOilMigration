#!/usr/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt  # Hunter J., 2007
import pandas as pd  # McKinney, 2010
import yaml
import os
import pickle

import matplotlib as mpl
mpl.use('Qt5Agg')

try:
    from project_helper import *
except ImportError:
    import sys
    if '/home/megavolts/git/1DVerticalOilMigration' in sys.path:
        raise
    sys.path.append('/home/megavolts/git/1DVerticalOilMigration')
    from project_helper import *
try:
    from numerical_helper import *
except ImportError:
    import sys
    if '/home/megavolts/git/1DVerticalOilMigration' in sys.path:
        raise
    sys.path.append('/home/megavolts/git/1DVerticalOilMigration')
    from numerical_helper import *

# Variable definition
data_dir = '/mnt/data/UAF-data/paper/4/'
output_dir = '/home/megavolts/Desktop/output'
case_fn = 'oil_spill_case/thick_ice.ini'

# Discretization
N_layers = 1000

# CASE
N_case = case_fn.split('/')[1].split('.ini')[0]  # 2: thin ice; 1: thick ice, 20: thin ice, late winter
run_n = 1
LOAD = False
OVERRIDE = False
UNCORRECTED_PS = False
HIGH_RES = False
TS = -20  # ice surface temperature
DEBUG = True

# t_step modification flag
t_step = 0.1  # Fallback t_step: large because flow is small
# TODO change True to Enable or Disable
R = True  # True to enable, if float
dL = 0.05  # False to disable QBLOSS, brine loss through channel wall
SI_PHYSIC = True  # False to use mean salinity, temperature; 'override' to use salinity and temperature from mat_dict
SI_STRAT = True  # False to disable, distance to granular/columnar layer. Negative from surface. Positive from bottom
TORT = True  # False to disable; 'override' to use tortuosity from mat_idct
SEASONNAL_EVOLUTION = False  # Crude schematic of seasonnal evolution
END_CHANNEL = 0.05  # False allows brine to surface, or float: distance from ice surface where the channel ends
# False


# Default
Nfig = 100
tau = 1
tau_g = 2.0
tau_c = 1.05
T_si = -5  # °C, initial sea-ice temperature
S_si = 5  # ‰, initial sea-ice bulk salinity
epsilon = 1e-12

MU_CST = False
RHO_CST = False
OIL_CST = False
BRINE_CST = False

# Initial condition
S_b = 60  # ‰, initial brine salinity
S_sw = 32  # ‰, initial sea-water salinity

# Load boundary condition
case_fp = case_fn
bc_dict = load_case(case_fp)

# Compute material properties according to physical properties and temperature
T_sw = float(pysic.property.sw.freezingtemp(S_sw)[0])
RHO_o_R = rho_oil(T_sw)
RHO_b = float(pysic.property.brine.density(T_si)[0])
RHO_sw = float(pysic.property.sw.density_p0(S_sw, T_sw)[0])

run_fn = str('run-%s-%04i-N%i_ts%.1f' % (N_case, run_n, N_layers, t_step))

run_fp = os.path.join(output_dir, run_fn)
pkl_fp = os.path.join(run_fp, run_fn + '-porsepace.pkl')

TS_season = False
if LOAD and (os.path.exists(run_fp) and os.path.exists(pkl_fp)):
    print('Loading existing data')
    with open(pkl_fp, 'rb') as f:
        os_dict, porespace, porespace_corr, TS_season = pickle.load(f)
    # create new file name
    if not OVERRIDE:
        run_n += 1
        run_fn = str('run-%s-%04i-N%i_ts%.1f' % (N_case, run_n, N_layers, t_step))
        run_fp = os.path.join(output_dir, run_fn)
        while os.path.exists(run_fp) and os.path.exists(pkl_fp):
            run_n += 1
            run_fn = str('run-%s-%04i-N%i_ts%.1f' % (N_case, run_n, N_layers, t_step))
            run_fp = os.path.join(output_dir, run_fn)
            pkl_fp = os.path.join(run_fp, run_fn + '-porsepace.pkl')
        if not os.path.exists(run_fp):
            os.makedirs(run_fp)
    print(run_fp, run_fn)

    # Update os_dict with latest bc_dict
    os_dict['BC'] = bc_dict

    pkl_fp = os.path.join(run_fp, run_fn + '-porsepace.pkl')
    with open(pkl_fp, 'wb') as f:
        pickle.dump([os_dict, porespace, porespace_corr, TS_season], f)
    # Backup config
    dict_df = run_fn + '-config.yaml'
    ps_fp = os.path.join(run_fp, dict_df)
    with open(ps_fp, 'w+') as f:
        yaml.dump(os_dict, f)

else:
    # Data output directory
    run_n = 1
    run_fn = str('run-%s-%04i-N%i_ts%.1f' % (N_case, run_n, N_layers, t_step))
    run_fp = os.path.join(output_dir, run_fn)
    ps_fn = run_fn+'-porespace.jpg'
    ps_fs = os.path.join(run_fp, ps_fn)
    if not OVERRIDE:
        while os.path.exists(run_fp) and os.path.exists(ps_fs):
            run_n += 1
            run_fn = str('run-%s-%04i-N%i_ts%.1f' % (N_case, run_n, N_layers, t_step))
            run_fp = os.path.join(output_dir, run_fn)
            ps_fn = run_fn + '-porespace.jpg'
            ps_fs = os.path.join(run_fp, ps_fn)
        if not os.path.exists(run_fp):
            os.makedirs(run_fp)
    print(run_fp, run_fn)

    ## Create pore space
    h_array = np.linspace(0, bc_dict['HI'], N_layers+1)

    if HIGH_RES:
        Hlim = 0.8
        N_layers_hr = N_layers
        h_array = h_array[h_array <= Hlim*bc_dict['HI']]
        h_array = np.linspace(0, bc_dict['HI']*Hlim, N_layers + 1)
        h_array = np.concatenate((h_array[:-1], np.linspace(max(h_array), bc_dict['HI'], N_layers+1)))
    N_layers = len(h_array)-1

    porespace = pd.DataFrame(h_array[:-1], columns=['h'])
    porespace['h_mid'] = h_array[:-1] + np.diff(h_array)/2
    porespace['dh'] = np.diff(h_array)

#    y_mid = np.arange(0, bc_dict['HI'], 0.05) + 0.025

    # Prescribe C-shape salinity profile
    s_profile, t_profile, TS_season = lookup_TS_si_cover(TS, bc_dict['HI'])
    y_profile = np.linspace(0, bc_dict['HI'], 21)
    y_profile = np.diff(y_profile)/2 + y_profile[:-1]

    # interpolate salinity and temperaure profile for porespace
    porespace['salinity'] = nearest_interp(porespace.h, y_profile, s_profile)[::-1]
    porespace['temperature'] = np.interp(porespace.h_mid, y_profile, t_profile)[::-1]

    # Create material dictionary
    os_dict = {'mat': {'sw': {'T': T_sw, 'S': S_sw},
                       'o': {'Tsw': T_sw, 'gamma': GAMMA_o, 'theta': THETA_o},
                       'b': {'S': S_b, 'gamma': GAMMA_b, 'theta': THETA_b},
                       'si': {'S': S_si, 'T': T_si, 'tau_c': tau_c, 'tau_g': tau_g, 'tau': tau}},
               'cst': {'MU_CST': MU_CST, 'OIL_CST': OIL_CST, 'RHO_CST': RHO_CST, 'BRINE_CST': BRINE_CST},
               'config': {'t_step': t_step, 'N_layer': N_layers, 'HighResLayer': HIGH_RES,
                         'dL': dL, 'R': R, 'TORT': TORT, 'SI_PHYSIC': SI_PHYSIC, 'SI_STRAT': SI_STRAT},
               'BC': bc_dict}

    porespace = generate_porespace_geometry(porespace, os_dict)
    porespace = generate_porespace_physics(porespace, os_dict)
    porespace['r0'] = porespace['r']
    porespace = correct_porespace_radius(porespace.copy())

    HD = ice_draft_nm(porespace, os_dict)[0]
    os_dict['BC']['HD'] = HD

    # Add surface layer for plotting:
    porespace = porespace.append(pd.DataFrame([np.nan * np.ones_like(porespace.iloc[0])], columns=porespace.columns))
    porespace['h'].iloc[-1] = bc_dict['HI']
    porespace['V'].iloc[-1] = 1
    porespace['dh'].iloc[-1] = porespace.iloc[-2]['dh']
    porespace['r'].iloc[-1] = porespace['r'].iloc[-2]
    porespace = porespace.reset_index(drop=True)

    pkl_fp = os.path.join(run_fp, run_fn + '-porsepace.pkl')
    with open(pkl_fp, 'wb') as f:
        pickle.dump([os_dict, porespace, TS_season], f)
    # Backup config
    dict_df = run_fn + '-config.yaml'
    ps_fp = os.path.join(run_fp, dict_df)
    with open(ps_fp, 'w+') as f:
        yaml.dump(os_dict, f)

if UNCORRECTED_PS:
    porespace['r'] = porespace['r0']
#
# # PLot porespace and properties condition
# ax = plot_porespace(porespace)
# plt.suptitle(str('N$_l$=%s, dt=%.2f' % (N_layers, t_step)))
# ps_fn = run_fn+'-porespace.jpg'
# ps_fp = os.path.join(run_fp, ps_fn)
# plt.savefig(ps_fp, dpi=300)  # Save figures

# Plot properties
plot_ps_data = porespace.copy()
ax = plot_properties(plot_ps_data)
ps_fn = run_fn+'-prop.jpg'
ps_fp = os.path.join(run_fp, ps_fn)
plt.savefig(ps_fp, dpi=300)  # Save figures

# PLot porespace (both corrected (r) and uncorrected (r0)
plot_ps_data = porespace.copy()
ax = plot_ps(plot_ps_data)
ps_fn = run_fn+'-ps_r.jpg'
ps_fp = os.path.join(run_fp, ps_fn)
plt.savefig(ps_fp, dpi=300)  # Save figures

plot_ps_data = porespace.copy()
plot_ps_data['r'] = plot_ps_data['r0']
ax = plot_ps(plot_ps_data)
ps_fn = run_fn+'-ps_r0.jpg'
ps_fp = os.path.join(run_fp, ps_fn)
plt.savefig(ps_fp, dpi=300)  # Save figures

## INITIALIZED
ps_df = porespace
ps_df['p_si'] = RHO_sw * g * (os_dict['BC']['HD']) - porespace['h'].apply(lambda x: RHO_sw * x * g if RHO_sw * x * g > 0 else 0)
HB, ii_hb, vf_bc = brine_hyraulic_head_dm(porespace, os_dict)  # height of brine in chann
HD = os_dict['BC']['HD']


t = 0  # TODO replace by t by t_elapsed
t_elpased = t
day_elpased = 0
ii_ho = 0
ii_t = 0
ho = 0  # oil penetration depth
qo = 0  # oil flow in the channel
hr = os_dict['BC']['HR']  # oil lens thickness
hb = HB  # height of brine in channel
vr = os_dict['BC']['VR']  # total volume of oil in lens
vo = 0  # total volume of oil in the brine channel
vb_channel = porespace['V'][:ii_hb].sum() + porespace['V'][ii_hb]*vf_bc  # total volume of brine in the channel
vb_loss = 0 # total volume of brine expelled from channel
vb_surface = 0  # total volume of brine at the surface
vb_surface_t =0 # total volume of brine at the surface at step t
dvo = 0  # volume of oil moving up at step t
vb_t = 0  # volume of brine move through ii_hb at step t
vb_loss_t = 0  # volume of brine loss at step t
vb_surface_t = 0  # volume of oil moving at the surface at step t
delta_p = 0
delta_t_ho = 0
qb_loss = 0

vb_channel_0 = vb_channel
vb_current = vb_channel
STOP = False

# Pressure data
data_p_headers = ['p0', 'pc_b', 'pho', 'phb', 'pc_o', 'p_qloss', 'delta_p', 'pbs', 'poc', 'poR', 'pbc', 't', 'ho']
data_p = []

# Pressure at the top of the channel
r_hb = ps_df.loc[(ps_df.h <= ho+HB), 'r'].iloc[-1]
p_hb = -pc_b(r_hb)

# Pressure at the bottom of the channel
r_0 = ps_df.r.iloc[0]
p_0 = p0(hr, os_dict)

# Store initial condition
data_bc_headers = ['ii', 't', 'dt', 'ho', 'hb', 'vo', 'vb_channel', 'vb_loss', 'vb_loss_t', 'qo', 'qb_loss',
                   'hr', 'vr', 'dvo', 'vb_t', 'vb_surface', 'vb_surface_t', 'p_0', 'p_hb', 'delta_p']
data_bc = [[ii_t, t, delta_t_ho, ho, hb, vo, vb_channel, vb_loss, vb_loss_t, qo, qb_loss, hr,
            vr, dvo, vb_t, vb_surface, vb_surface_t, p_0, p_hb, delta_p]]
# Flow data
data_q_headers = ['qo', 'qb', 'qob', 't', 'ho']
data_q = []

# Phase
data_c_headers = ['phase', 'mu_i_o', 'mu_i_b', 'rho_i_o', 'rho_i_b', 'p_qloss', 'Q_loss_i', 'q_bloss_b_i', 'pc']
data_c = np.zeros([1, 9, N_layers+1])
data_c[ii_t, 0, :ii_hb] = -1  # -1: brine, 1: oil
if ii_hb < N_layers:
    data_c[ii_t, 0, ii_hb] = -vf_bc  # Volume fraction of brine in upper cell
data_c[ii_t, 0, ii_hb + 1:] = 0  # air

HI = os_dict['BC']['HI']
HR = os_dict['BC']['HR']
HG = os_dict['BC']['HG']
VR = os_dict['BC']['VR']

# If brine is not able to surface, then there is an impermeable layer
if END_CHANNEL:
    if isinstance(END_CHANNEL, (int, float)):
        h_end_channel = HI - END_CHANNEL
        h_end_channel = ps_df.loc[ps_df.h < h_end_channel, 'h'].max()
        N_end_channel = ps_df.loc[ps_df.h == h_end_channel].index[0] + 1
    else:
        h_end_channel = HI
        h_end_channel = ps_df.loc[ps_df.h < h_end_channel, 'h'].max()
        N_end_channel = ps_df.loc[ps_df.h == h_end_channel].index[0]
else:
    N_end_channel = N_layers + 2

while not STOP: # and ii_ho < 11:
    vb_surface_t = 0
    t_elpased = t
    if SEASONNAL_EVOLUTION and TS_season:
        start_date = TS_season.date.min()
        current_date = start_date + pd.to_timedelta(t_elpased, 'second')
        if current_date - start_date > pd.to_timedelta(day_elpased + 1, 'day'):
            day_elpased += 1

            s_profile, t_profile = extract_TS_profile(TS_season, current_date, porespace.h)
            porespace['salinity'] = s_profile
            porespace['temperature'] = np.concatenate([t_profile, [np.nan]])

            # compute new physic profile
            porespace = generate_porespace_physics(porespace, os_dict)

    # Update volume fraction
    ps_df['f_b'] = fraction_cell(data_c[ii_t, 0, :], 'b')  # -
    ps_df['f_o'] = fraction_cell(data_c[ii_t, 0, :], 'o')  # -

    # oil pressure loss (8 mu_o / [pi r**4] dh)_i
    data_c[ii_t, 1, :-1] = ps_df[['mu_o', 'r', 'l', 'f_o']].apply(lambda x: q_mu_i_f(x), axis=1)[:-1]  # Pa s m-3

    # brine pressure loss  (8 mu_b / [pi r**4] dh)_i
    data_c[ii_t, 2, :-1] = ps_df[['mu_b', 'r', 'l', 'f_b']].apply(lambda x: q_mu_i_f(x), axis=1)[:-1]  # Pa s m-3

    # oil hydrostatic pressure (rho_o g dh)_i
    data_c[ii_t, 3, :-1] = ps_df[['rho_o', 'dh', 'f_o']].apply(lambda x: p_rho_i_f(x), axis=1)[:-1]  # kg m-1 s-2 = Pa

    # brine hydrostatic pressure (rho_b g dh)_i
    data_c[ii_t, 4, :-1] = ps_df[['rho_b', 'dh', 'f_b']].apply(lambda x: p_rho_i_f(x), axis=1)[:-1]  # kg m-1 s-2 = Pa

    # Pressure at the channel bottom z=0
    p_0 = p0(hr, os_dict)  # Pa

    # Capillary pressure at the air/brine interface ii_hb z = h_o + h_b
    r_hb = ps_df['r'].loc[ii_hb]
    pc_b_hb = -pc_b(r_hb)  # Pa

    # Capillary pressure at the brine/oil interface ii_hb z = h_o + h_b
    r_ho = ps_df['r'].iloc[ii_ho]
    pco = pc_o(r_ho)  # Pa

    # Pressure within the channel:
    # 0 <= h <= ho + hb
    if ii_ho > 0:
        data_c[ii_t, 8, :ii_hb + 1] = p_0 - pco - np.cumsum(data_c[ii_t, 3, :ii_hb+1])\
                                    - np.cumsum(data_c[ii_t, 4, :ii_hb+1]) + np.cumsum(data_c[ii_t-1, 5, :ii_hb+1])  # Pa
    else:
        data_c[ii_t, 8, :ii_hb + 1] = p_0 - pco - np.cumsum(data_c[ii_t, 3, :ii_hb + 1])\
                                      - np.cumsum(data_c[ii_t, 4, :ii_hb + 1]) + np.cumsum(data_c[ii_t-1, 5, :ii_hb+1])  # Pa
    # 0 <= h < ho (override)
    data_c[ii_t, 8, :ii_ho] = p_0 - np.cumsum(data_c[ii_t, 3, :ii_ho])  # Pa

    # Q_b_i : [4 (k1+k2) / mu_b * A / dL * (p_c - p_si) ]_i Brine flow out of the channel (m3 / s)
    dp = np.zeros(N_layers)
    dp = data_c[ii_t, 8, :] - ps_df['p_si']  # Pa
    dp[dp < 0] = 0

    # Lateral flow of brine for each cell
    # set to 0 if cell is already occupied by oil
    data_c[ii_t, 7, :] = ps_df[['kh1', 'kh2', 'mu_b', 'r', 'l', 'f_b']][:-1].apply(lambda x: q_brine_i_f(x, dL), axis=1) * dp  # (m3 / s)

    # Cumulative sum of lateral flow of brine: Qloss_i
    # For ho < h < hb
    data_c[ii_t, 6, :ii_hb+1] = np.cumsum(data_c[ii_t, 7, :ii_hb+1])  # (m3 / s)

    if dL:
        # 8 mu_b / (pi * r*4) * l * Qloss_i
        # TODO
        data_c[ii_t, 5, :] = ps_df[['mu_b', 'r', 'l']].apply(lambda x: 8 * x[0] / (np.pi * x[1]**4) * x[2], axis=1)  # Pa s m-3
        data_c[ii_t, 5, :] = data_c[ii_t, 5, :] * data_c[ii_t, 6, :]  # Pa

    # Compute pressure:
    p_qloss = np.sum(data_c[ii_t, 5, :-1])  # Pa
    p_hb = np.sum(data_c[ii_t, 4, :-1])  # Pa
    p_ho = np.sum(data_c[ii_t, 3, :-1])  # Pa

    # ['p0', 'pc_b', 'pho', 'phb', 'pc_o', 'p_qloss', ...]
    _data_p = [p_0, -pc_b_hb, -p_ho, -p_hb, -pco, p_qloss]  # Pa

    delta_p = np.sum(_data_p)  # Pa

    _data_q = [np.sum(data_c[ii_t, 1, :-1]), np.sum(data_c[ii_t, 2, :-1])]  # Pa s m-3
    qob = np.sum(_data_q[:2])  # Pa s m-3

    # Oil volumetric flux at ii_ho
    qo = delta_p / qob  # m3 / s

    # volume of oil to move, accounting for the volume of oil already present in the cell
    dvo = ps_df['V'].iloc[ii_ho]  # m3

    # Pressure gradient is sufficient to trigger flow:
    # if delta_p is < 0, brine cannot move upward.
    if delta_p > 0:
        mode = 'dp'
        # Time required to move oil into element ii_ho
        delta_t = ps_df['V'] / qo  # s
        delta_t_ho = delta_t[ii_ho]  # s

        if dL:
            mode = 'dp / Ql'
            # first move brine laterally, then move brine upward
            # Vertical flow of brine within in each channel cell := flow of oil - cumulative lateral flow of brine
            qb_i = np.zeros(N_layers+1)
            qb_i[ii_ho:ii_hb + 1] = qo - data_c[ii_t, 6, ii_ho:ii_hb + 1]  # (m3 / s)
            if ii_t < 10 or np.mod(ii_t, Nfig) == 0:
                fig, ax = plt.subplots()
                ax.plot(ps_df['p_si'], ps_df['h'], label='$p_{si}$')
                ax.plot(data_c[ii_t, 8, :], ps_df['h'], label='$p_{c}$')
                ax2 = ax.twiny()
                x_lim = ax.get_xlim()
                ax.plot(x_lim, [ps_df['h'].iloc[ii_ho]] * 2, 'k', alpha=0.5, label='$h_o$')
                ax.plot(x_lim, [hb] * 2, 'b', alpha=0.75, label='$h_b$')
                ax.plot(x_lim, [HB] * 2, 'b:', alpha=0.5, label='$H_b^{t_0}$')
                ax.plot(x_lim, [HD] * 2, 'b--', alpha=0.5, label='$H_D$')

                ax2.plot(data_c[ii_t, 7, :], ps_df['h'], 'r:', label='$q_{b,si,i}$')
                ax2.plot([qo, qo], [0, HI], 'k', label='$q_{o}$')
                ax2.plot(data_c[ii_t, 6, :], ps_df['h'], 'r', label='$Q_{loss,i}$')

                ax.set_ylim([0, HI])
                ax.set_xlim([0, HI * RHO_sw * g + (RHO_sw-RHO_o_R) * (hr+0.1) * g])
                ax2.set_xscale('log')
                ax2.set_xlim([1e-12, 1e-5])
                ax.set_ylabel('Ice thickness (m)')
                ax.set_xlabel('Pressure (Pa)')
                ax2.set_xlabel('Flow (m$^3$s$^{-1}$)')
                plt.suptitle(str('ii = %i - N$_{layers}$=%s, dt=%.2f (s) \nH$_R$=%.2f (cm), V$_R$=%.2f (l), R=%.2e (mm)' %
                                 (ii_t, N_layers, t_step, HR*1e2, VR*1e-3, ps_df.r[:1].mean()*1e3)))

                l_, h_ = ax.get_legend_handles_labels()
                l1_, h1_ = ax2.get_legend_handles_labels()
                l_.extend(l1_)
                h_.extend(h1_)
                plt.legend(l_, h_, loc='best', frameon=False)

                ii_t_fn = run_fn + str('-%04i.jpg' % ii_t)
                ps_fp = os.path.join(run_fp, ii_t_fn)
                plt.savefig(ps_fp, dpi=150)
            if DEBUG:
                plt.show()
            else:
                plt.close()

            # Total flow of brine expelled from the channel (lateral flow)
            qb_loss = np.sum(data_c[ii_t, 7, :-1])  # Check qb_si + qb[ii_hb] = qo (m3 / s)

            # If oil flow faster than, the brine, then the oil flow is limited by the brine flow out of the channel
            if qo <= qb_loss:
                qb_loss = qo
                mode = 'dp / Ql limited'

                # All the displaced brine is moved out of the channel, flow limited by oil flow
                # TODO: outside flow need to be connected with brine flow through bulk sea ice matrix
                # TODO: check if there is no necking slowing the brine movement vertically in the channel
                print("All the displaced brine is moved out of the channel in %.02f s" % delta_t_ho)
        else:
            qb_loss = 0

            # All brine is moved upward

        # Total volume of brine displaced through the channel
        #
        # # Check time integrity: there should not be limiting volumetric flow rate
        # q_lim = np.argwhere(delta_t_ho - delta_t.values > epsilon)
        #
        # if len(q_lim) > 0:
        #     print('Limiting flow in the following layers:')
        #     print('\tii\t(dt_ho - dt_ii)$')
        #     for q_ in q_lim:
        #         print(str('\t%i\t%.4e' % (q_, delta_t_ho - delta_t[q_].values)))
        #     break
    elif hb == HI and not END_CHANNEL:
        mode = 'brine surfaced'
        # Time required to move oil into element ii_ho
        delta_t = ps_df['V'] / qo  # s
        delta_t_ho = delta_t[ii_ho]  # s

        if dL:
            # Vertical flow of brine within in each channel cell := flow of oil - cumulative lateral flow of brine
            qb_i = np.zeros(N_layers+1)
            qb_i[ii_ho:ii_hb + 1] = qo - data_c[ii_t, 6, ii_ho:ii_hb + 1]  # (m3 / s)
            if ii_t < 10 or np.mod(ii_t, Nfig) == 0:
                fig, ax = plt.subplots()
                ax.plot(ps_df['p_si'], ps_df['h'], label='$p_{si}$')
                ax.plot(data_c[ii_t, 8, :], ps_df['h'], label='$p_{c}$')
                ax2 = ax.twiny()
                x_lim = ax.get_xlim()
                ax.plot(x_lim, [ps_df['h'].iloc[ii_ho]] * 2, 'k', alpha=0.5, label='$h_o$')
                ax.plot(x_lim, [hb] * 2, 'b', alpha=0.75, label='$h_b$')
                ax.plot(x_lim, [HB] * 2, 'b:', alpha=0.5, label='$H_b^{t_0}$')
                ax.plot(x_lim, [HD] * 2, 'b--', alpha=0.5, label='$H_D$')

                ax2.plot(data_c[ii_t, 7, :], ps_df['h'], 'r:', label='$q_{b,si,i}$')
                ax2.plot([qo, qo], [0, HI], 'k', label='$q_{o}$')
                ax2.plot(data_c[ii_t, 6, :], ps_df['h'], 'r', label='$Q_{loss,i}$')

                ax.set_ylim([0, HI])
                ax.set_xlim([0, HI * RHO_sw * g + (RHO_sw-RHO_o_R) * (hr+0.1) * g])
                ax2.set_xscale('log')
                ax2.set_xlim([1e-12, 1e-5])
                ax.set_ylabel('Ice thickness (m)')
                ax.set_xlabel('Pressure (Pa)')
                ax2.set_xlabel('Flow (m$^3$s$^{-1}$)')
                plt.suptitle(str('ii = %i - N$_{layers}$=%s, dt=%.2f (s) \nH$_R$=%.2f (cm), V$_R$=%.2f (l), R=%.2e (mm)' %
                                 (ii_t, N_layers, t_step, HR*1e2, VR*1e-3, ps_df.r[:1].mean()*1e3)))

                l_, h_ = ax.get_legend_handles_labels()
                l1_, h1_ = ax2.get_legend_handles_labels()
                l_.extend(l1_)
                h_.extend(h1_)
                plt.legend(l_, h_, loc='best', frameon=False)

                ii_t_fn = run_fn + str('-%04i.jpg' % ii_t)
                ps_fp = os.path.join(run_fp, ii_t_fn)
                plt.savefig(ps_fp, dpi=150)
            if DEBUG:
                plt.show()
            else:
                plt.close()

            # Total flow of brine expelled from the channel (lateral flow)
            qb_loss = np.sum(data_c[ii_t, 7, :-1])  # Check qb_si + qb[ii_hb] = qo (m3 / s)
    else:
        if dL:
            # Check for brine displacement:
            qb_loss = np.sum(data_c[ii_t, 7, :-1])  # Check qb_si + qb[ii_hb] = qo
            if qb_loss > 0:
                mode = 'Ql'
                delta_t_b = dvo / qb_loss
                delta_t_ho = t_step

                # volume of oil cannot exceed volume of brine
                dvo = qb_loss * delta_t_ho
                # dvbs = 0
            else:
                # The brine doesn't like to move it, move it
                mode = 'END'
                qb_loss = 0
                delta_t_ho = 0
                vb_loss_t = 0
                dvo = 0
                # dvbs = 0

                STOP = True
        else:
            mode = 'END'

            # no brine is displaced laterally, no brine moves up. It is the end
            qb_loss = 0
            delta_t_ho = 0
            # dvbs = 0
            STOP = True
        qo = qb_loss
        if ii_t < 10 or np.mod(ii_t, Nfig) == 0:
            fig, ax = plt.subplots()
            ax.plot(ps_df['p_si'], ps_df['h'], label='$p_{si}$')
            ax.plot(data_c[ii_t, 8, :], ps_df['h'], label='$p_{c}$')
            ax2 = ax.twiny()
            x_lim = ax.get_xlim()
            ax.plot(x_lim, [ps_df['h'].iloc[ii_ho]] * 2, 'k', alpha=0.5, label='$h_o$')
            ax.plot(x_lim, [hb] * 2, 'b', alpha=0.75, label='$h_b$')
            ax.plot(x_lim, [HB] * 2, 'b:', alpha=0.5, label='$H_b^{t_0}$')
            ax.plot(x_lim, [HD] * 2, 'b--', alpha=0.5, label='$H_D$')

            ax2.plot(data_c[ii_t, 7, :], ps_df['h'], 'r:', label='$q_{b,si,i}$')
            ax2.plot([qo, qo], [0, HI], 'k', label='$q_{o}$')
            ax2.plot(data_c[ii_t, 6, :], ps_df['h'], 'r', label='$Q_{loss,i}$')

            ax.set_ylim([0, HI])
            ax.set_xlim([0, HI * RHO_sw * g + (RHO_o_R - RHO_sw) * (hr + 0.1) * g])
            ax2.set_xscale('log')
            ax2.set_xlim([1e-12, 1e-6])
            ax.set_ylabel('Ice thickness (m)')
            ax.set_xlabel('Pressure (Pa)')
            ax2.set_xlabel('Flow (m$^3$s$^{-1}$)')
            plt.suptitle(str('ii = %i - N$_{layers}$=%s, dt=%.2f (s) \nH$_R$=%.2f (cm), V$_R$=%.2f (l), R=%.2e (mm)' %
                             (ii_t, N_layers, t_step, HR * 1e2, VR * 1e3, ps_df.r[:1].mean() * 1e3)))

            l_, h_ = ax.get_legend_handles_labels()
            l1_, h1_ = ax2.get_legend_handles_labels()
            l_.extend(l1_)
            h_.extend(h1_)
            plt.legend(l_, h_, loc='best', frameon=False)

            ii_t_fn = run_fn + str('-%04i.jpg' % ii_t)
            ps_fp = os.path.join(run_fp, ii_t_fn)
            plt.savefig(ps_fp, dpi=150)
        if DEBUG:
            plt.show()
        else:
            plt.close()

    # Total volume of brine expelled from the channel
    vb_loss_t = qb_loss * delta_t_ho  # m3

    # Second, we move brine upward
    # Volume of brine that is moving up = volume of oil - volume of brine expelled
    vb_t = dvo - vb_loss_t
    vb_t_residue = vb_t

    kk_hb = ii_hb
    if END_CHANNEL is False and hb == HI:
        print("Brine has already reached the ice surface")
        vb_surface_t = vb_t_residue
        vb_t_residue = 0
    elif vb_t_residue > 0:
        if data_c[ii_t, 0, kk_hb] > 0:
            print("ERROR Cannot move brine into a cell partially occupied by oil")
            STOP = True
        if kk_hb >= N_end_channel:
            print('Brine has reached the end of channel')
            hb = h_end_channel
        elif kk_hb < N_layers-1:
            while vb_t_residue > 0 and kk_hb < N_layers and kk_hb < N_end_channel:
                # Check if the cell is already partially filled with brine:
                if data_c[ii_t, 0, kk_hb] <= 0:
                    vb_cell = ps_df['V'][kk_hb] * np.abs(data_c[ii_t, 0, kk_hb])
                else:
                    vb_cell = ps_df['V'][kk_hb] * (1 - data_c[ii_t, 0, kk_hb])

                # Cell fraction occupied by vb_kk
                f_b_kk = (vb_cell + vb_t_residue) / ps_df['V'][kk_hb]
                if f_b_kk > 1:
                    data_c[ii_t, 0, kk_hb] = -1
                    vb_t_residue = vb_t_residue - (ps_df['V'][kk_hb] - vb_cell)
                    kk_hb += 1
                else:
                    data_c[ii_t, 0, kk_hb] = -f_b_kk
                    vb_t_residue = 0

            ii_hb = kk_hb
            if kk_hb == N_end_channel:
                print("Brine has reached the impermeable layer")
                hb = h_end_channel
            elif kk_hb > N_layers - 1:
                print("Brine reached the ice surface")
                vb_surface_t = vb_t_residue
                vb_t_residue = 0
                hb = HI
            else:
                hb = ps_df['h'].iloc[ii_hb] + ps_df['dh'].iloc[ii_hb] * f_b_kk

        else:
            print("Brine has already reached the ice surface 3 ")
            vb_surface_t = vb_t_residue
            vb_t_residue = 0
            hb = HI

    # Third moves oil upward (volume of brine move up and volume of brine expelled
    # move oil into element ii_ho
    # If there is some residual brine to move, then not all oil can move upward
    if vb_t_residue > 0:
        dvo = vb_t - vb_t_residue + vb_loss_t
    if dvo > 0:
        kk_ho = ii_ho
        vo_t_residue = dvo
        if END_CHANNEL and (kk_ho >= N_end_channel or ho > h_end_channel):
            print('Oil has reached the end of channel')
            ho = h_end_channel
            STOP = True
        else:
            while vo_t_residue > 0 and kk_ho < N_layers:
                # Check if the cell is already partially filled with oil
                if data_c[ii_t, 0, kk_ho] >= 0:
                    vo_cell = ps_df['V'][kk_ho] * data_c[ii_t, 0, kk_ho]
                else:
                    vo_cell = ps_df['V'][kk_ho] * (1 - np.abs(data_c[ii_t, 0, kk_ho]))
                f_o_kk = (vo_cell + vo_t_residue) / ps_df['V'][kk_ho]
                if f_o_kk > 1:
                    data_c[ii_t, 0, kk_ho] = 1
                    vo_t_residue = vo_t_residue - (ps_df['V'][kk_ho] - vo_cell)
                    kk_ho += 1
                else:
                    data_c[ii_t, 0, kk_ho] = f_o_kk
                    vo_t_residue = 0

            ho = ps_df['h'].iloc[kk_ho] + ps_df['dh'].iloc[kk_ho] * f_o_kk
            if ho > HI:
                ho = HI
                STOP = True
        if dvo < 0 or vo_t_residue < 0:
            # CHECK
            print("Something is fishy with dvo or dvos")
            STOP = True
        ii_ho = kk_ho
    # Increment time
    t += delta_t_ho

    # Remove oil from the reservoir
    vo += dvo
    hr = HR / VR * (VR - vo)
    vr = HR / VR * hr

    # Move brine around
    vb_surface += vb_surface_t
    vb_loss += vb_loss_t
    vb_current = vb_current - vb_loss_t - vb_surface_t

    if np.abs(vb_channel_0 - vb_current - vb_surface - vb_loss) < epsilon:
        vb_check = 'OK'
    else:
        print('Some brine is disappearing')
        vb_check = 'X'
        STOP = True

    # Compute pressure differential as control variable
    _f = data_c[-1, 0, :-1] - data_c[0, 0, :-1]
    _f = _f[data_c[-1, 0, :-1] - data_c[0, 0, :-1] < 0]
    _fn = np.where(data_c[-1, 0, :-1] - data_c[0, 0, :-1] < 0)[0]
    _r = ps_df.loc[ps_df.index.isin(_fn), ['rho_b', 'dh']]
    _r['f'] = np.abs(_f)

    if ho + HB < HI:
        _pbs = np.sum(_r[['rho_b', 'dh', 'f']].apply(lambda x: p_rho_i_f(x), axis=1).values)
    else:
        # TODO entery real RHO_b value of RHOB
        _pbs = RHO_b * g * ps_df['dh'].iloc[ii_ho]
    _poc = RHO_sw * ho * g - ps_df.loc[ps_df.index <= ii_ho, ['rho_o', 'dh']].apply(lambda x: p_rho_i_f(x), axis=1).sum()
    _poR = (RHO_sw - RHO_o_R) * hr * g
    if ho <= HB:
        _pbc = RHO_sw * ho * g - ps_df.loc[ps_df.index <= ii_ho, ['rho_b', 'dh']].apply(lambda x: p_rho_i_f(x), axis=1).sum()
    else:
        _pbc = 0

    # Expelled brine at the surface:
    if hb > HI:
        hb = HI
    elif ho > HI:
        hb = 0

    # [..., 'delta_p', 'pbs', 'poc', 'poR', 'pbc', 't', 'ho']
    _data_p.extend([delta_p, _pbs, _poc, _poR, _pbc, t, ho])
    _data_q.extend([qob, t, ho])

    if np.mod(ii_t, 20) == 0:
        print('ih\ti_t\t\tt\tdt_ho\tho\t\thb\t\tdp\t\t\tdvb_loss\t\tdvo\t\tdvb\t\tdvbs\t\tvb_check\t\tmode')
    print(str('%i\t%i\t%.2f\t%.2f\t%.4f\t%.4f\t%.2e\t%.3e\t%.3e\t%.3e\t%.3e\t%s\t%s' %
              (ii_ho, ii_t, t, delta_t_ho, ho, hb, delta_p, vb_loss_t, dvo, vb_t, vb_surface_t, vb_check, mode)))
    data_p.append(_data_p)
    data_q.append(_data_q)
    data_bc.append([ii_t, t, delta_t_ho, ho, hb, vo, vb_channel, vb_loss, vb_loss_t, qo, qb_loss, hr, vr, dvo, vb_t, vb_surface,
                    vb_surface_t, p_0, pc_b_hb, delta_p])

    if ho == HI:
        print('Oil has reached the surface, spreading is inevitable')
        STOP = True
    elif all(data_c[ii_t, 7, :-1] == 0) :
        print('p_si > p_c: no brine loss possible and brine cannot move upward')
        STOP = True

    if vb_t < 1e-14 and vb_loss_t < 1e-14 and ii_ho < 100 * ii_t :
        print('Brine loss < 1e-14')
        STOP = True

    # move brine up in next loop otherway
    if ii_ho >= N_layers:
        STOP = True

    if not STOP:
        _data_c = np.array([data_c[-1, :, :]])
        _data_c[0, 0, ii_ho-1] = 1
        data_c = np.concatenate([data_c, _data_c])
        ii_t += 1
        del delta_t_ho, vb_loss_t, qo, qb_loss, dvo, vb_t, vb_surface_t, p_0, pc_b_hb, delta_p, vb_check

    if np.mod(ii_t, 500) == 0 or ii_t > 150:
        data_bc_df = np.array(data_bc)
        data_df = pd.DataFrame(data_bc_df, columns=data_bc_headers)
        data_p_df = pd.DataFrame(data_p, columns=data_p_headers)
        data_q_df = pd.DataFrame(data_q, columns=data_q_headers)
        pkl_fp = os.path.join(run_fp, run_fn + 'results.pkl')
        with open(pkl_fp, 'wb') as f:
            pickle.dump([data_df, ps_df, data_p_df, data_q_df, os_dict, data_c], f)

data_bc_df = np.array(data_bc)
data_df = pd.DataFrame(data_bc_df, columns=data_bc_headers)
data_p_df = pd.DataFrame(data_p, columns=data_p_headers)
data_q_df = pd.DataFrame(data_q, columns=data_q_headers)

print('R\t\tHR\t\tVR\t\tHB\t\tt\t\tho\t\tDp')
print(str('%.2f\t%.2f\t%.1e\t%.4f\t%.2f\t%.4f\t%.0f' % (R, HR, VR, HB, data_df.t.iloc[-1], data_df.ho.iloc[-1], data_df.delta_p.iloc[-1])))
plt.close()
#
# ## Double figure:
# fig = plt.figure(figsize=(8, 6))
# ax = fig.subplots(1, 2)
#
# # Plot 1
# ax[0].plot(data_df.t / 3600, data_df.ho, 'k-', label='$h_o$')
# ax[0].plot(data_df.t / 3600, data_df.hb, 'b', label='$h_b+h_o$')
# ax[0].plot(data_df.t / 3600, [HI - HB] * len(data_df), label='$H_i - H_b$')
# ax[0].plot(data_df.t / 3600, [HI] * len(data_df), 'k', alpha=0.5, label='$H_i$')
# ax[0].plot(data_df.t / 3600, [HD] * len(data_df), 'b:', alpha=0.5, label='$H_D$')
# ax0 = ax[0].twinx()
# ax0.plot(data_df.t / 3600, data_df.qo, 'r', label='$Q_{o}$')
# if dL:
#     ax0.plot(data_df.t / 3600, data_df.qb_loss, ':', c='orange', label='$Q_{b,loss}$')
# ax0.spines['right'].set_color('r')
#
# ax[0].fill_between(data_df.t.astype(float) / 3600, [-HR - 0.1] * len(data_df), -data_df.hr.astype(float), color='b',
#                    alpha=0.5)  # ocean underneath
# ax[0].fill_between(data_df.t.astype(float) / 3600, -data_df.hr.astype(float), [0] * len(data_df), color='k')  # oil lens
# ax[0].plot(data_df.t / 3600, -data_df.hr, color='k')
#
# t_max = data_df.loc[data_df.ho <= data_df.ho.max(skipna=True), 't'].max(skipna=True) / 3600
# t_min = data_df.loc[data_df.t >= 0, 't'].min(skipna=True) / 3600
# ax[0].set_xlim([t_min, t_max])
#
# _x_text = data_df.t.max()
# ax[0].set_ylabel('penetration depth (m)')
# ax[0].set_xlabel('Time (h)')
# ax[0].set_ylim([-HR - 0.1, HI + 0.1])
# ax[0].text(data_df.t.max() / 3600 * 0.02, 0, 'Oil', color='w',
#            horizontalalignment='left', verticalalignment='top')
# ax[0].text(data_df.t.max() / 3600 / 2, -HR - 0.1, 'Sea water', color='w',
#            horizontalalignment='center', verticalalignment='bottom')
# ax[0].text(data_df.t.max() / 3600 * .98, HI / 2, 'Ice cover', color='k',
#            horizontalalignment='right', verticalalignment='center')
# ax[0].text(data_df.t.max() / 3600 * 0.98, 0.75 * HI, str('h$_o$=%.02f cm' % (data_df.ho.max() * 100)),
#            horizontalalignment='right', verticalalignment='center')
#
# ax0.set_yscale('log')
# ax0.set_ylabel('Flow (m$^3$s$^{-1}$)', c='red')
#
# y_lim = ax0.get_ylim()
# y_lim_min = 10 ** np.floor(np.log10(min(y_lim)))
# y_lim_max = 10 ** np.ceil(np.log10(max(y_lim)))
#
# if np.ceil(np.log10(max(y_lim))) < np.floor(np.log10(min(y_lim))) + 3:
#     y_lim_max = 10 ** np.floor(np.log10(min(y_lim) + 3))
#
# if y_lim_max > 1e-6:
#     y_lim_max = 1e-6
#
# ax0.set_ylim([y_lim_min, y_lim_max])
#
# ax[1].plot(data_p_df.ho, data_p_df.delta_p, 'k', label='$\Delta p_{tot}$')
# ax[1].plot(data_p_df.ho, data_p_df.poc + data_p_df.poR, label='$p_o$')  # large
# ax[1].plot(data_p_df.ho, data_p_df.pbc, label='$p_{o\\rightarrow b,c}$')
# ax[1].plot(data_p_df.ho,
#            data_p_df.poc + data_p_df.poR - data_p_df.pc_o + data_p_df.pc_b + data_p_df.p_qloss + data_p_df.pbc,
#            label='$[p+]$')
#
# ax1_label = '$\Delta p_{tot}$, $p_o$, $p+$'
# ax2_label = ' $p_{c,o}$, $p_{c,b}$'
#
# ax1 = ax[1].twinx()
# if max(data_p_df.pbs) > 200:
#     ax[1].plot(data_p_df.ho, data_p_df.pbs, 'r', label='$p_b|_{h> H_b}$ [p-]')  # Small or Large
#     ax1_label = ax1_label + ', $p_b|_{h> H_b}$ [p-]'
# else:
#     ax1.plot(data_p_df.ho, data_p_df.pbs, 'r:', label='$p_b|_{h> H_b}$ [p-]')  # Small or Large
#     ax2_label = ax2_label + ', $p_b|_{h> H_b}$ [p-]'
#
# ax1.plot(data_p_df.ho, data_p_df.pc_b, ':', label='$p(h_b)$')  # Small
# ax1.plot(data_p_df.ho, -data_p_df.pc_o, ':', label='-$p_{c,o}$')  # Small
#
# if dL:
#     if max(data_p_df.p_qloss) > 200:
#         ax[1].plot(data_p_df.ho, data_p_df.p_qloss, label='$p_{q_{b,loss}}$')
#         ax1_label = ax1_label + ', $p_{q_{b,loss}}$'
#     else:
#         ax1.plot(data_p_df.ho, data_p_df.p_qloss, ':', label='$p_{q_{b,loss}}$')
#         ax2_label = ax2_label + ', $p_{q_{b,loss}}$'
#
# y_lim = ax[1].get_ylim()
# ax[1].set_ylim()
# ax[1].set_xlim([0, data_p_df.ho.max()])
#
# if HI - HB < ax[1].get_xlim()[-1]:
#     ax[1].plot([HI - HB] * 2, y_lim, 'k--', alpha=0.5, label='$H_i - H_b$')
# if HB < ax[1].get_xlim()[-1]:
#     ax[1].plot([HB] * 2, y_lim, 'k-.', alpha=0.5, label='$H_D$')
# if SI_STRAT:
#     if HI - HG < ax[1].get_xlim()[-1]:
#         ax[1].plot([HI - HG] * 2, y_lim, 'k:', alpha=0.5, label='$H_C$')
#
# ax[1].set_xlabel('Penetration depth (m)')
# ax[1].set_ylabel('Pressure (Pa): ' + ax1_label)
# ax1.set_ylabel('Pressure (Pa): ' + ax2_label)
#
# # Legend
# l_, h_ = ax[0].get_legend_handles_labels()
# l0_, h0_ = ax0.get_legend_handles_labels()
# l_.extend(l0_)
# h_.extend(h0_)
# l1_, h1_ = ax[1].get_legend_handles_labels()
# l2_, h2_ = ax1.get_legend_handles_labels()
# l1_.extend(l2_)
# h1_.extend(h2_)
# ax[0].legend(l_, h_, loc='upper center', ncol=2, labelspacing=0.2, columnspacing=1,
#              bbox_to_anchor=(0.5, -0.1), frameon=False)
# ax[1].legend(l1_, h1_, loc='upper center', ncol=2, labelspacing=0.2, columnspacing=1,
#              bbox_to_anchor=(0.5, -0.1), frameon=False)
#
# fig.subplots_adjust(bottom=0, top=1)
# ho_fn = run_fn + '-ho.jpg'
# ps_fp = os.path.join(run_fp, ho_fn)
#
#
# plt.suptitle(str(
#     '$\overline{R}$=%.1e, H$_R$=%.2f, V$_R$=%.2e, $\Delta$L=%.2f, $\\tau$: %s, TS:%s\n N$_{layer}$=%i, dt=%.2f ' %
#     (ps_df.r[:-1].mean(), HR, VR, dL, str(TORT)[0], str(TORT)[0], N_layers, t_step)))
# plt.tight_layout()
#
# ps_fp = os.path.join(run_fp, run_fn + 'results.jpg')
# plt.savefig(ps_fp, dpi=300)  # Save figures
# plt.show()

# Final report figure:

## Single figure:
fig = plt.figure(figsize=(4, 4))
ax = fig.subplots(1, 1)
ax = [ax]
# Plot 1
ax[0].plot(data_df.t / 3600, data_df.ho, 'k-', label='$h_o$')
ax[0].plot(data_df.t / 3600, data_df.hb, 'b', label='$h_b$')
ax[0].plot(data_df.t / 3600, [HI] * len(data_df), 'k', alpha=0.5, label='$H_i$')
ax[0].plot(data_df.t / 3600, [HD] * len(data_df), 'b:', alpha=0.5, label='$H_D$')
ax0 = ax[0].twinx()
ax0.plot(data_df.t / 3600, data_df.qo, 'r', label='$Q_{o}$')
if dL:
    ax0.plot(data_df.t / 3600, data_df.qb_loss, ':', c='orange', label='$Q_{b,loss}$')
ax0.spines['right'].set_color('r')

ax[0].fill_between(data_df.t.astype(float) / 3600, [-HR - 0.1] * len(data_df), -data_df.hr.astype(float), color='b',
                   alpha=0.5)  # ocean underneath
ax[0].fill_between(data_df.t.astype(float) / 3600, -data_df.hr.astype(float), [0] * len(data_df), color='k')  # oil lens
ax[0].plot(data_df.t / 3600, -data_df.hr, color='k')

t_max = data_df.loc[data_df.ho <= data_df.ho.max(skipna=True), 't'].max(skipna=True) / 3600
t_min = data_df.loc[data_df.t >= 0, 't'].min(skipna=True) / 3600
ax[0].set_xlim([t_min, t_max])
_x_text = data_df.t.max()
ax[0].set_ylabel('penetration depth (m)')
ax[0].set_xlabel('Time (h)')
ax[0].set_ylim([-HR - 0.1, HI + 0.1])

ylim = ax[0].get_ylim()

ax[0].text(data_df.t.max() / 3600 * 0.05, -HR/2, 'Oil', color='w',
           horizontalalignment='left', verticalalignment='center')
ax[0].text(data_df.t.max() / 3600 * 0.05, -HR + (ylim[0] + HR)/2, 'Sea water', color='w',
           horizontalalignment='left', verticalalignment='center')
ax[0].text(data_df.t.max() / 3600 * .95, HI / 2, 'Ice cover', color='k',
           horizontalalignment='right', verticalalignment='center')
ax[0].text(data_df.t.max() / 3600 * 0.95, 0.75 * HI, str('h$_o$=%.02f cm' % (data_df.ho.max() * 100)),
           horizontalalignment='right', verticalalignment='center')

ax0.set_yscale('log')
ax0.set_ylabel('Flow (m$^3$s$^{-1}$)', c='red')

y_lim = ax0.get_ylim()
y_lim_min = 10 ** np.floor(np.log10(min(y_lim)))
y_lim_max = 10 ** np.ceil(np.log10(max(y_lim)))

if np.ceil(np.log10(max(y_lim))) < np.floor(np.log10(min(y_lim))) + 3:
    y_lim_max = 10 ** np.floor(np.log10(min(y_lim) + 3))

if y_lim_max > 1e-6:
    y_lim_max = 1e-6

ax0.set_ylim([y_lim_min, y_lim_max])

# Legend
l_, h_ = ax[0].get_legend_handles_labels()
l0_, h0_ = ax0.get_legend_handles_labels()
l_.extend(l0_)
h_.extend(h0_)
ax[0].legend(l_, h_, loc='upper center', ncol=3, labelspacing=0.2, columnspacing=1,
             bbox_to_anchor=(0.5, -0.15), frameon=False)


fig.subplots_adjust(bottom=0, top=1)
plt.tight_layout()

ps_fp = os.path.join(run_fp, run_fn + 'results-FR.jpg')
plt.savefig(ps_fp, dpi=300)  # Save figures
plt.show()

pkl_fp = os.path.join(run_fp, run_fn + 'config.pkl')
# Backup CONFIG
with open(pkl_fp, 'wb') as f:
    pickle.dump([data_df, ps_df, data_p_df, data_q_df, os_dict, data_c, porespace], f)
print(os.path.join(run_fp, run_fn))
