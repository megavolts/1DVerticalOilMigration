#!/usr/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt  # Hunter J., 2007
import pandas as pd  # McKinney, 2010
import yaml
import os
import pickle

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
fig_dir = '/home/megavolts/Desktop/figures/model/'
case_fn = 'oil_spill_case/Test-1_10_10.ini'

# Discretization
N_layers = 200
t_step = 0.1  # Fallback t_step: large because flow is small

HIGH_RES = False
Nfig = 10

DEBUG = True
epsilon = 1e-12

# TODO change True to Enable or Disable
R = True # True to enable, if float
HR = 0.9
VR = 300
dL = 0.05  # False to disable QBLOSS
SI_PHYSIC = True # False to use mean salinity, temperature; 'override' to use salinity and temperature from mat_dict
T_si = -5  # °C, initial sea-ice temperature
S_si = 5  # ‰, initial sea-ice bulk salinity
SI_STRAT = True  # False to disable, distance to granular/columnar layer. Negative from surface. Positive from bottom
TORT = True  # False to disable; 'override' to use tortuosity from mat_idct
tau = 1
tau_g = 2.0
tau_c = 1.05

MU_CST = False
RHO_CST = False
OIL_CST = False
BRINE_CST = False

# Initial condition

S_b = 60  # ‰, initial brine salinity
S_sw = 32  # ‰, initial sea-water salinity

# Load boundary condition
case_fp = case_fn

# case_fp =  os.path.join(data_dir, case_fn)
bc_dict = load_case(case_fp)
HI = bc_dict['HI']
HG = bc_dict['HG']
bc_dict['HG'] = 0.8
bc_dict['VR'] = 300
# Compute material properties according to physical properties and temperature
T_sw = float(pysic.property.sw.freezingtemp(S_sw)[0])
RHO_o_R = rho_oil(T_sw)
RHO_b = float(pysic.property.brine.density(T_si)[0])
RHO_sw = float(pysic.property.sw.density_p0(S_sw, T_sw)[0])

# Data output directory
run_fd = os.path.join(data_dir, str('run-%i' %N_layers))
run_n = 1
run_name = 'run2_'
run_fn = run_name + str('%04i-N%i_ts%.1f' % (run_n, N_layers, t_step))
run_fp = os.path.join(run_fd, run_fn)
ps_fn = run_fn+'-porespace.jpg'
ps_fs = os.path.join(run_fd, ps_fn)
while os.path.exists(run_fp) and os.path.exists(ps_fs):
    run_n += 1
    run_fn = run_name + str('%04i-N%i_ts%.1f' % (run_n, N_layers, t_step))
    run_fp = os.path.join(run_fd, run_fn)
    ps_fn = run_fn + '-porespace.jpg'
    ps_fs = os.path.join(run_fd, ps_fn)
print(os.path.join(run_fd, run_fn))

if not os.path.exists(os.path.join(run_fd, run_fn)):
    os.makedirs(os.path.join(run_fd, run_fn))

# Import or load salinity and temperature profile as needed
ic_name = ['BRW_CS-20130328', 'BRW_CS-20130331B']
ic_dir = '/mnt/data/UAF-data/seaice/core/'
ic_pkl_fp = os.path.join(data_dir, 'ic_data.csv')

if os.path.exists(ic_pkl_fp):
    s_profile = pd.read_csv(ic_pkl_fp, index_col=0)

for ic_collection in s_profile.name.unique():
    _flag = [False] * len(ic_name)
    for ii_ic, ic in enumerate(ic_name):
        if ic in ic_collection:
            _flag[ii_ic] = True
    if all(_flag) == True:
        s_profile = s_profile.loc[s_profile.name == ic_collection]
        break
    else:
        ic_list = []
        for ic in ic_name:
            place_ic = ic.split('_')[0]
            year_ic = ic.split('-')[1][:4]
            year_subdir = str('%i_' %(int(year_ic)-1)) + year_ic[-2:]
            ic_list.append(os.path.join(ic_dir, place_ic, year_subdir, ic+'.xlsx'))

        ic_dict = pysic.core.import_ic_list(ic_list)

        ic_data = pysic.core.corestack.stack_cores(ic_dict)
        ic_data_b = ic_data.copy()
        ic_data_b = ic_data_b.set_orientation('bottom')

        y_bins = np.arange(0, ic_data.y_sup.max(), 0.05)
        if (np.abs(y_bins[-1] - ic_data.y_sup.max()) < 0.025):
            y_bins[-1] = ic_data.y_sup.max()

        ic_data = ic_data.discretize(y_bins)
        y_bins = np.unique(np.concatenate([ic_data.y_low.dropna(), ic_data.y_sup.dropna()]))
        ic_stat = ic_data.section_stat(groups=['y_mid'], variables=['salinity', 'temperature'])

        ic_data_b = ic_data_b.discretize(y_bins)
        ic_stat_b = ic_data_b.section_stat(groups=['y_mid'], variables=['salinity', 'temperature'])

        # Create composite salinity profile
        l_core = ic_stat.loc[ic_stat.y_sup.notna(), 'y_sup'].max()
        s_profile = ic_stat[['y_low', 'y_mid', 'y_sup', 'salinity_mean', 'v_ref']]
        s_profile = s_profile.rename(columns={'salinity_mean': 'salinity'})

        s_profile_t = s_profile
        s_profile_b = ic_stat_b[['y_low', 'y_mid', 'y_sup', 'salinity_mean', 'v_ref']]
        s_profile_b = s_profile_b.rename(columns={'salinity_mean': 'salinity'})

        index_t = s_profile[s_profile.y_mid <= 2/5 * l_core].index.max()
        index_b = s_profile_b[s_profile_b.y_mid <= 2/5 * l_core].index.max()

        s_mid_t = s_profile.loc[(s_profile.index > index_t) & (s_profile.index <= s_profile.index.max() - index_b), 'salinity']
        s_mid_b = s_profile_b.loc[(s_profile_b.index > index_b) & (s_profile_b.index <= s_profile_b.index.max() - index_t), 'salinity'][::-1]
        if len(s_mid_t) > len(s_mid_b):
            s_mid = s_mid_t[:-len(s_mid_b)]
            s_mid = np.concatenate([s_mid, np.average(np.vstack([s_mid_t[-len(s_mid_b):], s_mid_b]), axis=0)])
        elif len(s_mid_b) > len(s_mid_t):
            s_mid = s_mid_b[:-len(s_mid_t)]
            s_mid = np.concatenate([s_mid, np.average(np.vstack([s_mid_b[-len(s_mid_t):], s_mid_t]), axis=0)])
        else:
            s_mid = np.average(np.vstack([s_mid_b, s_mid_t]), axis=0)

        s_profile.loc[s_profile.index > index_t, 'salinity'] = np.nan
        s_profile.loc[s_profile.index > s_profile.index.max() - index_b, 'salinity'] = s_profile_b.loc[s_profile_b.index < index_b, 'salinity'][::-1].tolist()
        s_profile.loc[(s_profile.index > index_t) & (s_profile.index <= s_profile.index.max() - index_b), 'salinity'] = s_mid

        # Rescale temperature profile to salinity profile
        # TODO rescale temperature profile before statistic
        t_profile = ic_stat[['y_mid', 'temperature_mean']].dropna()
        t_profile = t_profile.rename(columns={'temperature_mean': 'temperature'})
        if ic_stat['temperature_mean'].argmax(skipna=True) != ic_stat.index[ic_stat.y_sup == l_core].to_list()[0]:
            # Extend temperature profile to surface temperature
            if 0 not in t_profile.y_mid:
                _y = t_profile['y_mid'][0:3]
                _t = t_profile['temperature'][0:3]

                # linear fit:
                def linear_fit(x, a, b):
                    return a * x + b
                from scipy.optimize import curve_fit
                dtdy, t0 = curve_fit(lambda x, a, b: a * x + b, _y, _t)[0]

                t_profile = t_profile.append(pd.DataFrame([[0, t0]], columns=['y_mid', 'temperature']))

            # Extend temperature profile at the bottom to T=-1.8
            if np.abs(t_profile.loc[t_profile.y_mid == t_profile.y_mid.max(), 'temperature'].values[0] - 1.8) > 0.2:
                # linear fit at bottom for t=-1.8 (sea water temperature)
                _yb = t_profile['y_mid'][-3:]
                _tb = t_profile['temperature'][-3:]

                dydt, tb = curve_fit(lambda x, a, b: a * x + b, _tb, _yb)[0]
                y_sw = dydt * -1.8 + tb

                t_profile = t_profile.append(pd.DataFrame([[y_sw, -1.8]], columns=['y_mid', 'temperature']))
        t_profile = t_profile.sort_values('y_mid').sort_values(by=['y_mid'])
        t_profile = t_profile.reset_index(drop=True)

        # Rescale t_profile to s_profile
        ratio_t2s = s_profile.y_sup.max() / t_profile.y_mid.max()
        t_profile.y_mid = t_profile.y_mid * ratio_t2s

        # Discretize t_profile to match s_profile
        t_new = np.interp(s_profile.y_mid, t_profile.y_mid, t_profile.temperature)
        s_profile['temperature'] = t_new
        s_profile['name'] = [ic_name]*len(s_profile)

        s_profile.to_csv(ic_pkl_fp)

## Create pore space
h_array = np.linspace(0, HI, N_layers+1)
if HIGH_RES:
    Hlim = 0.8
    N_layers_hr = N_layers
    h_array = h_array[h_array <= Hlim*HI]
    h_array = np.linspace(0, HI*Hlim, N_layers + 1)
    h_array = np.concatenate((h_array[:-1], np.linspace(max(h_array), HI, N_layers+1)))
N_layers = len(h_array)-1

porespace = pd.DataFrame(h_array[:-1], columns=['h'])
porespace['h_mid'] = h_array[:-1] + np.diff(h_array)/2
porespace['dh'] = np.diff(h_array)

# Salinity profile
ratio_s_HI = HI/s_profile.y_sup.max()
porespace['salinity'] = nearest_interp(porespace.h, s_profile.y_mid.values*ratio_s_HI, s_profile.salinity.values)

# Temperature profile
ratio_t_HI = HI/s_profile.y_mid.max()
porespace['temperature'] = np.interp(porespace.h_mid, s_profile.y_mid*ratio_t_HI, s_profile.temperature)

# Orient profile correctly with z=0 at the oil lens interface
if len(s_profile.v_ref.unique()) == 1 and s_profile.v_ref.unique()[0] == 'top':
    porespace['h'] = porespace['h'].max() - porespace['h']
    porespace = porespace.sort_values(by='h').reset_index(drop=True)

# Create material dicitionary
os_dict = {'mat': {'sw': {'T': T_sw, 'S': S_sw},
                   'o': {'Tsw': T_sw, 'gamma': GAMMA_o, 'theta': THETA_o},
                   'b': {'S': S_b, 'gamma': GAMMA_b, 'theta': THETA_b},
                   'si': {'S': S_si, 'T': T_si, 'tau_c': tau_c, 'tau_g': tau_g, 'tau': tau}},
           'cst': {'MU_CST': MU_CST, 'OIL_CST': OIL_CST, 'RHO_CST': RHO_CST, 'BRINE_CST': BRINE_CST},
           'config': {'t_step': t_step, 'N_layer': N_layers, 'HighResLayer': HIGH_RES,
                     'dL': dL, 'R': R, 'TORT': TORT, 'SI_PHYSIC': SI_PHYSIC, 'SI_STRAT': SI_STRAT},
           'BC': bc_dict}

porespace = generate_TS_porespace(porespace, os_dict)

HD = ice_draft_nm(porespace, os_dict)[0]
os_dict['BC']['HD'] = HD

# Plot add surface layer:
porespace = porespace.append(pd.DataFrame([np.nan * np.ones_like(porespace.iloc[0])], columns=porespace.columns))
porespace['h'].iloc[-1] = 1
porespace['dh'].iloc[-1] = porespace.iloc[-2]['dh']
porespace['r'] .iloc[-1] = 1e6
porespace = porespace.reset_index(drop=True)

# PLot initial condition
ax = plot_porespace(porespace)
plt.suptitle(str('N$_l$=%s, dt=%.2f' % (N_layers, t_step)))
ps_fn = run_fn+'-porespace.jpg'
ps_fp = os.path.join(fig_dir, ps_fn)
plt.savefig(ps_fp, dpi=300)  # Save figures
plt.show()  # Display Figures

## INITIALIZED
ps_df = porespace

ps_df['p_si'] = RHO_sw * g * (HD) - porespace['h'].apply(lambda x: RHO_sw * x * g if RHO_sw * x * g >0 else 0)

HB, ii_hb, vf_bc = brine_hyraulic_head_dm(porespace, os_dict)

t = 0
ii_ho = 0
ii_t = 0
ho = 0  # oil penetration depth
qo = 0  # oil flow in the channel
hr = HR  # oil lens thickness
hb = HB  # height of brine in channel
vr = VR  # total volume of oil in lens
vo = 0  # total volume of oil in the brine channel
vb = porespace['V'][:ii_hb].sum() + porespace['V'][ii_hb]*vf_bc  # total volume of brine in the channel
vb_loss = 0 # total volume of brine expelled from channel
vbs = 0  # total volume of brine at the surface
dvo = 0  # volume of oil moving up at step t
dvb = 0  # volume of brine move through ii_hb at step t
dvb_loss = 0  # volume of brine loss at step t
dvbs = 0  # volume of oil moving at the surface at step t
delta_p = 0
delta_t_ho = 0
qb_loss = 0

vb0 = vb
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
data_bc_headers = ['ii', 't', 'dt', 'ho', 'hb', 'vo', 'vb', 'vb_loss', 'dvb_loss', 'qo', 'qb_loss', 'hr', 'vr', 'dvo', 'dvb', 'vbs', 'dvbs', 'p_0', 'p_hb', 'delta_p']
data_bc = [[ii_t, t, delta_t_ho, ho, hb, vo, vb, vb_loss, dvb_loss, qo, qb_loss, hr, vr, dvo, dvb, vbs, dvbs, p_0, p_hb, delta_p]]
# Flow data
data_q_headers = ['qo', 'qb', 'qob', 't', 'ho']
data_q = []

# Phase
data_c_headrs = ['phase', 'mu_i_o', 'mu_i_b', 'rho_i_o', 'rho_i_b', 'p_qloss', 'Q_loss_i', 'q_bloss_b_i', 'pc']

data_c = np.zeros([1, 9, N_layers+1])
data_c[ii_t, 0, :ii_hb] = -1  # -1: brine, 1: oil
if ii_hb < N_layers:
    data_c[ii_t, 0, ii_hb] = -vf_bc  # Volume fraction of brine in upper cell
data_c[ii_t, 0, ii_hb + 1:] = 0

# Backup config
dict_df = run_fn + '-config.yaml'
ps_fp = os.path.join(run_fd, run_fn, dict_df)
with open(ps_fp, 'w+') as f:
    yaml.dump(os_dict, f)
pickle_fp = os.path.join(run_fd, run_fn, run_fn + '.pkl')

N_fig = 5
while not STOP:
    # Update volume fraction
    ps_df['f_b'] = fraction_cell(data_c[ii_t, 0, :], 'b')
    ps_df['f_o'] = fraction_cell(data_c[ii_t, 0, :], 'o')

    #  (8 mu_o / [pi r**4] dh)_i
    data_c[ii_t, 1, :-1] = ps_df[['mu_o', 'r', 'l', 'f_o']].apply(lambda x: q_mu_i_f(x), axis=1)[:-1]

    #  (8 mu_b / [pi r**4] dh)_i
    data_c[ii_t, 2, :-1] = ps_df[['mu_b', 'r', 'l', 'f_b']].apply(lambda x: q_mu_i_f(x), axis=1)[:-1]

    #  (rho_o g dh)_i
    data_c[ii_t, 3, :-1] = ps_df[['rho_o', 'dh', 'f_o']].apply(lambda x: p_rho_i_f(x), axis=1)[:-1]  # Pa

    #  (rho_b g dh)_i
    data_c[ii_t, 4, :-1] = ps_df[['rho_b', 'dh', 'f_b']].apply(lambda x: p_rho_i_f(x), axis=1)[:-1]  # Pa


    # Pressure at the channel bottom z=0
    p_0 = p0(hr, os_dict)

    # Pressure at z = h_o + h_b
    r_hb = ps_df['r'].loc[ii_hb]
    pc_b_hb = -pc_b(r_hb)
    # Pressure at z = h_o
    r_ho = ps_df['r'].iloc[ii_ho]
    pco = pc_o(r_ho)



    # Pressure within the channel:
    # 0 <= h <= ho + hb
    if ii_ho > 0:
        data_c[ii_t, 8, :ii_hb+1] = p_0 - pco - np.cumsum(data_c[ii_t, 3, :ii_hb+1])\
                                    - np.cumsum(data_c[ii_t, 4, :ii_hb+1]) + np.cumsum(data_c[ii_t-1, 5, :ii_hb+1])
    else:
        data_c[ii_t, 8, :ii_hb + 1] = p_0 - pco - np.cumsum(data_c[ii_t, 3, :ii_hb + 1])\
                                      - np.cumsum(data_c[ii_t, 4, :ii_hb + 1]) + np.cumsum(data_c[ii_t-1, 5, :ii_hb+1])
    # 0 <= h < ho (override)
    data_c[ii_t, 8, :ii_ho] = p_0 - np.cumsum(data_c[ii_t, 3, :ii_ho])

    # Q_b_i : [4 (k1+k2) / mu_b * (p_c - p_si) / dL ]_i Brine flow out of the channel (m3 / s)
    dp = np.zeros(N_layers)
    dp = data_c[ii_t, 8, :] - ps_df['p_si']
    dp[dp < 0] = 0

    data_c[ii_t, 7, :] = ps_df[['kh1', 'kh2', 'mu_b', 'r', 'l', 'f_b']][:-1].apply(lambda x: q_brine_i_f(x, dL), axis=1) * dp  # (m3 / s)

    # Qloss_i
    # ho < h < hb
    data_c[ii_t, 6, :ii_hb+1] = np.cumsum(data_c[ii_t, 7, :ii_hb+1])

    if dL:
        # 8 mu_b / (pi * r*4) * l * Qloss_i
        data_c[ii_t, 5, :] = ps_df[['mu_b', 'r', 'l']].apply(lambda x: 8 * x[0] / (np.pi * x[1]**4) * x[2], axis=1)
        data_c[ii_t, 5, :] = data_c[ii_t, 5, :] * data_c[ii_t, 6, :]


    # Compute volumetric flow rate:
    p_qloss = np.sum(data_c[ii_t, 5, :-1])
    p_hb = np.sum(data_c[ii_t, 4, :-1])
    p_ho = np.sum(data_c[ii_t, 3, :-1])

    # ['p0', 'pc_b', 'pho', 'phb', 'pc_o', 'p_qloss', ...]
    _data_p = [p_0, -pc_b_hb, -p_ho, -p_hb, -pco, p_qloss]


    delta_p = np.sum(_data_p)

    _data_q = [np.sum(data_c[ii_t, 1, :-1]), np.sum(data_c[ii_t, 2, :-1])]
    qob = np.sum(_data_q[:2])

    # Oil volumetric flux at ii_ho
    qo = delta_p / qob  # m3 / s

    # Pressure gradient is sufficient to trigger flow:
    if delta_p > 0:
        mode = 'dp'

        # Oil volumetric flow
        u_ho = qo / (np.pi * r_ho ** 2)  # m / s

        # Time required to move oil into element ii_ho
        delta_t = ps_df['V'] / qo
        delta_t_ho = delta_t[ii_ho]

        # Amount of oil moved
        dvo = ps_df['V'].iloc[ii_ho] - cell_volume(ps_df[['r', 'l', 'f_o']].iloc[ii_ho])
        if dvo < 0:
            print("DVO something is fishy")

        if dL:
            mode = 'dp / Ql'
            # Vertical flow of brine within channel
            qb_i = np.zeros(N_layers+1)
            qb_i[ii_ho:ii_hb + 1] = qo - data_c[ii_t, 6, ii_ho:ii_hb + 1]
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
                ps_fp = os.path.join(run_fd, run_fn, ii_t_fn)
                plt.savefig(ps_fp, dpi=150)
            if DEBUG:
                plt.show()
            else:
                plt.close()
            # Total flow of brine loss
            qb_loss = np.sum(data_c[ii_t, 7, :-1])  # Check qb_si + qb[ii_hb] = qo
            if qo <= qb_loss:
                qb_loss = qo
                mode = 'dp / Ql limited'

                # All the displaced brine is moved out of the channel, flow limited by oil flow
                # TODO: need to be refined
                print("All the displaced brine is moved out of the channel")
                # Amount fo brine expelled from channel
                # dvbs = 0
                #
                # qb_loss = qo

                # Amount of brine moved through channel
                # vb_i = np.zeros(N_layers)
                # vb_i[ii_ho:ii_hb + 1] = qb_i[ii_ho:ii_hb + 1] * delta_t_ho

            # delta_t_b = vb_i / qb_i
            # delta_t[ii_ho:ii_hb+1] = delta_t_b[ii_ho:ii_hb+1]
        else:
            qb_loss = 0

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

    else:
        if dL:
            # Check for brine displacement:
            qb_loss = np.sum(data_c[ii_t, 7, :-1])  # Check qb_si + qb[ii_hb] = qo
            if qb_loss > 0:
                mode = 'Ql'
                delta_t_ho = t_step
                dvbs = 0
            else:
                # The brine doesn't like to move it, move it
                mode = ''
                qb_loss = 0
                delta_t_ho = 0
                dvbs = 0

                STOP = True
        else:
            mode = ''

            # no brine is displaced laterally, no brine moves up. It is the end
            qb_loss = 0
            delta_t_ho = 0
            dvbs = 0
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
            ps_fp = os.path.join(run_fd, run_fn, ii_t_fn)
            plt.savefig(ps_fp, dpi=150)
        if DEBUG:
            plt.show()
        else:
            plt.close()
    dvb_loss = qb_loss * delta_t_ho
    dvo = qo * delta_t_ho

    # Volume of brine moving up = volume of oil - volume of brine expelled
    dvb = dvo - dvb_loss

    # move oil into element ii_ho
    kk_ho = ii_ho
    dvos = dvo
    while dvos > 0 and ii_ho < N_layers:
        vo_kk = dvos + ps_df[['V', 'f_o']].apply(lambda x: x[0] * x[1], axis=1)[kk_ho]
        f_o_ii = vo_kk / ps_df['V'][ii_ho]
        if f_o_ii - 1 > -epsilon:
            data_c[ii_t, 0, kk_ho] = 1
            kk_ho += 1
            ii_ho = kk_ho
            dvos = ps_df['V'][kk_ho] * (f_o_ii - 1)
        else:
            data_c[ii_t, 0, kk_ho] = f_o_ii
            dvos = 0
    ho = ps_df['h'].iloc[ii_ho]  + ps_df['dh'].iloc[ii_ho] * data_c[ii_t, 0, kk_ho]
    if dvo < 0 or dvos < -epsilon:
        print("Something is fishy with dvo dvos")
        STOP = True
        # OLD
        # vo_ii = dvo + ps_df[['V', 'f_o']].apply(lambda x: x[0] * x[1], axis=1)[ii_ho]
        # f_o_ii = vo_ii / ps_df['V'][ii_ho]
        # if np.abs(f_o_ii - 1) >= epsilon:
        #     print(ii_t, f_o_ii)
        # while np.abs((f_o_ii - 1)) < epsilon:
        #     data_c[ii_t, 0, ii_ho] = 1
        #     ii_ho += 1
        #     f_o_ii = f_o_ii - 1
        #
        # if np.abs(f_o_ii) > epsilon:
        #     data_c[ii_t, 0, ii_ho] = f_o_ii

        # New oil penetration depth

    dvbs = dvb
    kk_hb = ii_hb

    if data_c[ii_t, 0, kk_hb] > 0:
        print("ERROR Cannot move brine into a cell partially occupied by oil")
        STOP = True
    if kk_hb < N_layers-1:
        vb_kk = dvbs + ps_df[['V', 'f_b']].apply(lambda x: x[0] * x[1], axis=1)[kk_hb]
        f_b_kk = vb_kk / ps_df['V'][kk_hb]
        print(dvbs, hb, kk_hb, f_b_kk)
        while f_b_kk - 1 > -epsilon and ii_ho < N_layers-1:
            data_c[ii_t, 0, kk_hb] = -1
            kk_hb += 1
            ii_hb = kk_hb
            dvbs = ps_df['V'][kk_hb] * (f_b_kk - 1)
            f_b_kk = f_b_kk - 1
        print(dvbs, hb, kk_hb, f_b_kk)
        if ii_hb < N_layers-1:
            data_c[ii_t, 0, kk_hb] = -f_b_kk
            hb = ps_df['h'].iloc[ii_hb] + ps_df['dh'].iloc[ii_hb] * data_c[ii_t, 0, kk_hb]
            dvbs = 0
    else:
        hb = HI

    # Increment time
    t += delta_t_ho

    # Move oil around
    vo += dvo
    hr = HR / VR * (VR - vo)
    vr = HR / VR * hr

    # Move brine around
    vbs += dvbs
    vb_loss += dvb_loss
    vb = vb - dvbs - dvb_loss

    if np.abs(vb0 - vb - vbs - vb_loss) < epsilon:
        vb_check = 'OK'
    else:
        print('Some brine is disappearing')
        vb_check = 'X'
        STOP = True

    # Check if brine left in channel correspond to vb
    # vb_temp = data_c[ii_t, 0, :] * ps_df.V
    # vb_temp = -np.sum(vb_temp[vb_temp < 0])
    # if np.abs(vb_temp - vb) < epsilon:
    #     vb_check2 = 'OK'
    # else:
    #     print('Some brine is disappearing by magic, is it the angel share?')
    #     vb_check = 'X'
    #     STOP = True
    vb_check2 = 'x'
    vb_temp = 'x'
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
              (ii_ho, ii_t, t, delta_t_ho, ho, hb, delta_p, dvb_loss, dvo, dvb, dvbs, vb_check, mode)))
    data_p.append(_data_p)
    data_q.append(_data_q)
    data_bc.append([ii_t, t, delta_t_ho, ho, hb, vo, vb, vb_loss, dvb_loss, qo, qb_loss, hr, vr, dvo, dvb, vbs,
                    dvbs, p_0, pc_b_hb, delta_p])

    if all(data_c[ii_t, 7, :-1] == 0):
        print('p_si > p_c: no brine loss possible')

    if dvb < 1e-14 and dvb_loss < 1e-14 and ii_ho < 100 * ii_t :
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
        del delta_t_ho, dvb_loss, qo, qb_loss, dvo, dvb, dvbs, p_0, pc_b_hb, delta_p, vb_check, vb_check2, vb_temp

    if np.mod(ii_t, 500) == 0:
        data_bc_df = np.array(data_bc)
        data_df = pd.DataFrame(data_bc_df, columns=data_bc_headers)
        data_p_df = pd.DataFrame(data_p, columns=data_p_headers)
        data_q_df = pd.DataFrame(data_q, columns=data_q_headers)
        with open(pickle_fp, 'wb') as f:
            pickle.dump([data_df, ps_df, data_p_df, data_q_df, os_dict, data_c], f)

data_bc_df = np.array(data_bc)
data_df = pd.DataFrame(data_bc_df, columns=data_bc_headers)
data_p_df = pd.DataFrame(data_p, columns=data_p_headers)
data_q_df = pd.DataFrame(data_q, columns=data_q_headers)

print('R\t\tHR\t\tVR\t\tHB\t\tt\t\tho\t\tDp')
print(str('%.2f\t%.2f\t%.1e\t%.4f\t%.2f\t%.4f\t%.0f' % (R, HR, VR, HB, data_df.t.iloc[-1], data_df.ho.iloc[-1], data_df.delta_p.iloc[-1])))
plt.close()

## Double figure:
fig = plt.figure(figsize=(8, 6))
ax = fig.subplots(1, 2)

# Plot 1
ax[0].plot(data_df.t / 3600, data_df.ho, 'k-', label='$h_o$')
ax[0].plot(data_df.t / 3600, data_df.hb, 'b', label='$h_b+h_o$')
ax[0].plot(data_df.t / 3600, [HI - HB] * len(data_df), label='$H_i - H_b$')
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
y_lim_min = 10 ** np.floor(np.log10(min(y_lim)))
y_lim_max = 10 ** np.ceil(np.log10(max(y_lim)))

if np.ceil(np.log10(max(y_lim))) < np.floor(np.log10(min(y_lim))) + 3:
    y_lim_max = 10 ** np.floor(np.log10(min(y_lim) + 3))

if y_lim_max > 1e-6:
    y_lim_max = 1e-6

ax0.set_ylim([y_lim_min, y_lim_max])

ax[1].plot(data_p_df.ho, data_p_df.delta_p, 'k', label='$\Delta p_{tot}$')
ax[1].plot(data_p_df.ho, data_p_df.poc + data_p_df.poR, label='$p_o$')  # large
ax[1].plot(data_p_df.ho, data_p_df.pbc, label='$p_{o\\rightarrow b,c}$')
ax[1].plot(data_p_df.ho,
           data_p_df.poc + data_p_df.poR - data_p_df.pc_o + data_p_df.pc_b + data_p_df.p_qloss + data_p_df.pbc,
           label='$[p+]$')

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

if dL:
    if max(data_p_df.p_qloss) > 200:
        ax[1].plot(data_p_df.ho, data_p_df.p_qloss, label='$p_{q_{b,loss}}$')
        ax1_label = ax1_label + ', $p_{q_{b,loss}}$'
    else:
        ax1.plot(data_p_df.ho, data_p_df.p_qloss, ':', label='$p_{q_{b,loss}}$')
        ax2_label = ax2_label + ', $p_{q_{b,loss}}$'

y_lim = ax[1].get_ylim()
ax[1].set_ylim()
ax[1].set_xlim([0, data_p_df.ho.max()])

if HI - HB < ax[1].get_xlim()[-1]:
    ax[1].plot([HI - HB] * 2, y_lim, 'k--', alpha=0.5, label='$H_i - H_b$')
if HB < ax[1].get_xlim()[-1]:
    ax[1].plot([HB] * 2, y_lim, 'k-.', alpha=0.5, label='$H_D$')
if SI_STRAT:
    if HI - HG < ax[1].get_xlim()[-1]:
        ax[1].plot([HI - HG] * 2, y_lim, 'k:', alpha=0.5, label='$H_C$')

ax[1].set_xlabel('Penetration depth (m)')
ax[1].set_ylabel('Pressure (Pa): ' + ax1_label)
ax1.set_ylabel('Pressure (Pa): ' + ax2_label)

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
ps_fp = os.path.join(fig_dir, ho_fn)

plt.suptitle(str(
    '$\overline{R}$=%.1e, H$_R$=%.2f, V$_R$=%.2e, $\Delta$L=%.2f, $\\tau$: %s, TS:%s\n N$_{layer}$=%i, dt=%.2f ' %
    (ps_df.r[:-1].mean(), HR, VR, dL, str(TORT)[0], str(TORT)[0], N_layers, t_step)))
plt.tight_layout()
plt.savefig(ps_fp, dpi=300)
plt.show()

# Backup CONFIG
with open(pickle_fp, 'wb') as f:
    pickle.dump([data_df, ps_df, data_p_df, data_q_df, os_dict, data_c], f)
print(os.path.join(run_fd, run_fn))
