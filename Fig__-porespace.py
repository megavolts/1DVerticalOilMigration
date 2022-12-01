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

# Discretization
N_layers = 1000
t_step = 0.1  # Fallback t_step: large because flow is small

HIGH_RES = False
Nfig = 10

DEBUG = True
epsilon = 1e-12

# TODO change True to Enable or Disable
R = True # True to enable, if float
HR = 0.04
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
run_name = 'run_test_'
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
ps_fn = 'Fig_porespace.jpg'
ps_fp = os.path.join('/home/megavolts/UAF/paper/Chapter4/figures/', ps_fn)
plt.savefig(ps_fp, dpi=300)  # Save figures
pdf_fn = 'Fig_porespace.pdf'
pdf_fp = os.path.join('/home/megavolts/UAF/paper/Chapter4/figures/', pdf_fn)
plt.savefig(pdf_fp)  # Save figures
plt.show()  # Display Figures
